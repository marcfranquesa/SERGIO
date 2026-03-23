"""
GRN data structure and file generator for the co-regulation simulation benchmark.

Biological framing
------------------
Each program models a "lineage toggle module": a gene set that is actively turned ON in one
set of cell types (by TF_p_act) and actively turned OFF in complementary cell types (by
TF_p_rep).  Both TFs are anti-correlated across cell types, analogous to e.g. the GATA1 /
PRC2-repressor balance that toggles erythroid vs. myeloid targets across hematopoietic states.
KD of both TFs simultaneously disrupts both arms of the toggle, producing opposite-sign deltas
in the two cell-type groups — a cell-type-heterogeneous response for the SAME target gene.

Gene layout
-----------
  Master regulators — 8 genes total (2 per program):

    Gene 0 = TF0_act  activates Program 0  — high production in bins 0-2
    Gene 1 = TF0_rep  represses Program 0  — high production in bins 3-5
    Gene 2 = TF1_act  activates Program 1  — nearly identical to TF0_act (confounded)
    Gene 3 = TF1_rep  represses Program 1  — nearly identical to TF0_rep (confounded)
    Gene 4 = TF2_act  activates Program 2  — high production in bins 3-5
    Gene 5 = TF2_rep  represses Program 2  — high production in bins 6-8
    Gene 6 = TF3_act  activates Program 3  — nearly identical to TF2_act (confounded)
    Gene 7 = TF3_rep  represses Program 3  — nearly identical to TF2_rep (confounded)

  Program targets:
    Program 0: genes  8-17
    Program 1: genes 18-27
    Program 2: genes 28-37
    Program 3: genes 38-47

Regulation model
----------------
Every target gene in program p is regulated by BOTH TF_p_act (weight K_ACT = +5)
and TF_p_rep (weight K_REP = −5).  Note: K is the SERGIO interaction weight, not the
Hill half-saturation constant.  SERGIO computes the actual Hill threshold (half_response)
dynamically as the mean TF expression across all bins; K scales the contribution.

This dual regulation creates a cell-type-heterogeneous response for the SAME gene:

  • Bins where TF_p_act is high: activation dominates → expression HIGH.
    KD both TFs → activation drops → Δ = obs − kd > 0.

  • Bins where TF_p_rep is high: repression dominates → expression LOW.
    KD both TFs → repression lifted → expression rises → Δ = obs − kd < 0.

Consequence:
  • Per-cell |Δ| is LARGE in both activation and repression bins, giving a distinctive
    per-program signature that sample-level NMF recovers.
  • ATE = mean(Δ) is NEAR ZERO per gene for programs with symmetric bin coverage
    (positive and negative cell contributions approximately cancel).  Program 0 achieves
    mean|ATE| ≈ 0.027; Programs 1–3 show mean|ATE| ≈ 0.11–0.12 due to stochastic
    asymmetries in SERGIO's CLE dynamics that break exact cancellation.

Confounding design
------------------
  • TF0_act ≈ TF1_act  and  TF0_rep ≈ TF1_rep  →  P0 and P1 are confounded.
    (Same active bins 0-2 / 3-5; only within-bin rate ordering differs: [8,8,7] vs [7,8,8].)
  • TF2_act ≈ TF3_act  and  TF2_rep ≈ TF3_rep  →  P2 and P3 are confounded.
  • Programs 0+1 are anti-correlated with 2+3 across cell types.
"""

from pathlib import Path

import numpy as np

N_TFS              = 4                                # number of gene programs
N_TF_GENES         = 2 * N_TFS                       # 8 master-regulator genes
N_GENES_PER_PROGRAM = 10
N_BINS             = 9
N_GENES            = N_TF_GENES + N_TFS * N_GENES_PER_PROGRAM  # 48

# Per-program TF gene-pair indices  (act_id, rep_id)
TF_PAIR_IDS = [(2 * p, 2 * p + 1) for p in range(N_TFS)]  # (0,1),(2,3),(4,5),(6,7)
TF_ACT_IDS  = [pair[0] for pair in TF_PAIR_IDS]            # [0, 2, 4, 6]
TF_REP_IDS  = [pair[1] for pair in TF_PAIR_IDS]            # [1, 3, 5, 7]
ALL_TF_IDS  = list(range(N_TF_GENES))                      # [0 … 7]

PROGRAM_GENE_IDS = {
    p: list(range(
        N_TF_GENES + p * N_GENES_PER_PROGRAM,
        N_TF_GENES + (p + 1) * N_GENES_PER_PROGRAM,
    ))
    for p in range(N_TFS)
}

K_ACT = 5.0    # Interaction weight for activation (SERGIO convention: |K| × Hill-function value)
K_REP = -5.0   # Interaction weight for repression (negative sign → repressive Hill; SERGIO uses |K| × (1-Hill))

# Production rates of master-regulator genes, shape (N_TF_GENES, N_BINS)
# Designed so that:
#   TF0_act ≈ TF1_act  (high bins 0-2)
#   TF0_rep ≈ TF1_rep  (high bins 3-5)  →  P0 and P1 are observationally confounded
#   TF2_act ≈ TF3_act  (high bins 3-5)
#   TF2_rep ≈ TF3_rep  (high bins 6-8)  →  P2 and P3 are observationally confounded
#   (P0+P1) anti-correlated with (P2+P3)
TF_RATES = np.array([
    # ── Program 0 ─────────────────────────────────────────────────
    # 3 activation bins (0-2) + 3 repression bins (3-5) — symmetric rates → ATE ≈ 0
    [8, 8, 7, 1, 1, 1, 1, 1, 1],   # gene 0 = TF0_act  (high bins 0-2)
    [1, 1, 1, 8, 8, 7, 1, 1, 1],   # gene 1 = TF0_rep  (high bins 3-5)
    # ── Program 1  (≈ P0 → confounded with P0) ───────────────────
    # Same active bins as P0, only within-bin rate order differs → observationally confounded
    [7, 8, 8, 1, 1, 1, 1, 1, 1],   # gene 2 = TF1_act  (≈ TF0_act, rate order 7,8,8)
    [1, 1, 1, 7, 8, 8, 1, 1, 1],   # gene 3 = TF1_rep  (≈ TF0_rep, rate order 7,8,8)
    # ── Program 2 ─────────────────────────────────────────────────
    # 3 activation bins (3-5) + 3 repression bins (6-8) — symmetric rates → ATE ≈ 0
    [1, 1, 1, 8, 8, 7, 1, 1, 1],   # gene 4 = TF2_act  (high bins 3-5)
    [1, 1, 1, 1, 1, 1, 8, 8, 7],   # gene 5 = TF2_rep  (high bins 6-8)
    # ── Program 3  (≈ P2 → confounded with P2) ───────────────────
    # Same active bins as P2, only within-bin rate order differs → observationally confounded
    [1, 1, 1, 7, 8, 8, 1, 1, 1],   # gene 6 = TF3_act  (≈ TF2_act, rate order 7,8,8)
    [1, 1, 1, 1, 1, 1, 7, 8, 8],   # gene 7 = TF3_rep  (≈ TF2_rep, rate order 7,8,8)
])


def generate_grn_files(output_dir):
    """
    Write Interaction.txt and Regs.txt in SERGIO format to output_dir.

    Each target gene has 2 regulators: TF_p_act (K_ACT) and TF_p_rep (K_REP).
    All 8 master regulators are written to Regs.txt.

    Returns
    -------
    interaction_file, regs_file : str paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_file = output_dir / "Interaction.txt"
    regs_file        = output_dir / "Regs.txt"

    # Interaction.txt — each target gene: 2 regulators (_act activates, _rep represses)
    # SERGIO format: target_id, n_regs, reg1_id, reg2_id, K1, K2
    # (all regulator IDs first, then all K values — NOT interleaved)
    with open(interaction_file, "w") as f:
        for p in range(N_TFS):
            act_id, rep_id = TF_PAIR_IDS[p]
            for gene_id in PROGRAM_GENE_IDS[p]:
                f.write(f"{gene_id},2,{act_id},{rep_id},{K_ACT},{K_REP}\n")

    # Regs.txt — all 8 master regulators
    with open(regs_file, "w") as f:
        for tf_idx, tf_id in enumerate(ALL_TF_IDS):
            rates_str = ",".join(str(r) for r in TF_RATES[tf_idx])
            f.write(f"{tf_id},{rates_str}\n")

    return str(interaction_file), str(regs_file)
