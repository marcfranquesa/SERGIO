"""
Simulation pipeline for the co-regulation benchmark.

Key design constraint: same numpy random seed before EVERY simulation to produce
exact counterfactual pairs.  For KD_Prog_n, genes NOT downstream of Program n will
have EXACTLY the same expression as in obs (same seed → same random draws →
identical trajectories).

Perturbations are implemented as *knockdowns* (rates reduced by 1 - KNOCKDOWN_FRACTION)
rather than full knockouts, to better reflect CRISPRi biology.

Each program's knockdown targets BOTH its activating and repressing master regulators
(TF_p_act and TF_p_rep), since both contribute to the observed expression pattern.
"""

import sys
from pathlib import Path

import numpy as np

# Add repo root to sys.path so SERGIO can be imported from any working directory
_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo))
from SERGIO.sergio import sergio  # noqa: E402

from co_reg_sim.grn import (  # noqa: E402
    N_BINS,
    N_GENES,
    N_GENES_PER_PROGRAM,
    N_TFS,
    TF_PAIR_IDS,
    ALL_TF_IDS,
    PROGRAM_GENE_IDS,
    generate_grn_files,
)

SEED             = 42
N_SC             = 150          # cells per bin
SAMPLING_STATE   = 5            # sampling_state for speed
NOISE_PARAMS     = 0.5
DECAYS           = 0.8
KNOCKDOWN_FRACTION = 0.1        # TF rates are reduced to this fraction (90% knockdown)


def run_single_simulation(interaction_file, regs_file, seed, perturb_program=None):
    """
    Run a single SERGIO simulation.

    Parameters
    ----------
    interaction_file : str
    regs_file : str
    seed : int
        Numpy random seed set BEFORE calling build_graph.
    perturb_program : int or None
        If not None, knock down BOTH TF_p_act and TF_p_rep for this program
        (CRISPRi-style: rates × KNOCKDOWN_FRACTION).

    Returns
    -------
    data : np.ndarray, shape (N_GENES, N_SC * N_BINS)
    sim  : sergio
    cell_type_labels : np.ndarray, shape (N_SC * N_BINS,)
    """
    np.random.seed(seed)

    sim = sergio(
        number_genes=N_GENES,
        number_bins=N_BINS,
        number_sc=N_SC,
        noise_params=NOISE_PARAMS,
        noise_type="sp",
        decays=DECAYS,
        sampling_state=SAMPLING_STATE,
    )

    sim.build_graph(
        input_file_taregts=interaction_file,
        input_file_regs=regs_file,
        shared_coop_state=2,
    )

    # Knock down both _act and _rep for the perturbed program
    if perturb_program is not None:
        act_id, rep_id = TF_PAIR_IDS[perturb_program]
        for tf_id in [act_id, rep_id]:
            sim.graph_[tf_id]["rates"] = [
                r * KNOCKDOWN_FRACTION for r in sim.graph_[tf_id]["rates"]
            ]

    sim.simulate()

    expr = sim.getExpressions()               # (N_BINS, N_GENES, N_SC)
    data = np.concatenate(expr, axis=1)       # (N_GENES, N_BINS * N_SC)
    cell_type_labels = np.repeat(np.arange(N_BINS), N_SC)

    return data, sim, cell_type_labels


def run_all_simulations(output_dir, force_recompute=False):
    """
    Run or load all simulations (obs + 4 knockdowns).

    Returns
    -------
    results : dict
        Keys: "obs", "kd_prog0" … "kd_prog3", "cell_types"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_file  = output_dir / "simulations.npz"
    meta_file = output_dir / "grn_metadata.npz"

    if sim_file.exists() and not force_recompute:
        print(f"Loading cached simulations from {sim_file}")
        data = np.load(sim_file)
        return {k: data[k] for k in data.files}

    print("Generating GRN files…")
    interaction_file, regs_file = generate_grn_files(output_dir)

    results = {}

    print("Running observational simulation…")
    obs_data, _, cell_types = run_single_simulation(
        interaction_file, regs_file, seed=SEED, perturb_program=None
    )
    results["obs"]        = obs_data
    results["cell_types"] = cell_types

    for prog in range(N_TFS):
        key = f"kd_prog{prog}"
        print(f"Running knockdown simulation for Program {prog} "
              f"(TF{prog}_act + TF{prog}_rep, {int((1 - KNOCKDOWN_FRACTION)*100)}% reduction)…")
        kd_data, _, _ = run_single_simulation(
            interaction_file, regs_file, seed=SEED, perturb_program=prog
        )
        results[key] = kd_data

    np.savez(sim_file, **results)
    print(f"Saved simulations to {sim_file}")

    # Save GRN metadata
    program_labels = np.full(N_GENES, -1, dtype=int)
    for p, gene_ids in PROGRAM_GENE_IDS.items():
        for gid in gene_ids:
            program_labels[gid] = p
    for tf_id in ALL_TF_IDS:
        program_labels[tf_id] = tf_id // 2   # TF0_act=0, TF0_rep=0, TF1_act=1, …

    np.savez(
        meta_file,
        all_tf_ids=np.array(ALL_TF_IDS),
        program_labels=program_labels,
        n_genes=np.array(N_GENES),
        n_bins=np.array(N_BINS),
        n_sc=np.array(N_SC),
        n_tfs=np.array(N_TFS),
        n_genes_per_program=np.array(N_GENES_PER_PROGRAM),
    )
    print(f"Saved GRN metadata to {meta_file}")

    return results
