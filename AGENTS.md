# SERGIO Usage Reference

SERGIO is a stochastic single-cell RNA-seq simulator based on the
Chemical Langevin Equation (CLE). It simulates steady-state or dynamic
gene expression across discrete cell types ("bins") given a gene regulatory
network (GRN).

## Package

```python
from SERGIO import sergio
```

## Core concepts

| Term | Meaning |
|------|---------|
| **bin** | A cell type / cell state. Cells in the same bin share the same GRN dynamics. |
| **master regulator (MR)** | A gene with no regulators; its production rate is set directly per bin in `Regs.txt`. |
| **target gene** | A gene whose production rate is computed via Hill functions from its regulators. |
| **K** | Interaction weight. `K > 0` = activation, `K < 0` = repression. This is NOT the Hill half-saturation constant. |
| **half_response** | SERGIO computes this dynamically as `mean(regulator_expression)` across all bins. You do not set it. |
| **shared_coop_state** | Hill coefficient `n` applied to all interactions. Typical value: 2. |

## Constructor

```python
sim = sergio(
    number_genes   = N_GENES,       # total genes (MRs + targets)
    number_bins    = N_BINS,        # number of cell types
    number_sc      = N_SC,          # cells to sample per bin
    noise_params   = 0.5,           # CLE noise amplitude (scalar or per-gene array)
    noise_type     = "sp",          # "sp" | "spd" | "dpd"
    decays         = 0.8,           # mRNA decay rate (scalar or per-gene array)
    sampling_state = 5,             # sample from last sampling_state * N_SC steps
)
```

`noise_type="sp"` (single production noise) is the standard choice for synthetic benchmarks.

## Preferred: build GRN from Python data structures

```python
sim.build_graph_from_arrays(
    master_regulators = {
        # gene_id: [rate_bin0, rate_bin1, ..., rate_binN]
        0: [8.0, 8.0, 7.0, 1.0, 1.0, 1.0],   # activating TF: high in bins 0-2
        1: [1.0, 1.0, 1.0, 8.0, 8.0, 7.0],   # repressing TF: high in bins 3-5
    },
    interactions = {
        # target_id: [(reg_id, K), ...]   K>0 activates, K<0 represses
        2: [(0, 5.0), (1, -5.0)],
        3: [(0, 5.0), (1, -5.0)],
    },
    shared_coop_state = 2,
)
```

## Alternative: build GRN from CSV files

```python
sim.build_graph(
    input_file_targets = "Interaction.txt",   # one row per target gene
    input_file_regs    = "Regs.txt",          # one row per master regulator
    shared_coop_state  = 2,
)
```

**Interaction.txt format** (one row per target):
```
target_id, n_regs, reg1_id, reg2_id, K1, K2
```
All regulator IDs come first, then all K values — NOT interleaved.

**Regs.txt format** (one row per master regulator):
```
reg_id, rate_bin0, rate_bin1, ..., rate_binN
```

## Run simulation and get expression

```python
np.random.seed(42)   # set before simulate() for reproducibility
sim.simulate()

expr = sim.getExpressions()
# shape: (N_BINS, N_GENES, N_SC)
# to get (N_GENES, N_BINS * N_SC):
import numpy as np
data = np.concatenate(expr, axis=1)
cell_type_labels = np.repeat(np.arange(N_BINS), N_SC)
```

## Applying perturbations (knockdowns)

Modify `sim.graph_[gene_id]["rates"]` **after** `build_graph` and **before** `simulate()`:

```python
KNOCKDOWN_FRACTION = 0.1   # 90% CRISPRi-style knockdown

sim.build_graph_from_arrays(master_regulators, interactions)

# KD gene 0
sim.graph_[0]["rates"] = [r * KNOCKDOWN_FRACTION for r in sim.graph_[0]["rates"]]

sim.simulate()
```

For exact counterfactuals (non-downstream genes identical between obs and KD),
use the **same `np.random.seed()` before every `simulate()` call**.

## Typical benchmark pattern

```python
import numpy as np
from SERGIO import sergio

SEED = 42
N_GENES, N_BINS, N_SC = 48, 9, 150

def run(master_regulators, interactions, kd_gene_ids=None):
    sim = sergio(
        number_genes=N_GENES, number_bins=N_BINS, number_sc=N_SC,
        noise_params=0.5, noise_type="sp", decays=0.8, sampling_state=5,
    )
    sim.build_graph_from_arrays(master_regulators, interactions, shared_coop_state=2)
    if kd_gene_ids:
        for gid in kd_gene_ids:
            sim.graph_[gid]["rates"] = [r * 0.1 for r in sim.graph_[gid]["rates"]]
    np.random.seed(SEED)
    sim.simulate()
    expr = sim.getExpressions()               # (N_BINS, N_GENES, N_SC)
    data = np.concatenate(expr, axis=1)       # (N_GENES, N_BINS * N_SC)
    cell_types = np.repeat(np.arange(N_BINS), N_SC)
    return data, cell_types

obs, cell_types = run(master_regulators, interactions)
kd0, _          = run(master_regulators, interactions, kd_gene_ids=[0, 1])
```

## Key gotchas

- Gene IDs must be 0-indexed contiguous integers from 0 to `number_genes - 1`.
- Every gene must be either a master regulator (in `Regs.txt`) or a target (in `Interaction.txt`). A gene that appears in neither causes a silent error.
- A gene cannot be both a master regulator and a target. Putting an MR in `Interaction.txt` raises a `ValueError`.
- `half_response` is computed as the mean expression of the regulator across all bins — it is not a free parameter.
- `sampling_state` controls how many steady-state steps to sample from. Higher = less autocorrelation between sampled cells but slower simulation. 5 is a reasonable default for benchmarks.
