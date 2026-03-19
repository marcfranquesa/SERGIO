import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo

    return mo, np


@app.cell
def _(mo):
    mo.md("""
    # SERGIO Simulation
    Steady-state single-cell gene expression simulation.
    """)
    return


@app.cell
def _(np):
    import sys
    from pathlib import Path

    _repo = Path(__file__).parent
    sys.path.insert(0, str(_repo))
    from SERGIO.sergio import sergio

    _ds = _repo / "data_sets" / "De-noised_100G_9T_300cPerT_4_DS1"
    sim = sergio(
        number_genes=100,
        number_bins=9,
        number_sc=300,
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type="dpd",
    )
    sim.build_graph(
        input_file_taregts=str(_ds / "Interaction_cID_4.txt"),
        input_file_regs=str(_ds / "Regs_cID_4.txt"),
        shared_coop_state=2,
    )
    sim.simulate()
    expr = sim.getExpressions()
    data = np.concatenate(expr, axis=1)
    return (data,)


@app.cell
def _(data, mo, np):
    mo.md(f"""
    ## Results

    - **Shape**: {data.shape} (genes × cells)
    - **Mean expression**: {np.mean(data):.4f}
    - **Std**: {np.std(data):.4f}
    - **Min**: {np.min(data):.4f}
    - **Max**: {np.max(data):.4f}
    """)
    return


@app.cell
def _(data, np):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(data.flatten(), bins=50, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Expression level")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Expression distribution (all genes/cells)")

    gene_means = np.mean(data, axis=1)
    axes[1].bar(range(len(gene_means)), np.sort(gene_means)[::-1], color="salmon")
    axes[1].set_xlabel("Gene (ranked)")
    axes[1].set_ylabel("Mean expression")
    axes[1].set_title("Per-gene mean expression")

    plt.tight_layout()
    fig


if __name__ == "__main__":
    app.run()
