import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import NMF, PCA

    import marimo as mo

    return (
        NMF,
        PCA,
        Path,
        mo,
        np,
        plt,
        sys,
    )


@app.cell
def _(Path, sys):
    # Compute paths relative to this file and add repo root to sys.path
    _notebook_dir = Path(__file__).parent
    _repo_root = _notebook_dir.parent

    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    # Import our modules using absolute paths
    import importlib.util as _ilu

    def _load_module(name, path):
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    grn = _load_module("co_reg_sim.grn", str(_notebook_dir / "grn.py"))
    simulate = _load_module("co_reg_sim.simulate", str(_notebook_dir / "simulate.py"))

    OUTPUT_DIR = _notebook_dir / "output"

    return OUTPUT_DIR, grn, simulate


@app.cell
def _(OUTPUT_DIR, simulate):
    # Run or load all simulations
    results = simulate.run_all_simulations(str(OUTPUT_DIR), force_recompute=False)

    obs = results["obs"]            # (44, 1350)
    ko_tf0 = results["ko_tf0"]     # (44, 1350)
    ko_tf1 = results["ko_tf1"]
    ko_tf2 = results["ko_tf2"]
    ko_tf3 = results["ko_tf3"]
    cell_types = results["cell_types"]  # (1350,)

    return cell_types, ko_tf0, ko_tf1, ko_tf2, ko_tf3, obs, results


@app.cell
def _(mo):
    return mo.md("# SERGIO Co-regulation Benchmark")



@app.cell
def _(grn, plt):
    # GRN structure: TF production rates heatmap (4 TFs x 9 bins)
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _im = _ax.imshow(
        grn.TF_RATES,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    _ax.set_xticks(range(grn.N_BINS))
    _ax.set_xticklabels([f"Bin {b}" for b in range(grn.N_BINS)], rotation=45, ha="right")
    _ax.set_yticks(range(grn.N_TFS))
    _ax.set_yticklabels([f"TF{t}" for t in grn.TF_IDS])
    _ax.set_title("TF Production Rates per Cell Type (Bin)")
    _ax.set_xlabel("Cell type (bin)")
    _ax.set_ylabel("Transcription factor")
    _fig.colorbar(_im, ax=_ax, label="Production rate")
    _fig.tight_layout()
    _fig


@app.cell
def _(cell_types, grn, np, obs, plt):
    # Observational expression heatmap: genes x cells, sorted by cell type
    _sort_idx = np.argsort(cell_types)
    _obs_sorted = obs[:, _sort_idx]

    _fig, _ax = plt.subplots(figsize=(14, 6))
    _im = _ax.imshow(
        np.log1p(_obs_sorted),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    _fig.colorbar(_im, ax=_ax, label="log(1 + expression)")

    # Mark program boundaries (horizontal lines between gene groups)
    _boundaries = [grn.N_TFS]  # after TFs
    for _p in range(grn.N_TFS):
        _boundaries.append(_boundaries[-1] + grn.N_GENES_PER_PROGRAM)
    for _b in _boundaries[:-1]:
        _ax.axhline(_b - 0.5, color="white", linewidth=1.5, linestyle="--")

    # Mark cell type boundaries (vertical lines)
    _n_cells = obs.shape[1]
    _cells_per_bin = _n_cells // grn.N_BINS
    for _b in range(1, grn.N_BINS):
        _ax.axvline(_b * _cells_per_bin - 0.5, color="red", linewidth=0.8, alpha=0.6)

    _ax.set_title("Observational Expression Heatmap (log scale, cells sorted by type)")
    _ax.set_xlabel("Cells (sorted by cell type)")
    _ax.set_ylabel("Genes")

    # Y-tick labels for program groups
    _ytick_positions = [grn.N_TFS // 2]
    _ytick_labels = ["TFs"]
    for _p in range(grn.N_TFS):
        _center = grn.N_TFS + _p * grn.N_GENES_PER_PROGRAM + grn.N_GENES_PER_PROGRAM // 2
        _ytick_positions.append(_center)
        _ytick_labels.append(f"Prog {_p}")
    _ax.set_yticks(_ytick_positions)
    _ax.set_yticklabels(_ytick_labels)

    _fig.tight_layout()
    _fig


@app.cell
def _(PCA, cell_types, grn, np, obs, plt):
    # Observational PCA: PC1 vs PC2 colored by cell type
    _pca = PCA(n_components=2)
    _coords = _pca.fit_transform(obs.T)  # (N_cells, 2)

    _cmap = plt.get_cmap("tab10")
    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _b in range(grn.N_BINS):
        _mask = cell_types == _b
        _ax.scatter(
            _coords[_mask, 0],
            _coords[_mask, 1],
            c=[_cmap(_b)],
            label=f"Bin {_b}",
            alpha=0.6,
            s=20,
        )
    _ax.set_xlabel(f"PC1 ({_pca.explained_variance_ratio_[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({_pca.explained_variance_ratio_[1]*100:.1f}%)")
    _ax.set_title("Observational PCA (colored by cell type)")
    _ax.legend(loc="best", ncol=3, fontsize=8)
    _fig.tight_layout()
    _fig


@app.cell
def _(NMF, grn, np, obs, plt):
    # Observational NMF (4 components): gene scores heatmap
    _nmf_obs = NMF(n_components=grn.N_TFS, random_state=0, max_iter=500)
    # obs is (N_GENES, N_cells); NMF expects (samples, features) -> transpose
    _W_obs = _nmf_obs.fit_transform(obs.T)  # (N_cells, 4)
    _H_obs = _nmf_obs.components_             # (4, N_GENES)

    _H_obs_norm = _H_obs / (_H_obs.max(axis=1, keepdims=True) + 1e-10)

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _im = _ax.imshow(
        _H_obs_norm,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )
    _fig.colorbar(_im, ax=_ax, label="NMF gene loading (row-normalised)")

    # Vertical lines marking program boundaries
    _boundaries = [grn.N_TFS]
    for _p in range(grn.N_TFS):
        _boundaries.append(_boundaries[-1] + grn.N_GENES_PER_PROGRAM)
    for _b in _boundaries[:-1]:
        _ax.axvline(_b - 0.5, color="cyan", linewidth=1.5, linestyle="--")

    _ax.set_title("Observational NMF Gene Scores (4 components)\nNote: components likely do NOT align with true programs")
    _ax.set_xlabel("Genes")
    _ax.set_ylabel("NMF component")
    _ax.set_yticks(range(grn.N_TFS))
    _ax.set_yticklabels([f"NMF-{i}" for i in range(grn.N_TFS)])
    _fig.tight_layout()
    _fig


@app.cell
def _(cell_types, grn, ko_tf0, ko_tf1, ko_tf2, ko_tf3, np, obs, plt):
    # Per-perturbation delta heatmaps: mean delta per gene per cell type
    _ko_list = [ko_tf0, ko_tf1, ko_tf2, ko_tf3]
    _fig, _axes = plt.subplots(1, grn.N_TFS, figsize=(14, 5), sharey=True)

    for _ti, (_ax, _ko) in enumerate(zip(_axes, _ko_list)):
        # Compute mean delta per gene per bin
        _delta_mean = np.zeros((grn.N_GENES, grn.N_BINS))
        for _b in range(grn.N_BINS):
            _mask = cell_types == _b
            _delta_mean[:, _b] = np.mean(obs[:, _mask] - _ko[:, _mask], axis=1)

        _im = _ax.imshow(
            _delta_mean,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-np.percentile(np.abs(_delta_mean), 99),
            vmax=np.percentile(np.abs(_delta_mean), 99),
        )
        _fig.colorbar(_im, ax=_ax, label="Mean delta")

        # Horizontal lines marking program boundaries
        _boundaries = [grn.N_TFS]
        for _p in range(grn.N_TFS):
            _boundaries.append(_boundaries[-1] + grn.N_GENES_PER_PROGRAM)
        for _b_line in _boundaries[:-1]:
            _ax.axhline(_b_line - 0.5, color="black", linewidth=1.0, linestyle="--")

        _ax.set_title(f"KO TF{_ti}: mean(obs - ko)")
        _ax.set_xlabel("Cell type (bin)")
        if _ti == 0:
            _ax.set_ylabel("Genes")

    _fig.suptitle("Per-Perturbation Delta Heatmaps\n(non-zero only for program genes downstream of knocked-out TF)", y=1.02)
    _fig.tight_layout()
    _fig


@app.cell
def _(cell_types, grn, ko_tf0, np, obs, plt):
    # Counterfactual validation: 2-panel scatter obs vs KO_TF0 expression
    # Panel 1: all cells, log1p scale; Panel 2: active bins 0-2 only, log1p scale
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 7))

    _alpha = 0.15
    _s = 4
    _active_bins_mask = np.isin(cell_types, [0, 1, 2])  # bins where TF0 is high

    for _panel_idx, (_ax, _mask) in enumerate([(_axes[0], np.ones(obs.shape[1], dtype=bool)),
                                                (_axes[1], _active_bins_mask)]):
        _obs_sub = obs[:, _mask]
        _ko_sub  = ko_tf0[:, _mask]

        for _p in [1, 2, 3]:
            _gids = grn.PROGRAM_GENE_IDS[_p]
            _x = np.log1p(_obs_sub[_gids].ravel())
            _y = np.log1p(_ko_sub[_gids].ravel())
            _ax.scatter(_x, _y, c="steelblue", alpha=_alpha, s=_s,
                        label="Programs 1/2/3 (should be on y=x)" if _p == 1 else None)

        _gids0 = grn.PROGRAM_GENE_IDS[0]
        _x0 = np.log1p(_obs_sub[_gids0].ravel())
        _y0 = np.log1p(_ko_sub[_gids0].ravel())
        _ax.scatter(_x0, _y0, c="crimson", alpha=_alpha * 2, s=_s * 2,
                    label="Program 0 (KO target)")

        _x_tfs = np.log1p(_obs_sub[grn.TF_IDS].ravel())
        _y_tfs = np.log1p(_ko_sub[grn.TF_IDS].ravel())
        _ax.scatter(_x_tfs, _y_tfs, c="gray", alpha=0.4, s=_s * 2, label="TFs")

        _lim = max(np.log1p(obs.max()), np.log1p(ko_tf0.max()))
        _ax.plot([0, _lim], [0, _lim], "k--", linewidth=1, label="y=x (exact counterfactual)")
        _ax.set_xlabel("log1p(Observational expression)")
        _ax.set_ylabel("log1p(KO TF0 expression)")
        _ax.legend(loc="upper left", fontsize=9, markerscale=3)

    _axes[0].set_title("Panel 1: All cells (log1p scale)\nBlue→ y=x;  Red→ below y=x")
    _axes[1].set_title("Panel 2: Active bins 0-2 only (log1p scale)\nDeviation of Program 0 more visible")
    _fig.suptitle("Counterfactual Validation: obs vs KO_TF0", fontsize=12)
    _fig.tight_layout()
    _fig


@app.cell
def _(grn, ko_tf0, ko_tf1, ko_tf2, ko_tf3, np, obs, plt):
    # Counterfactual residual validation:
    # For KO_TFk, genes NOT in Program k should have EXACTLY zero residual (same random seed).
    # This validates the exact counterfactual guarantee.
    _ko_list_v = [ko_tf0, ko_tf1, ko_tf2, ko_tf3]
    _fig, _axes = plt.subplots(1, grn.N_TFS, figsize=(14, 4), sharey=True)

    for _ti, (_ax, _ko) in enumerate(zip(_axes, _ko_list_v)):
        _delta = obs - _ko  # (44, 1350)
        _mean_abs_delta = np.mean(np.abs(_delta), axis=1)  # per-gene mean |delta|

        _colors = []
        for _g in range(grn.N_GENES):
            if _g in grn.TF_IDS:
                _colors.append("gray")
            elif _g in grn.PROGRAM_GENE_IDS[_ti]:
                _colors.append("crimson")
            else:
                _colors.append("steelblue")

        _ax.bar(range(grn.N_GENES), _mean_abs_delta, color=_colors, width=1.0, linewidth=0)
        _ax.set_title(f"KO TF{_ti}: mean|obs - ko| per gene")
        _ax.set_xlabel("Gene index")
        if _ti == 0:
            _ax.set_ylabel("Mean |delta|")

        # Mark program boundaries
        for _b_line in [grn.N_TFS] + [grn.N_TFS + _p * grn.N_GENES_PER_PROGRAM for _p in range(grn.N_TFS - 1)]:
            _ax.axvline(_b_line - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)

    _fig.suptitle(
        "Residual per Gene per Perturbation\n"
        "Red = program genes (should be non-zero), Blue = others (should be exactly 0), Gray = TFs",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig


@app.cell
def _(NMF, grn, ko_tf0, ko_tf1, ko_tf2, ko_tf3, np, obs, plt):
    # Perturbation NMF: stack all |delta| matrices, apply NMF with 4 components
    _ko_list_pert = [ko_tf0, ko_tf1, ko_tf2, ko_tf3]
    # Use positive part: genes downstream of KO have obs > ko (reduced expression after KO)
    # np.maximum clips noise near zero cleanly; np.abs would inflate noise floor
    _deltas = [np.maximum(obs - _ko, 0.0) for _ko in _ko_list_pert]  # list of (44, 1350)

    # Stack horizontally: (44, 4 * 1350)
    _stacked = np.concatenate(_deltas, axis=1)  # (N_GENES, 4 * N_cells)

    _nmf_pert = NMF(n_components=grn.N_TFS, random_state=0, max_iter=1000)
    _W_pert = _nmf_pert.fit_transform(_stacked.T)   # (4*N_cells, 4)
    _H_pert = _nmf_pert.components_                  # (4, N_GENES)

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _im = _ax.imshow(
        _H_pert,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )
    _fig.colorbar(_im, ax=_ax, label="NMF gene loading")

    # Vertical lines marking program boundaries
    _boundaries_p = [grn.N_TFS]
    for _p in range(grn.N_TFS):
        _boundaries_p.append(_boundaries_p[-1] + grn.N_GENES_PER_PROGRAM)
    for _b_line in _boundaries_p[:-1]:
        _ax.axvline(_b_line - 0.5, color="cyan", linewidth=1.5, linestyle="--")

    _ax.set_title("Perturbation NMF Gene Loadings (4 components on stacked |delta|)\nShould recover true co-regulation programs")
    _ax.set_xlabel("Genes")
    _ax.set_ylabel("NMF component")
    _ax.set_yticks(range(grn.N_TFS))
    _ax.set_yticklabels([f"NMF-{i}" for i in range(grn.N_TFS)])
    _fig.tight_layout()
    _fig


@app.cell
def _(NMF, grn, ko_tf0, ko_tf1, ko_tf2, ko_tf3, np, obs, plt):
    # Program comparison summary
    # True programs: top-10 genes per program (by definition)
    # Obs NMF: top-10 genes per component
    # Perturbation NMF: top-10 genes per component

    _ko_list_cmp = [ko_tf0, ko_tf1, ko_tf2, ko_tf3]

    # --- Re-fit obs NMF ---
    _nmf_obs_cmp = NMF(n_components=grn.N_TFS, random_state=0, max_iter=500)
    _H_obs_cmp = _nmf_obs_cmp.fit_transform(obs.T).T   # actually get H via components_
    _H_obs_cmp = _nmf_obs_cmp.components_               # (4, N_GENES)

    # --- Re-fit perturbation NMF ---
    _deltas_cmp = [np.maximum(obs - _ko, 0.0) for _ko in _ko_list_cmp]
    _stacked_cmp = np.concatenate(_deltas_cmp, axis=1)
    _nmf_pert_cmp = NMF(n_components=grn.N_TFS, random_state=0, max_iter=1000)
    _nmf_pert_cmp.fit(_stacked_cmp.T)
    _H_pert_cmp = _nmf_pert_cmp.components_             # (4, N_GENES)

    TOP_K = grn.N_GENES_PER_PROGRAM + 1  # 11 (10 targets + 1 TF)

    def _top_genes(H, k):
        """Return top-k gene indices per component (row), as list of sets."""
        return [set(np.argsort(H[i])[::-1][:k]) for i in range(H.shape[0])]

    # True program gene sets (target genes + TF gene)
    _true_sets = [set(grn.PROGRAM_GENE_IDS[p]) | {p} for p in range(grn.N_TFS)]

    _obs_top = _top_genes(_H_obs_cmp, TOP_K)
    _pert_top = _top_genes(_H_pert_cmp, TOP_K)

    def _jaccard_matrix(true_sets, pred_sets):
        """Returns J[i,j] = Jaccard(true_sets[i], pred_sets[j]). max(axis=1) = best pred per true."""
        J = np.zeros((len(true_sets), len(pred_sets)))
        for i, ts in enumerate(true_sets):
            for j, ps in enumerate(pred_sets):
                inter = len(ts & ps)
                union = len(ts | ps)
                J[i, j] = inter / union if union > 0 else 0.0
        return J

    _J_obs = _jaccard_matrix(_true_sets, _obs_top)
    _J_pert = _jaccard_matrix(_true_sets, _pert_top)

    # Best Jaccard per true program (greedy match)
    _best_obs = _J_obs.max(axis=1)
    _best_pert = _J_pert.max(axis=1)

    # Print summary
    print("Jaccard scores (true program vs best-matching NMF component):")
    print(f"{'Program':<10} {'Obs NMF':>10} {'Pert NMF':>10}")
    for _p in range(grn.N_TFS):
        print(f"Program {_p}  {_best_obs[_p]:>10.3f} {_best_pert[_p]:>10.3f}")

    # --- Visualization ---
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) True program assignment: binary matrix (programs x genes)
    _true_matrix = np.zeros((grn.N_TFS, grn.N_GENES))
    for _p in range(grn.N_TFS):
        for _gid in grn.PROGRAM_GENE_IDS[_p]:
            _true_matrix[_p, _gid] = 1.0

    _axes[0].imshow(_true_matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    for _b_line in [grn.N_TFS] + [grn.N_TFS + _p * grn.N_GENES_PER_PROGRAM for _p in range(grn.N_TFS)][:-1]:
        _axes[0].axvline(_b_line - 0.5, color="red", linewidth=1.5, linestyle="--")
    _axes[0].set_title("(a) True Program Assignments")
    _axes[0].set_xlabel("Gene index")
    _axes[0].set_ylabel("Program")
    _axes[0].set_yticks(range(grn.N_TFS))
    _axes[0].set_yticklabels([f"Prog {p}" for p in range(grn.N_TFS)])

    # (b) Obs NMF top genes: binary matrix (components x genes)
    _obs_matrix = np.zeros((grn.N_TFS, grn.N_GENES))
    for _ci, _gset in enumerate(_obs_top):
        for _gid in _gset:
            _obs_matrix[_ci, _gid] = 1.0
    _axes[1].imshow(_obs_matrix, aspect="auto", cmap="Oranges", interpolation="nearest")
    for _b_line in [grn.N_TFS] + [grn.N_TFS + _p * grn.N_GENES_PER_PROGRAM for _p in range(grn.N_TFS)][:-1]:
        _axes[1].axvline(_b_line - 0.5, color="red", linewidth=1.5, linestyle="--")
    _axes[1].set_title(f"(b) Obs NMF Top-{TOP_K} Genes\nMean Jaccard: {_best_obs.mean():.3f}")
    _axes[1].set_xlabel("Gene index")
    _axes[1].set_yticks(range(grn.N_TFS))
    _axes[1].set_yticklabels([f"NMF-{i}" for i in range(grn.N_TFS)])

    # (c) Perturbation NMF top genes
    _pert_matrix = np.zeros((grn.N_TFS, grn.N_GENES))
    for _ci, _gset in enumerate(_pert_top):
        for _gid in _gset:
            _pert_matrix[_ci, _gid] = 1.0
    _axes[2].imshow(_pert_matrix, aspect="auto", cmap="Greens", interpolation="nearest")
    for _b_line in [grn.N_TFS] + [grn.N_TFS + _p * grn.N_GENES_PER_PROGRAM for _p in range(grn.N_TFS)][:-1]:
        _axes[2].axvline(_b_line - 0.5, color="red", linewidth=1.5, linestyle="--")
    _axes[2].set_title(f"(c) Perturbation NMF Top-{TOP_K} Genes\nMean Jaccard: {_best_pert.mean():.3f}")
    _axes[2].set_xlabel("Gene index")
    _axes[2].set_yticks(range(grn.N_TFS))
    _axes[2].set_yticklabels([f"NMF-{i}" for i in range(grn.N_TFS)])

    _fig.suptitle("Program Recovery Comparison\n(red dashed lines = program boundaries)", fontsize=12)
    _fig.tight_layout()
    _fig


if __name__ == "__main__":
    app.run()
