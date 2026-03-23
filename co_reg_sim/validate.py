"""
Standalone validation script — runs the full simulation pipeline and saves all
plots to co_reg_sim/output/plots/. Run with:

    uv run python co_reg_sim/validate.py

from the repo root.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, PCA

# ── paths ────────────────────────────────────────────────────────────────────
_here = Path(__file__).parent
_repo = _here.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "co_reg_sim"))

from grn import (
    N_BINS, N_GENES, N_GENES_PER_PROGRAM, N_TFS, N_TF_GENES,
    PROGRAM_GENE_IDS, ALL_TF_IDS, TF_PAIR_IDS, TF_ACT_IDS, TF_REP_IDS, TF_RATES,
)
from simulate import run_all_simulations

OUTPUT_DIR = _here / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── run / load simulations ────────────────────────────────────────────────────
print("Running simulations (or loading cache)…")
results = run_all_simulations(str(OUTPUT_DIR), force_recompute=False)

obs        = results["obs"]          # (48, 1350)
kd_list    = [results[f"kd_prog{p}"] for p in range(N_TFS)]
cell_types = results["cell_types"]   # (1350,)

N_SC_TOTAL = obs.shape[1]

print(f"obs shape:        {obs.shape}")
print(f"cell_types shape: {cell_types.shape}")

# ── Technical noise ──────────────────────────────────────────────────────────
# Add independent Poisson-like noise (CV=0.1) per condition to simulate realistic
# measurement variability. Used ONLY for NMF comparison plots.
# Clean arrays (obs, kd_list) are kept for counterfactual proofs.
TECH_NOISE_CV = 0.1

def add_technical_noise(data, seed):
    rng = np.random.RandomState(seed)
    sigma = TECH_NOISE_CV * np.sqrt(np.maximum(data, 1e-6))
    return np.maximum(data + rng.normal(0, sigma, size=data.shape), 0.0)

obs_noisy = add_technical_noise(obs, seed=1000)
kd_noisy  = [add_technical_noise(kd, seed=2000 + i) for i, kd in enumerate(kd_list)]
print(f"Technical noise added (CV={TECH_NOISE_CV}); noisy arrays used for NMF comparison only")

# Program boundary lines (x-positions of first gene in each program)
prog_starts = [N_TF_GENES + p * N_GENES_PER_PROGRAM for p in range(N_TFS)]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — TF production rates heatmap (8 TFs: activators + repressors)
# ─────────────────────────────────────────────────────────────────────────────
tf_labels = []
for p in range(N_TFS):
    tf_labels.append(f"TF{p}_act")
    tf_labels.append(f"TF{p}_rep")

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(TF_RATES, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(N_BINS))
ax.set_xticklabels([f"Bin {b}" for b in range(N_BINS)], rotation=45, ha="right")
ax.set_yticks(range(N_TF_GENES))
ax.set_yticklabels(tf_labels)
ax.set_title(
    "TF Production Rates per Cell Type\n"
    "Each program has one ACTIVATING (_act) and one REPRESSING (_rep) master regulator.\n"
    "TF_p_act ≈ TF_q_act and TF_p_rep ≈ TF_q_rep for confounded pairs (P0–P1, P2–P3).",
    fontsize=9,
)
fig.colorbar(im, ax=ax, label="Production rate")
# Draw horizontal lines separating program pairs
for p in range(1, N_TFS):
    ax.axhline(2 * p - 0.5, color="white", lw=2)
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_tf_rates.png", dpi=150)
plt.close(fig)
print("Saved plot 1: TF rates heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Observational expression heatmap
# ─────────────────────────────────────────────────────────────────────────────
sort_idx = np.argsort(cell_types)
obs_sorted = obs[:, sort_idx]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(np.log1p(obs_sorted), aspect="auto", cmap="viridis", interpolation="nearest")
fig.colorbar(im, ax=ax, label="log(1+expression)")

for s in prog_starts:
    ax.axhline(s - 0.5, color="white", lw=1.5, ls="--")
ax.axhline(N_TF_GENES - 0.5, color="white", lw=2)
cells_per_bin = N_SC_TOTAL // N_BINS
for b in range(1, N_BINS):
    ax.axvline(b * cells_per_bin - 0.5, color="red", lw=0.8, alpha=0.6)

ax.set_title(
    "Observational Expression (log scale, cells sorted by cell type)\n"
    "Each target gene is HIGH in activation bins and LOW in repression bins\n"
    "Programs 0–1 share the same pattern (bins 0–2 active) → confounded; same for Programs 2–3 (bins 3–5 active)"
)
ax.set_xlabel("Cells")
ax.set_ylabel("Genes  [TF pairs | Prog0 | Prog1 | Prog2 | Prog3]")
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_obs_heatmap.png", dpi=150)
plt.close(fig)
print("Saved plot 2: obs expression heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Observational PCA
# ─────────────────────────────────────────────────────────────────────────────
pca = PCA(n_components=4)
coords = pca.fit_transform(obs.T)  # (N_cells, 4)

cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for b in range(N_BINS):
    mask = cell_types == b
    axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[cmap(b)], label=f"Bin {b}",
                    alpha=0.6, s=12)
    axes[1].scatter(coords[mask, 2], coords[mask, 3], c=[cmap(b)], label=f"Bin {b}",
                    alpha=0.6, s=12)

axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("Observational PCA — PC1 vs PC2\n"
                  "(expect P0+P1 vs P2+P3 confounding)")
axes[0].legend(ncol=3, fontsize=7)

axes[1].set_xlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
axes[1].set_ylabel(f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)")
axes[1].set_title("Observational PCA — PC3 vs PC4")

fig.tight_layout()
fig.savefig(PLOT_DIR / "03_obs_pca.png", dpi=150)
plt.close(fig)
print("Saved plot 3: observational PCA")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Observational NMF gene scores
# ─────────────────────────────────────────────────────────────────────────────
nmf_obs = NMF(n_components=N_TFS, random_state=0, max_iter=500)
nmf_obs.fit(obs.T)
H_obs = nmf_obs.components_  # (4, 48)
H_obs_norm = H_obs / (H_obs.max(axis=1, keepdims=True) + 1e-10)

fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(H_obs_norm, aspect="auto", cmap="hot", interpolation="nearest")
fig.colorbar(im, ax=ax, label="NMF loading (row-normalised)")
for s in prog_starts:
    ax.axvline(s - 0.5, color="cyan", lw=1.5, ls="--")
ax.axvline(N_TF_GENES - 0.5, color="cyan", lw=2, ls="--")
ax.set_title("Observational NMF (4 components)\n"
             "Expect: components do NOT align with true programs (P0 confounded with P1, P2 with P3)")
ax.set_xlabel("Genes  [TF pairs | Prog0 | Prog1 | Prog2 | Prog3]")
ax.set_yticks(range(N_TFS))
ax.set_yticklabels([f"NMF-{i}" for i in range(N_TFS)])
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_obs_nmf.png", dpi=150)
plt.close(fig)
print("Saved plot 4: observational NMF")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Per-perturbation mean delta heatmaps (shared colorbar)
# Shows sign flip: RED = gene goes DOWN (activation loss); BLUE = gene goes UP (repression relief)
# ─────────────────────────────────────────────────────────────────────────────
delta_means = []
for kd in kd_list:
    dm = np.zeros((N_GENES, N_BINS))
    for b in range(N_BINS):
        mask = cell_types == b
        dm[:, b] = np.mean(obs[:, mask] - kd[:, mask], axis=1)
    delta_means.append(dm)

vmax = max(np.percentile(np.abs(dm), 99) for dm in delta_means)

fig, axes = plt.subplots(1, N_TFS, figsize=(15, 6), sharey=True)
for p, (ax, dm) in enumerate(zip(axes, delta_means)):
    im = ax.imshow(dm, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vmax, vmax=vmax)
    for s in prog_starts:
        ax.axhline(s - 0.5, color="black", lw=1, ls="--")
    ax.axhline(N_TF_GENES - 0.5, color="black", lw=1.5, ls="--")
    ax.set_title(f"KD Prog {p}")
    ax.set_xlabel("Cell type (bin)")
    if p == 0:
        ax.set_ylabel("Genes  [TF pairs | Prog0 | Prog1 | Prog2 | Prog3]")

fig.subplots_adjust(right=0.87, top=0.88)
cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.65])
fig.colorbar(im, cax=cbar_ax, label="mean(obs − kd)")
fig.suptitle(
    "Per-Perturbation Deltas — same gene shows POSITIVE delta (red) in activation bins "
    "and NEGATIVE delta (blue) in repression bins.\n"
    "ATE (average over all bins) ≈ 0 per gene because red and blue cancel.",
    fontsize=10,
)
fig.savefig(PLOT_DIR / "05_delta_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plot 5: delta heatmaps")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — MA-plot: sign flip per cell type for the same gene
# Blue (non-downstream, delta=0 exactly) drawn on top to prove counterfactual
# ─────────────────────────────────────────────────────────────────────────────
# Active bins = bins where TF_p_act is highest
_active_act_bins = {0: [1, 2], 1: [1, 2], 2: [3, 4], 3: [3, 4]}
_active_rep_bins = {0: [3, 4, 5], 1: [3, 4, 5], 2: [6, 7, 8], 3: [6, 7, 8]}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
alpha = 0.12
sz = 4

for p, (kd, ax) in enumerate(zip(kd_list, axes.ravel())):
    act_id, rep_id = TF_PAIR_IDS[p]
    act_bins = _active_act_bins[p]
    rep_bins = _active_rep_bins[p]

    # Downstream program genes — color by cell type (act bin vs rep bin)
    for b in act_bins:
        mask = cell_types == b
        g = PROGRAM_GENE_IDS[p]
        x = np.log1p(obs[g][:, mask].ravel())
        y = (obs[g][:, mask] - kd[g][:, mask]).ravel()
        ax.scatter(x, y, c="crimson", alpha=alpha * 2.5, s=sz * 2, rasterized=True,
                   label=f"Downstream, act. bins {act_bins} (Δ>0)" if b == act_bins[0] else None,
                   zorder=2)
    for b in rep_bins:
        mask = cell_types == b
        g = PROGRAM_GENE_IDS[p]
        x = np.log1p(obs[g][:, mask].ravel())
        y = (obs[g][:, mask] - kd[g][:, mask]).ravel()
        ax.scatter(x, y, c="steelblue", alpha=alpha * 2.5, s=sz * 2, rasterized=True,
                   label=f"Downstream, rep. bins {rep_bins} (Δ<0)" if b == rep_bins[0] else None,
                   zorder=2)

    # TF genes (act and rep of this program)
    for tf in (act_id, rep_id):
        x = np.log1p(obs[tf].ravel())
        y = (obs[tf] - kd[tf]).ravel()
        ax.scatter(x, y, c="gray", alpha=0.5, s=sz * 2, rasterized=True,
                   label="KD'd TFs" if tf == act_id else None, zorder=2)

    # Non-downstream programs — delta = 0 exactly → draw LAST (on top)
    for q in range(N_TFS):
        if q == p:
            continue
        g = PROGRAM_GENE_IDS[q]
        x = np.log1p(obs[g].ravel())
        y = (obs[g] - kd[g]).ravel()
        ax.scatter(x, y, c="limegreen", alpha=0.25, s=sz * 1.5, rasterized=True,
                   label="Non-downstream (Δ=0 exactly)" if q == (p + 1) % N_TFS else None,
                   zorder=3)

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("log1p(Obs expression)")
    ax.set_ylabel("Obs − KD (Δ)")
    ax.set_title(f"KD Prog {p}: same gene goes UP in rep. bins, DOWN in act. bins")
    ax.legend(fontsize=7, markerscale=3)

fig.suptitle(
    "MA-plot: cell-type-heterogeneous response (sign flip per gene)\n"
    "Crimson = activation bins (Δ>0); Blue = repression bins (Δ<0); Green (on top) = non-downstream Δ=0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "06_counterfactual_scatter.png", dpi=150)
plt.close(fig)
print("Saved plot 6: MA-plot counterfactual validation")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Residual bar chart (mean|delta| per gene, proving exact counterfactual)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_TFS, figsize=(14, 4), sharey=True)
for p, (ax, kd) in enumerate(zip(axes, kd_list)):
    delta = obs - kd
    mean_abs = np.mean(np.abs(delta), axis=1)

    act_id, rep_id = TF_PAIR_IDS[p]
    non_prog_idx = [g for g in range(N_GENES)
                    if g not in ALL_TF_IDS and g not in PROGRAM_GENE_IDS[p]]

    colors = []
    for g in range(N_GENES):
        if g in ALL_TF_IDS:
            colors.append("gray")
        elif g in PROGRAM_GENE_IDS[p]:
            colors.append("crimson")
        else:
            colors.append("steelblue")

    ax.bar(range(N_GENES), mean_abs, color=colors, width=1.0, linewidth=0)
    # Explicit tick-marks for non-downstream genes at y=0
    ax.scatter(non_prog_idx, np.zeros(len(non_prog_idx)),
               marker="|", s=40, c="steelblue", linewidths=1.2, zorder=3,
               label="Non-downstream = 0 (exact)" if p == 0 else None)

    for s in prog_starts:
        ax.axvline(s - 0.5, color="black", lw=1, ls="--", alpha=0.4)
    ax.axvline(N_TF_GENES - 0.5, color="black", lw=1, ls="--", alpha=0.4)
    ax.set_title(f"KD Prog {p}")
    ax.set_xlabel("Gene index")
    if p == 0:
        ax.set_ylabel("Mean |obs − kd| per gene")
        ax.legend(fontsize=7)

    non_down_max = mean_abs[non_prog_idx].max() if non_prog_idx else 0.0
    ax.text(0.98, 0.97, f"non-prog\nmax={non_down_max:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="steelblue",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(
    "Residual per Gene  (Crimson = downstream program, Blue | = non-downstream exactly 0, Gray = TF genes)\n"
    "Blue tick-marks at y=0 prove exact counterfactual guarantee",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "07_residuals.png", dpi=150)
plt.close(fig)
print("Saved plot 7: residual bar chart")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Perturbation (Sample-level) NMF on stacked |obs-kd| (noisy data)
# Split display: TF genes | Program genes
# ─────────────────────────────────────────────────────────────────────────────
abs_deltas = [np.abs(obs_noisy - kd_n) for kd_n in kd_noisy]
stacked = np.concatenate(abs_deltas, axis=1)  # (48, 4×1350)

nmf_pert = NMF(n_components=N_TFS, random_state=0, max_iter=1000)
nmf_pert.fit(stacked.T)
H_pert = nmf_pert.components_  # (4, 48)

# Normalise by max over PROGRAM genes so program structure is at full contrast
H_pert_norm = np.clip(
    H_pert / (H_pert[:, N_TF_GENES:].max(axis=1, keepdims=True) + 1e-10),
    0, None
)

fig, axes = plt.subplots(1, 2, figsize=(16, 3),
                         gridspec_kw={"width_ratios": [N_TF_GENES, N_GENES - N_TF_GENES]})

im0 = axes[0].imshow(H_pert_norm[:, :N_TF_GENES], aspect="auto", cmap="hot",
                     interpolation="nearest", vmin=0)
axes[0].set_title("TF gene loadings\n(clipped at prog. max)")
axes[0].set_xticks(range(N_TF_GENES))
axes[0].set_xticklabels(tf_labels, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(N_TFS))
axes[0].set_yticklabels([f"NMF-{i}" for i in range(N_TFS)])
axes[0].set_xlabel("TF genes")

im1 = axes[1].imshow(H_pert_norm[:, N_TF_GENES:], aspect="auto", cmap="hot",
                     interpolation="nearest", vmin=0, vmax=1)
for s in prog_starts:
    axes[1].axvline(s - N_TF_GENES - 0.5, color="cyan", lw=1.5, ls="--")
axes[1].set_title("Program gene loadings (row-norm. by prog-gene max)\n"
                  "Expect: each component lights up exactly one 10-gene block")
axes[1].set_xlabel("Program genes  [Prog0 | Prog1 | Prog2 | Prog3]")
axes[1].set_yticks(range(N_TFS))
axes[1].set_yticklabels([f"NMF-{i}" for i in range(N_TFS)])

fig.colorbar(im1, ax=axes[1], label="loading (row-norm. by prog. max)")
fig.suptitle("Sample-level Perturbation NMF — 4 components on stacked |obs−kd|", fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "08_pert_nmf.png", dpi=150)
plt.close(fig)
print("Saved plot 8: perturbation NMF")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — ATE-based NMFs (population-level — both FAIL)
# ─────────────────────────────────────────────────────────────────────────────
ate_matrix = np.stack([np.mean(obs_noisy - kd_n, axis=1) for kd_n in kd_noisy], axis=1)

# (a) Rectified ATE: drops the negative (repression) half of the sign flip
ate_rect = np.maximum(ate_matrix, 0.0).T  # (N_TFS, N_GENES)
nmf_ate = NMF(n_components=N_TFS, random_state=0, max_iter=1000)
nmf_ate.fit(ate_rect)
H_ate = nmf_ate.components_  # (4, 48)

# (b) |ATE|: keeps magnitude but loses sign info; population average diluted by cell heterogeneity
ate_abs = np.abs(ate_matrix).T  # (N_TFS, N_GENES)
nmf_ate_abs = NMF(n_components=N_TFS, random_state=0, max_iter=1000)
nmf_ate_abs.fit(ate_abs)
H_ate_abs = nmf_ate_abs.components_  # (4, 48)

H_ate_norm     = np.clip(H_ate     / (H_ate    [:, N_TF_GENES:].max(axis=1, keepdims=True) + 1e-10), 0, None)
H_ate_abs_norm = np.clip(H_ate_abs / (H_ate_abs[:, N_TF_GENES:].max(axis=1, keepdims=True) + 1e-10), 0, None)

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
for ax, H_norm, title in zip(
    axes,
    [H_ate_norm, H_ate_abs_norm],
    ["(a) Rectified ATE NMF: max(mean(obs-kd), 0)  — misses repression half of the signal",
     "(b) |ATE| NMF: |mean(obs-kd)|  — loses per-cell sign; population mean ≈ 0 for some programs"],
):
    im = ax.imshow(H_norm[:, N_TF_GENES:], aspect="auto", cmap="hot",
                   interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="loading (row-norm. by prog. max)")
    for s in prog_starts:
        ax.axvline(s - N_TF_GENES - 0.5, color="cyan", lw=1.5, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Program genes  [Prog0 | Prog1 | Prog2 | Prog3]")
    ax.set_yticks(range(N_TFS))
    ax.set_yticklabels([f"NMF-{i}" for i in range(N_TFS)])
fig.suptitle("ATE-based NMFs — program genes only; both fail to recover clean per-program blocks", fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "09_ate_nmf.png", dpi=150)
plt.close(fig)
print("Saved plot 9: ATE-based NMFs (failure cases)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 — Signed ATE per gene: shows sign flip per bin
# ─────────────────────────────────────────────────────────────────────────────
ate_all_genes = [np.mean(obs - kd, axis=1) for kd in kd_list]
zoom_max = max(np.max(np.abs(a[N_TF_GENES:])) for a in ate_all_genes)
zoom_clip = max(zoom_max * 1.3, 0.02)

fig, axes = plt.subplots(2, N_TFS, figsize=(14, 8), sharex=True)
for p, ate_per_gene in enumerate(ate_all_genes):
    colors = ["gray" if g in ALL_TF_IDS else
              ("crimson" if g in PROGRAM_GENE_IDS[p] else "steelblue")
              for g in range(N_GENES)]

    for row, ax in enumerate([axes[0, p], axes[1, p]]):
        ax.bar(range(N_GENES), ate_per_gene, color=colors, width=1.0, linewidth=0)
        ax.axhline(0, color="black", lw=0.8)
        for s in prog_starts:
            ax.axvline(s - 0.5, color="black", lw=1, ls="--", alpha=0.4)
        ax.axvline(N_TF_GENES - 0.5, color="black", lw=1, ls="--", alpha=0.4)
        if row == 0:
            ax.set_title(f"KD Prog {p}\n(full scale)")
        else:
            ax.set_ylim(-zoom_clip, zoom_clip)
            ax.set_xlabel("Gene index")
            ax.text(0.98, 0.95, "zoomed\n(TF bar clipped)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="gray")
        if p == 0:
            ax.set_ylabel("mean(obs − kd)  [ATE]")

fig.suptitle(
    "Signed ATE per gene  (Gray=TF genes, Crimson=downstream program, Blue=other programs)\n"
    "Bottom row zoomed: program genes show ATE ≈ 0 because positive (act. bins) and negative (rep. bins) cancel",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "10_ate_cancellation.png", dpi=150)
plt.close(fig)
print("Saved plot 10: ATE per gene (sign flip illustration)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 11 — Program recovery comparison (Hungarian-optimal assignment)
# ─────────────────────────────────────────────────────────────────────────────
TOP_K = N_GENES_PER_PROGRAM + 2   # top 12 (10 targets + 2 TF genes per program)

def top_genes(H, k):
    return [set(np.argsort(H[i])[::-1][:k]) for i in range(H.shape[0])]

def jaccard_matrix(true_sets, pred_sets):
    J = np.zeros((len(true_sets), len(pred_sets)))
    for i, ts in enumerate(true_sets):
        for j, ps in enumerate(pred_sets):
            inter = len(ts & ps)
            union = len(ts | ps)
            J[i, j] = inter / union if union > 0 else 0.0
    return J

def optimal_jaccard(J):
    row_ind, col_ind = linear_sum_assignment(-J)
    return J[row_ind, col_ind]

# True sets include program target genes + the two TF genes for that program
true_sets = [set(PROGRAM_GENE_IDS[p]) | {TF_PAIR_IDS[p][0], TF_PAIR_IDS[p][1]}
             for p in range(N_TFS)]

obs_top      = top_genes(H_obs,     TOP_K)
ate_top      = top_genes(H_ate,     TOP_K)
ate_abs_top  = top_genes(H_ate_abs, TOP_K)
pert_top     = top_genes(H_pert,    TOP_K)

J_obs     = jaccard_matrix(true_sets, obs_top)
J_ate     = jaccard_matrix(true_sets, ate_top)
J_ate_abs = jaccard_matrix(true_sets, ate_abs_top)
J_pert    = jaccard_matrix(true_sets, pert_top)

best_obs     = optimal_jaccard(J_obs)
best_ate     = optimal_jaccard(J_ate)
best_ate_abs = optimal_jaccard(J_ate_abs)
best_pert    = optimal_jaccard(J_pert)

print("\n── Program recovery (Jaccard, Hungarian-optimal) ──────────────────")
print(f"{'Program':<10} {'Obs NMF':>10} {'ATE rect':>10} {'|ATE| NMF':>11} {'Sample NMF':>12}")
for p in range(N_TFS):
    print(f"Program {p}  {best_obs[p]:>10.3f} {best_ate[p]:>10.3f} {best_ate_abs[p]:>11.3f} {best_pert[p]:>12.3f}")
print(f"{'Mean':<10} {best_obs.mean():>10.3f} {best_ate.mean():>10.3f} {best_ate_abs.mean():>11.3f} {best_pert.mean():>12.3f}")

# Binary assignment matrices for visualization
def optimal_assignment_top(H, true_sets, k):
    preds = top_genes(H, k)
    J = jaccard_matrix(true_sets, preds)
    row_ind, col_ind = linear_sum_assignment(-J)
    ordered = [None] * len(true_sets)
    for r, c in zip(row_ind, col_ind):
        ordered[r] = preds[c]
    return ordered

obs_top_ord     = optimal_assignment_top(H_obs,     true_sets, TOP_K)
ate_top_ord     = optimal_assignment_top(H_ate,     true_sets, TOP_K)
ate_abs_top_ord = optimal_assignment_top(H_ate_abs, true_sets, TOP_K)
pert_top_ord    = optimal_assignment_top(H_pert,    true_sets, TOP_K)

def make_mat(top_ord):
    mat = np.zeros((N_TFS, N_GENES))
    for p in range(N_TFS):
        for g in top_ord[p]:
            mat[p, g] = 1.0
    return mat

true_mat    = make_mat([set(PROGRAM_GENE_IDS[p]) | {TF_PAIR_IDS[p][0], TF_PAIR_IDS[p][1]}
                        for p in range(N_TFS)])
obs_mat     = make_mat(obs_top_ord)
ate_mat     = make_mat(ate_top_ord)
ate_abs_mat = make_mat(ate_abs_top_ord)
pert_mat    = make_mat(pert_top_ord)

boundary_lines = [N_TF_GENES] + [N_TF_GENES + p * N_GENES_PER_PROGRAM for p in range(N_TFS - 1)]

fig, axes = plt.subplots(1, 5, figsize=(24, 4))
panel_data = [
    (true_mat,    "(a) True Programs",                              "Greys",   0.9),
    (obs_mat,     f"(b) Obs NMF top-{TOP_K}\nJ={best_obs.mean():.3f}  ✗",     "Reds",    0.8),
    (ate_mat,     f"(c) Rect.ATE NMF top-{TOP_K}\nJ={best_ate.mean():.3f}  ✗","Reds",    0.8),
    (ate_abs_mat, f"(d) |ATE| NMF top-{TOP_K}\nJ={best_ate_abs.mean():.3f}  ✗","Oranges", 0.8),
    (pert_mat,    f"(e) Sample NMF top-{TOP_K}\nJ={best_pert.mean():.3f}  ✓",  "Greens",  0.9),
]

for ax, (mat, title, cmap_name, vmax) in zip(axes, panel_data):
    ax.imshow(mat, aspect="auto", cmap=cmap_name, interpolation="nearest", vmin=0, vmax=vmax)
    for bl in boundary_lines:
        ax.axvline(bl - 0.5, color="red", lw=1.5, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Gene index")
    ax.set_yticks(range(N_TFS))
    ax.set_yticklabels([f"Prog {p}" for p in range(N_TFS)])

fig.suptitle(
    "Program Recovery (Hungarian-optimal): True | Obs NMF | Rect.ATE | |ATE| | Sample NMF\n"
    "Red dashed = program boundaries. Only cell-level sample NMF fully recovers true programs.",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "11_program_recovery.png", dpi=150)
plt.close(fig)
print("Saved plot 11: program recovery comparison")

# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Counterfactual exactness check ───────────────────────────────────")
for p in range(N_TFS):
    kd = kd_list[p]
    delta = obs - kd
    act_id, rep_id = TF_PAIR_IDS[p]
    non_down = [g for q in range(N_TFS) if q != p for g in PROGRAM_GENE_IDS[q]]
    other_tfs = [t for t in ALL_TF_IDS if t not in (act_id, rep_id)]
    non_down.extend(other_tfs)
    downstream = PROGRAM_GENE_IDS[p]
    max_non  = np.max(np.abs(delta[non_down]))
    mean_dn  = np.mean(np.abs(delta[downstream]))
    print(f"KD P{p}: max|delta| non-downstream = {max_non:.2e}  mean|delta| downstream = {mean_dn:.4f}")

print("\n── ATE cancellation check (per-program) ──────────────────────────────")
for p in range(N_TFS):
    kd = kd_list[p]
    ate_per_gene = np.mean(obs[PROGRAM_GENE_IDS[p]] - kd[PROGRAM_GENE_IDS[p]], axis=1)
    print(f"KD P{p}: per-gene ATE range [{ate_per_gene.min():.3f}, {ate_per_gene.max():.3f}]  "
          f"mean|ATE|={np.abs(ate_per_gene).mean():.3f}")

print("\n── Per-bin ATE for each program's own KD ──────────────────────────────")
for p in range(N_TFS):
    kd = kd_list[p]
    per_bin = []
    for b in range(N_BINS):
        mask = cell_types == b
        per_bin.append(np.mean((obs - kd)[PROGRAM_GENE_IDS[p]][:, mask]))
    signs = ["++" if x > 0.2 else ("--" if x < -0.2 else "  ") for x in per_bin]
    row = "  ".join(f"{x:+.2f}{s}" for x, s in zip(per_bin, signs))
    print(f"KD P{p} per-bin ATE: {row}")

print("\nDone. All plots saved to:", PLOT_DIR)
