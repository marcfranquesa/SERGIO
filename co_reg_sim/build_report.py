"""
Build a self-contained HTML report for the co_reg_sim benchmark.
Run with:  uv run python co_reg_sim/build_report.py
"""

import base64
import sys
from pathlib import Path

import markdown as md_lib

_here = Path(__file__).parent
_repo = _here.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_here))

PLOT_DIR = _here / "output" / "plots"
OUT_HTML = _here / "output" / "report.html"


def img_b64(name: str) -> str:
    path = PLOT_DIR / name
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.18);">'


REPORT_MD = f"""
# Co-Regulation Simulation Benchmark

**Goal:** Validate that interventional (CRISPRi knockdown) data is necessary and sufficient
to identify individual gene programs that are confounded in purely observational data.

Two key requirements are enforced by design:

1. **Cell-type-heterogeneous response** — each target gene is regulated by both an activating
   and a repressing master regulator with anti-correlated expression across cell types. The
   same gene shows a positive delta in activation bins and a negative delta in repression bins,
   making the signed population-level ATE near zero (effects cancel in the mean).
2. **Perturbations are knockdowns, not knockouts** — TF production rates are reduced to 10%
   (90% reduction), reflecting CRISPRi biology rather than full gene deletion.

---

## 1 · Experimental Design

The benchmark uses the **SERGIO** single-cell RNA simulator to generate ground-truth data
with full control over the gene regulatory network (GRN).

### Gene Regulatory Network

| Element | Detail |
|---|---|
| Master-regulator gene pairs (TF_act + TF_rep) | 4 programs × 2 = 8 TF genes |
| Programs | 4 (10 target genes each) |
| Regulation per target gene | TF_p_act (K = +5) **and** TF_p_rep (K = −5) |
| Total genes | 48 (8 TFs + 40 targets) |
| Cell types (bins) | 9 |
| Cells per bin | 150 |
| Total cells | 1 350 |
| Knockdown fraction | 0.10 (90% reduction, both TF_act and TF_rep KD'd together) |

Each target gene is regulated by **both** TF_p_act and TF_p_rep simultaneously.
When both are knocked down, the gene expression changes in **opposite directions** depending
on the cell type:

- In activation bins (TF_p_act high, TF_p_rep low): KD removes activation → expression drops → Δ = obs − kd > 0.
- In repression bins (TF_p_rep high, TF_p_act low): KD lifts repression → expression rises → Δ = obs − kd < 0.

**Confounding design** — TF production rates are engineered so that:

- TF0_act ≈ TF1_act (both high in bins 0–2); TF0_rep ≈ TF1_rep (both high in bins 3–5)  →  Programs 0 and 1 share identical active bin patterns, differing only in within-bin rate ordering.
- TF2_act ≈ TF3_act (both high in bins 3–5); TF2_rep ≈ TF3_rep (both high in bins 6–8)  →  Programs 2 and 3 are similarly confounded.
- Programs 0+1 are anti-correlated with Programs 2+3 across cell types.

{img_b64("01_tf_rates.png")}
*Figure 1 — TF production rates per cell type (yellow = high). Rows 0–3: activating TFs (TF0_act – TF3_act). Rows 4–7: repressing TFs (TF0_rep – TF3_rep). TF0_act ≈ TF1_act and TF2_act ≈ TF3_act by design — observationally confounded.*

---

## 2 · Observational Expression

Five simulations are run from the **same random seed (42)**:

1. **Observational** — no perturbation
2. **KD Prog 0** — TF0_act + TF0_rep knocked down (rates × 0.10)
3. **KD Prog 1** — TF1_act + TF1_rep knocked down
4. **KD Prog 2** — TF2_act + TF2_rep knocked down
5. **KD Prog 3** — TF3_act + TF3_rep knocked down

Setting the same seed before every `build_graph()` call guarantees **exact counterfactuals**:
any gene not downstream of the knocked-down TF pair receives identical random draws → delta = 0
exactly. This holds for knockdowns (not only knockouts) because the same random draws are
consumed regardless of the knockdown fraction.

{img_b64("02_obs_heatmap.png")}
*Figure 2 — Observational expression heatmap (log scale, cells sorted by cell type). Dashed white lines = program boundaries; red lines = cell-type boundaries. Dark bands within each program block = repressed genes in that program's repression bins.*

---

## 3 · Why Observational Analysis Fails

### 3.1 PCA

{img_b64("03_obs_pca.png")}
*Figure 3 — Observational PCA. PC1 separates bins 0–2 (P0+P1 active) from bins 3–8 (P2+P3 active). Individual programs within each confounded pair (P0 vs P1, P2 vs P3) are inseparable.*

### 3.2 NMF on Observational Data

{img_b64("04_obs_nmf.png")}
*Figure 4 — Observational NMF gene loadings (row-normalised). Components do not align with true program boundaries — programs are mixed across components.*

**Mean Jaccard (obs NMF vs true programs): 0.320** — far from perfect recovery (1.0).

---

## 4 · Perturbation Data

### 4.1 Per-Perturbation Deltas

For each knockdown, the mean expression change `obs − kd` per gene and per cell type reveals
the program structure. The cell-type-heterogeneous response (positive in activation bins,
negative in repression bins) is visible as the red/blue split within each program block:

{img_b64("05_delta_heatmaps.png")}
*Figure 5 — Mean `obs − kd` per gene (rows) and cell type (columns) for each KD, single shared colour scale. Red = activation effect (obs > kd); blue = repression-lifting effect (obs < kd). Non-downstream genes are exactly zero (white).*

### 4.2 Exact Counterfactual Validation — MA-plot

The MA-plot shows `obs − kd` (y-axis) against `log1p(obs)` (x-axis). **Non-downstream
points (green) are rendered on top** so the exact-zero guarantee is visually verifiable.
Crimson = activation-bin cells (delta > 0); steelblue = repression-bin cells (delta < 0):

{img_b64("06_counterfactual_scatter.png")}
*Figure 6 — MA-plot for all 4 KDs (relevant bins only). Limegreen (on top) = non-downstream programs at exactly 0; Crimson = activation-bin cells; Steelblue = repression-bin cells; Gray = TFs.*

### 4.3 Residual Bar Chart — Numerical Proof

{img_b64("07_residuals.png")}
*Figure 7 — Mean |obs − kd| per gene for each KD. Gray = TFs, Crimson = activation effect, Steelblue = repression-lifting effect. Tick-marks at y = 0 for non-downstream genes prove exact counterfactuals.*

**Counterfactual exactness (all 4 KDs):**

| KD | max&#124;delta&#124; non-downstream | mean&#124;delta&#124; downstream |
|---|---|---|
| KD Prog 0 | **0.00e+00** | 0.660 |
| KD Prog 1 | **0.00e+00** | 0.314 |
| KD Prog 2 | **0.00e+00** | 0.501 |
| KD Prog 3 | **0.00e+00** | 0.725 |

*(Downstream signal is smaller than in full knockouts because TFs retain 10% activity.)*

---

## 5 · Cell-type Heterogeneity and ATE Cancellation

### 5.1 Per-Bin ATE — Heterogeneous Response

The per-bin ATE (mean delta per cell type, averaged over the 10 target genes in each program)
demonstrates the design intent: opposite-sign contributions across activation and repression bins.

{img_b64("10_ate_cancellation.png")}
*Figure 8 — Signed ATE per gene. Top row: full scale (TF bar dominates). Bottom row: zoomed to program genes. Red bars = positive ATE (activation bins, obs > kd); blue bars = negative ATE (repression bins, obs < kd). The per-cell sign flip is clearly visible.*

**Per-bin ATE summary (++ strong positive, -- strong negative, values averaged over program genes):**

| KD | Bin 0 | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Bin 6 | Bin 7 | Bin 8 | Per-gene &#124;ATE&#124; |
|---|---|---|---|---|---|---|---|---|---|---|
| KD P0 | −0.14 | **+0.97** | **+1.55** | −0.16 | **−0.75** | **−1.10** | −0.24 | −0.33 | −0.04 | 0.027 |
| KD P1 | +0.13 | **+0.31** | **+0.22** | −0.12 | +0.07 | **−0.88** | −0.11 | **−0.59** | −0.12 | 0.121 |
| KD P2 | +0.17 | +0.14 | **+0.37** | **+0.94** | **+0.90** | +0.19 | **−0.65** | **−0.58** | **−0.45** | 0.114 |
| KD P3 | **+0.82** | **+0.92** | −0.26 | **+0.52** | **+0.30** | **+1.04** | **−0.88** | **−0.96** | **−0.65** | 0.124 |

Program 0 achieves near-perfect ATE cancellation (mean|ATE| = 0.027) because its activation and
repression bins carry symmetric rate profiles. Programs 1–3 show small but consistent net ATEs
(mean|ATE| ≈ 0.11–0.12) due to asymmetric stochastic dynamics in SERGIO's CLE simulator.
In all cases, the **per-bin signal is large** (>0.5 in active bins) even as the mean cancels.

### 5.2 ATE-based NMF — Failure for Near-Zero ATE Programs

Independent technical noise (CV = 0.1) is added per condition to simulate realistic measurement
variability. NMF is then applied to the population-level ATE matrix:

{img_b64("09_ate_nmf.png")}
*Figure 9 — ATE-based NMF gene loadings over program genes. (a) Rectified ATE: max(mean, 0) misses repression-lifting effect entirely. (b) |ATE| NMF: near-zero signal for Program 0 (ATE ≈ 0 by design) causes mis-assignment (Jaccard = 0.600 for P0).*

**Critical finding:** Program 0 — where ATE cancels most cleanly — is **not** recovered by |ATE| NMF
(Jaccard = 0.600), demonstrating that population-level summaries are insufficient when the per-gene
signed delta cancels across cell types.

---

## 6 · Program Recovery with Sample (Cell-Level) NMF

NMF is applied to the **stacked sample-level signal**:
`|obs_noisy − kd_noisy|` (absolute per-cell delta) concatenated across all 4 KDs →
shape (48 genes, 5 400 cells). The absolute value captures both activation and repression
effects. The per-cell structure survives where the population mean cancels.

{img_b64("08_pert_nmf.png")}
*Figure 10 — Sample NMF gene loadings. Left panel: TF gene loadings. Right panel: program gene loadings (row-normalised by program-gene max). Each NMF component lights up exactly one 10-gene block — perfect separation for all 4 programs.*

---

## 7 · Program Recovery Comparison

{img_b64("11_program_recovery.png")}
*Figure 11 — Binary gene assignment (TOP-12 genes per component, Hungarian-optimal matching). Obs NMF and ATE-based methods fail; sample-level NMF recovers all four program blocks exactly.*

### Jaccard Scores (Hungarian-optimal assignment)

| Program | Obs NMF | Rect. ATE | &#124;ATE&#124; NMF | Sample NMF |
|---|---|---|---|---|
| Program 0 | 0.200 | 0.263 | 0.600 | **1.000** |
| Program 1 | 0.412 | 0.091 | 1.000 | **1.000** |
| Program 2 | 0.333 | 1.000 | 1.000 | **1.000** |
| Program 3 | 0.333 | 1.000 | 1.000 | **1.000** |
| **Mean** | 0.320 | 0.589 | 0.900 | **1.000** |

Note: |ATE| NMF achieves perfect Jaccard for Programs 1–3 because SERGIO's stochastic dynamics
produce a small but consistent net ATE (≈ 0.12) for those programs. However, it fails precisely
for Program 0, where ATE cancellation is most complete by design. Sample-level NMF achieves
perfect recovery across **all** programs regardless of whether ATE cancels.

---

## 8 · Conclusion

| Property | Obs NMF | Rect. ATE | &#124;ATE&#124; NMF | Sample NMF |
|---|---|---|---|---|
| Identifies all programs | ✗ | ✗ | ✗ (P0 fails) | ✓ (all 1.000) |
| Mean Jaccard | 0.320 | 0.589 | 0.900 | **1.000** |
| Exact counterfactuals | — | — | — | ✓ (max δ = 0.00e+00) |
| Robust to ATE cancellation | n/a | ✗ | ✗ (P0 fails) | ✓ |

Observational data alone **cannot** separate co-regulated programs that share similar
cell-type activity patterns. Population-level ATE — whether signed or unsigned — is
unreliable: it works by accident when stochastic dynamics create a net ATE, but fails
precisely when ATE cancels as intended (Program 0, mean|ATE| = 0.027 → Jaccard = 0.600).

Only **sample-level (cell-level) perturbation signal** — `|obs − kd|` per cell, stacked
across all knockdowns — provides the structural information needed for guaranteed program
identification independent of whether population-level ATE cancels or not.

This benchmark validates two core requirements of the method under development:

1. **Interventional data is necessary** — observational data fails regardless of ATE behaviour.
2. **Cell-level structure is necessary** — per-cell signal succeeds even when population ATE ≈ 0.
"""

# ── convert markdown → HTML ───────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Co-Regulation Simulation Benchmark</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 16px;
      line-height: 1.7;
      color: #1a1a2e;
      background: #f5f6fa;
      margin: 0;
      padding: 2rem 1rem;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 10px;
      box-shadow: 0 4px 24px rgba(0,0,0,.10);
      padding: 3rem 3.5rem;
    }}
    h1 {{
      font-size: 2rem;
      font-weight: 700;
      color: #16213e;
      border-bottom: 3px solid #0f3460;
      padding-bottom: .5rem;
      margin-bottom: 1.5rem;
    }}
    h2 {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #0f3460;
      margin-top: 2.5rem;
      border-left: 4px solid #e94560;
      padding-left: .75rem;
    }}
    h3 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: #16213e;
      margin-top: 1.8rem;
    }}
    p {{ margin: .75rem 0; }}
    em {{
      display: block;
      font-size: .88rem;
      color: #555;
      margin-top: .4rem;
      margin-bottom: 1.2rem;
    }}
    strong {{ color: #0f3460; }}
    hr {{
      border: none;
      border-top: 1px solid #e0e0e0;
      margin: 2.5rem 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0 1.5rem;
      font-size: .93rem;
    }}
    th {{
      background: #0f3460;
      color: #fff;
      font-weight: 600;
      padding: .55rem .9rem;
      text-align: left;
    }}
    td {{
      padding: .5rem .9rem;
      border-bottom: 1px solid #e8e8e8;
    }}
    tr:nth-child(even) td {{ background: #f8f9fc; }}
    code {{
      background: #f0f0f0;
      border-radius: 3px;
      padding: .1em .35em;
      font-size: .9em;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }}
    img {{
      display: block;
      margin: 1rem auto;
    }}
    .highlight {{
      background: #e8f4fd;
      border-left: 4px solid #2196f3;
      padding: .75rem 1rem;
      border-radius: 0 6px 6px 0;
      margin: 1rem 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""

body_html = md_lib.markdown(
    REPORT_MD,
    extensions=["tables", "fenced_code"],
)

html = HTML_TEMPLATE.format(body=body_html)
OUT_HTML.write_text(html, encoding="utf-8")
print(f"Report written to: {OUT_HTML}")
print(f"File size: {OUT_HTML.stat().st_size / 1024:.0f} KB")
