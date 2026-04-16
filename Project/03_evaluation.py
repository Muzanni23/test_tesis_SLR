# ============================================================
# 03_evaluation.py
# TAHAP 3: Evaluasi Lengkap & Visualisasi
#
# Apa yang dilakukan file ini:
#   1. Load semua hasil dari Tahap 2
#   2. Hitung metrik evaluasi komprehensif
#   3. Buat 6 grafik visualisasi wajib
#   4. Cetak laporan evaluasi ke konsol
#
# Cara menjalankan:
#   python 03_evaluation.py
#   (Jalankan SETELAH 02_modeling.py)
#
# Output:
#   outputs/figures/01_feature_importance.png
#   outputs/figures/02_actual_vs_predicted.png
#   outputs/figures/03_picp_comparison.png
#   outputs/figures/04_interval_width_dist.png
#   outputs/figures/05_prediction_intervals.png
#   outputs/figures/06_coverage_comparison.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend untuk save ke file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_CLEAN_PATH, MODEL_BASE_PATH, MODEL_CQR_PATH,
    MODEL_CAL_PATH, REPORT_PRED_PATH, FIGURE_DIR,
    FEATURE_COLS, TARGET_COL, ALPHA_LEVELS,
    RANDOM_SEED, create_dirs
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# KONFIGURASI STYLE GRAFIK
# ============================================================
plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 10,
    'axes.titlesize' : 12,
    'axes.labelsize' : 10,
    'figure.dpi'     : 150,
    'axes.facecolor' : '#F8FAFC',
    'figure.facecolor': '#FFFFFF',
    'axes.grid'      : True,
    'grid.alpha'     : 0.3,
    'grid.linestyle' : '--',
})

# Warna tema
C_BLUE   = '#1F4E79'
C_LBLUE  = '#2E74B5'
C_ACCENT = '#BDD7EE'
C_GREEN  = '#375623'
C_LGREEN = '#E2EFDA'
C_RED    = '#C00000'
C_ORANGE = '#ED7D31'


# ============================================================
# FUNGSI VISUALISASI
# ============================================================

def plot_feature_importance(model, feature_names, save_path):
    """
    Grafik 1: Feature Importance bar chart horizontal.

    Highlight fitur novelty (Faktor_Depresiasi) dengan warna berbeda.
    """
    fi = model.feature_importances_
    idx = np.argsort(fi)  # Urutkan dari kecil ke besar (untuk barh)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [C_ORANGE if 'Depresiasi' in feature_names[i] else C_LBLUE for i in idx]
    bars = ax.barh([feature_names[i] for i in idx], fi[idx],
                   color=colors, edgecolor='white', linewidth=0.5)

    # Label nilai di setiap bar
    for bar, val in zip(bars, fi[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    # Legend
    novelty_patch = mpatches.Patch(color=C_ORANGE, label='Fitur Depresiasi (NOVELTY PMK)')
    normal_patch  = mpatches.Patch(color=C_LBLUE,  label='Fitur lainnya')
    ax.legend(handles=[novelty_patch, normal_patch], fontsize=9)

    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Feature Importance — Gradient Boosting Model\n'
                 '(Semakin tinggi = semakin berpengaruh terhadap prediksi harga)',
                 fontsize=12, fontweight='bold', color=C_BLUE)
    ax.set_xlim(0, max(fi) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


def plot_actual_vs_predicted(df_pred, metrics, save_path):
    """
    Grafik 2: Scatter plot Actual vs Predicted.

    Titik diwarnai berdasarkan apakah masuk dalam interval 90%.
    Garis diagonal = prediksi sempurna.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # Sample max 2000 titik agar tidak terlalu penuh
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(df_pred), min(2000, len(df_pred)), replace=False)
    y_act  = df_pred['y_actual'].values[idx] / 1e6
    y_pred = df_pred['y_pred_point'].values[idx] / 1e6
    covered = df_pred['covered_90'].values[idx]

    colors_sc = [C_GREEN if c else C_RED for c in covered]
    ax.scatter(y_act, y_pred, c=colors_sc, alpha=0.4, s=8, linewidth=0)

    # Garis prediksi sempurna
    max_val = max(y_act.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5,
            label='Prediksi Sempurna', alpha=0.7)

    # Teks metrik di pojok kiri atas
    textstr = (f"R² = {metrics['R2']:.4f}\n"
               f"MAE = Rp{metrics['MAE']/1e6:.2f}M\n"
               f"MAPE = {metrics['MAPE']:.1f}%")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    in_patch  = mpatches.Patch(color=C_GREEN, alpha=0.7, label='Dalam interval 90%')
    out_patch = mpatches.Patch(color=C_RED,   alpha=0.7, label='Luar interval 90%')
    ax.legend(handles=[
        plt.Line2D([0],[0], color='black', linestyle='--', label='Prediksi Sempurna'),
        in_patch, out_patch
    ], fontsize=9)

    ax.set_xlabel('Harga Aktual (Juta Rp)', fontsize=11)
    ax.set_ylabel('Harga Prediksi (Juta Rp)', fontsize=11)
    ax.set_title(f'Actual vs Predicted (n={len(idx):,} sampel)',
                 fontsize=12, fontweight='bold', color=C_BLUE)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


def plot_picp_comparison(cqr_metrics, save_path):
    """
    Grafik 3: Perbandingan PICP aktual vs target coverage.

    Bar biru = target, bar hijau/oranye = aktual.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels   = ['90%', '85%', '80%']
    targets  = [90, 85, 80]
    actuals  = [cqr_metrics[a]['picp'] * 100 for a in ALPHA_LEVELS]
    x        = np.arange(len(labels))
    w        = 0.35

    ax.bar(x - w/2, targets, w, label='Target Coverage', color=C_ACCENT,
           edgecolor=C_LBLUE, linewidth=1.2)
    bar_colors = [C_GREEN if abs(a-t) <= 1.5 else C_ORANGE
                  for a, t in zip(actuals, targets)]
    bars2 = ax.bar(x + w/2, actuals, w, label='PICP Aktual',
                   color=bar_colors, edgecolor='white')

    # Label nilai di atas bar
    for bar, val in zip(bars2, actuals):
        status = '✓' if val >= targets[bars2.index(bar)] - 1.5 else '~'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                f'{val:.1f}%{status}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Coverage\n{l}' for l in labels])
    ax.set_ylabel('Coverage (%)')
    ax.set_ylim(70, 100)
    ax.set_title('PICP Aktual vs Target Coverage\n'
                 '(Selisih ≤ 1.5% = jaminan CQR terpenuhi ✓)',
                 fontsize=12, fontweight='bold', color=C_BLUE)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


def plot_interval_width_distribution(df_pred, save_path):
    """
    Grafik 4: Distribusi lebar interval prediksi CQR 90%.

    Garis vertikal = median dan mean.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    widths = df_pred['width_90'].values / 1e6  # Konversi ke juta Rp
    # Filter 95 percentil atas untuk tampilan lebih baik
    widths_trim = widths[widths < np.percentile(widths, 95)]

    ax.hist(widths_trim, bins=50, color=C_LBLUE, edgecolor='white', alpha=0.85)
    ax.axvline(np.median(widths), color=C_BLUE, linewidth=2, linestyle='--',
               label=f'Median: Rp{np.median(widths):.1f}M')
    ax.axvline(np.mean(widths), color=C_ORANGE, linewidth=2, linestyle=':',
               label=f'Mean: Rp{np.mean(widths):.1f}M')

    ax.set_xlabel('Lebar Interval Prediksi (Juta Rp)')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi Lebar Interval CQR 90%\n'
                 '(Interval lebih sempit = prediksi lebih informatif)',
                 fontsize=12, fontweight='bold', color=C_BLUE)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


def plot_prediction_intervals(df_pred, save_path, n_samples=40):
    """
    Grafik 5: Contoh interval prediksi CQR untuk n sampel.

    Setiap bar vertikal = interval prediksi satu kendaraan.
    Diamond biru = harga aktual.
    Kotak hijau = dalam interval, merah = luar interval.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Ambil n_samples tersebar merata berdasarkan urutan harga aktual
    idx = np.argsort(df_pred['y_actual'].values)
    step = max(1, len(idx) // n_samples)
    idx_sample = idx[::step][:n_samples]

    y_act  = df_pred['y_actual'].values[idx_sample] / 1e6
    y_low  = df_pred['low_90'].values[idx_sample] / 1e6
    y_high = df_pred['high_90'].values[idx_sample] / 1e6
    y_pred = df_pred['y_pred_point'].values[idx_sample] / 1e6
    covered = df_pred['covered_90'].values[idx_sample]

    x = np.arange(len(idx_sample))

    for i, (lo, hi, ya, cov) in enumerate(zip(y_low, y_high, y_act, covered)):
        bg = C_LGREEN if cov else '#FFCCCC'
        ax.fill_between([i - 0.4, i + 0.4], [lo, lo], [hi, hi],
                        alpha=0.5, color=bg)
        ax.plot([i - 0.4, i + 0.4], [lo, lo], color=C_LBLUE, linewidth=0.8)
        ax.plot([i - 0.4, i + 0.4], [hi, hi], color=C_LBLUE, linewidth=0.8)

    ax.scatter(x, y_act,  color=C_BLUE,   s=40, zorder=5, marker='D',
               label='Harga Aktual')
    ax.scatter(x, y_pred, color=C_ORANGE, s=20, zorder=4, marker='o',
               alpha=0.8, label='Prediksi Titik')

    in_patch  = mpatches.Patch(color=C_LGREEN, alpha=0.8, label='Dalam Interval ✓')
    out_patch = mpatches.Patch(color='#FFCCCC', alpha=0.8, label='Luar Interval ✗')

    ax.legend(handles=[
        in_patch, out_patch,
        plt.scatter([], [], color=C_BLUE,   marker='D', s=40),
        plt.scatter([], [], color=C_ORANGE, marker='o', s=20),
    ], labels=['Dalam Interval ✓', 'Luar Interval ✗', 'Harga Aktual', 'Prediksi Titik'],
       fontsize=8, loc='upper left')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(len(x))], fontsize=7)
    ax.set_xlabel('Sampel (diurutkan berdasarkan harga aktual)')
    ax.set_ylabel('Harga (Juta Rp)')
    ax.set_title(f'Contoh {n_samples} Interval Prediksi CQR 90%\n'
                 'Setiap interval menunjukkan estimasi rentang harga limit lelang kendaraan',
                 fontsize=12, fontweight='bold', color=C_BLUE)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


def plot_coverage_comparison(df_pred, save_path):
    """
    Grafik 6: Perbandingan interval 3 level coverage pada 100 sampel.

    3 panel berdampingan untuk 90%, 85%, 80%.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Perbandingan Interval Prediksi pada 3 Level Coverage',
                 fontsize=13, fontweight='bold', color=C_BLUE)

    configs = [
        ('low_90', 'high_90', '90%', C_BLUE,   C_ACCENT),
        ('low_85', 'high_85', '85%', C_GREEN,   C_LGREEN),
        ('low_80', 'high_80', '80%', '#833C00', '#FCE4D6'),
    ]

    # Ambil 100 sampel tersebar
    idx100 = np.argsort(df_pred['y_actual'].values)
    step   = max(1, len(idx100) // 100)
    idx100 = idx100[::step][:100]

    for ax, (col_l, col_h, label, col_dark, col_light) in zip(axes, configs):
        y_a = df_pred['y_actual'].values[idx100] / 1e6
        y_l = df_pred[col_l].values[idx100] / 1e6
        y_h = df_pred[col_h].values[idx100] / 1e6
        x   = np.arange(100)

        ax.fill_between(x, y_l, y_h, alpha=0.4, color=col_light)
        ax.plot(x, y_l, color=col_dark, linewidth=0.8, alpha=0.7)
        ax.plot(x, y_h, color=col_dark, linewidth=0.8, alpha=0.7)
        ax.scatter(x, y_a, color='black', s=4, zorder=5, alpha=0.7)

        # Hitung PICP untuk 100 sampel ini
        covered = ((df_pred[col_l].values[idx100] <= df_pred['y_actual'].values[idx100]) &
                   (df_pred['y_actual'].values[idx100] <= df_pred[col_h].values[idx100]))
        picp    = covered.mean()
        mpiw    = np.mean(df_pred[col_h].values - df_pred[col_l].values) / 1e6

        ax.set_title(f'Coverage {label}\nPICP={picp*100:.1f}% | MPIW=Rp{mpiw:.1f}M',
                     fontweight='bold', color=col_dark, fontsize=11)
        ax.set_xlabel('Sampel')
        ax.set_ylabel('Harga (Juta Rp)')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Disimpan: {save_path}")


# ============================================================
# PIPELINE EVALUASI UTAMA
# ============================================================

def jalankan_evaluasi():
    """
    Pipeline evaluasi: load hasil → cetak metrik → buat visualisasi.
    """
    print("=" * 60)
    print("  TAHAP 3: EVALUASI & VISUALISASI")
    print("=" * 60)

    # ── Load hasil dari Tahap 2 ───────────────────────────────
    print("\n📂 Load hasil modeling...")

    if not os.path.exists(REPORT_PRED_PATH):
        print(f"❌ ERROR: {REPORT_PRED_PATH} tidak ditemukan.")
        print("   Jalankan dulu: python 02_modeling.py")
        sys.exit(1)

    df_pred = pd.read_csv(REPORT_PRED_PATH)
    gbm     = joblib.load(MODEL_BASE_PATH)
    cal_data = joblib.load(MODEL_CAL_PATH)

    metrics_path = os.path.join(os.path.dirname(REPORT_PRED_PATH), 'metrics_summary.json')
    with open(metrics_path) as f:
        metrics_summary = json.load(f)

    print(f"   ✅ Loaded {len(df_pred):,} prediksi")

    bm_metrics  = metrics_summary['base_model']
    cqr_metrics = {
        float(k): v
        for k, v in metrics_summary['cqr_results'].items()
    }

    # ── Cetak Ringkasan Metrik ────────────────────────────────
    print("\n" + "=" * 60)
    print("  RINGKASAN METRIK EVALUASI")
    print("=" * 60)

    print("\n📊 Base Model (Gradient Boosting):")
    print(f"   MAE  : Rp{bm_metrics['MAE']:,.0f}")
    print(f"   RMSE : Rp{bm_metrics['RMSE']:,.0f}")
    print(f"   R²   : {bm_metrics['R2']:.4f}  "
          f"({'Baik ✅' if bm_metrics['R2'] >= 0.70 else 'Cukup ⚠️'})")
    print(f"   MAPE : {bm_metrics['MAPE']:.2f}%")

    print("\n🎯 Conformalized Quantile Regression (CQR):")
    print(f"   {'Level':8s} {'Target':10s} {'PICP':10s} {'Selisih':10s} "
          f"{'MPIW':15s} {'Status':12s}")
    print("   " + "-" * 70)
    for alpha in ALPHA_LEVELS:
        m   = cqr_metrics[alpha]
        target = (1 - alpha) * 100
        picp   = m['picp'] * 100
        selisih = abs(picp - target)
        mpiw    = m['mpiw']
        status  = "✅ Terpenuhi" if selisih <= 1.5 else "⚠️ Hampir"
        print(f"   {target:.0f}%    {target:.2f}%    {picp:.2f}%    "
              f"{selisih:.2f}%    Rp{mpiw/1e6:.2f}M    {status}")

    # ── Buat Visualisasi ──────────────────────────────────────
    print("\n🎨 Membuat visualisasi...")

    plot_feature_importance(
        gbm, FEATURE_COLS,
        os.path.join(FIGURE_DIR, '01_feature_importance.png')
    )
    plot_actual_vs_predicted(
        df_pred, bm_metrics,
        os.path.join(FIGURE_DIR, '02_actual_vs_predicted.png')
    )
    plot_picp_comparison(
        cqr_metrics,
        os.path.join(FIGURE_DIR, '03_picp_comparison.png')
    )
    plot_interval_width_distribution(
        df_pred,
        os.path.join(FIGURE_DIR, '04_interval_width_dist.png')
    )
    plot_prediction_intervals(
        df_pred,
        os.path.join(FIGURE_DIR, '05_prediction_intervals.png')
    )
    plot_coverage_comparison(
        df_pred,
        os.path.join(FIGURE_DIR, '06_coverage_comparison.png')
    )

    print("\n" + "=" * 60)
    print("  ✅ EVALUASI & VISUALISASI SELESAI!")
    print(f"  Grafik tersimpan di: {FIGURE_DIR}/")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    create_dirs()
    jalankan_evaluasi()
