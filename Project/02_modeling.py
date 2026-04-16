# ============================================================
# 02_modeling.py
# TAHAP 2: Training Base Model + Conformalized Quantile Regression (CQR)
#
# Apa yang dilakukan file ini:
#   1. Load data bersih hasil preprocessing
#   2. Split data: 60% train / 20% calibration / 20% test
#   3. Training Gradient Boosting base model (point prediction)
#   4. Training Quantile GBM models (lower & upper bound) per level α
#   5. Kalibrasi CQR: hitung conformal scores & q̂
#   6. Hasilkan interval prediksi pada test set
#   7. Simpan semua model & hasil prediksi
#
# Cara menjalankan:
#   python 02_modeling.py
#   (Jalankan SETELAH 01_preprocessing.py)
#
# Output:
#   outputs/models/gbm_base.pkl
#   outputs/models/cqr_models.pkl
#   outputs/models/cqr_calibration.pkl
#   outputs/reports/hasil_prediksi.csv
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_CLEAN_PATH, MODEL_BASE_PATH, MODEL_CQR_PATH,
    MODEL_CAL_PATH, REPORT_PRED_PATH,
    FEATURE_COLS, TARGET_COL,
    RANDOM_SEED, TRAIN_SIZE, CAL_SIZE, TEST_SIZE,
    ALPHA_LEVELS, GBM_PARAMS, QUANTILE_GBM_PARAMS,
    create_dirs
)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# FUNGSI-FUNGSI UTAMA
# ============================================================

def split_data(df):
    """
    Split data menjadi 3 subset: Train / Calibration / Test.

    Kenapa 3 subset (bukan 2)?
    → CQR membutuhkan Calibration Set yang TERPISAH dari Training Set.
      Jika dicampur, jaminan coverage statistik CQR tidak berlaku secara teoritis.

    Proporsi: 60% / 20% / 20%

    Args:
        df: DataFrame bersih

    Returns:
        tuple: X_train, X_cal, X_test, y_train, y_cal, y_test
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Split pertama: pisahkan test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # Split kedua: dari sisa data, pisahkan calibration set
    # 0.25 × 80% = 20% dari total
    cal_ratio = CAL_SIZE / (TRAIN_SIZE + CAL_SIZE)  # = 0.25
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp,
        test_size=cal_ratio,
        random_state=RANDOM_SEED
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def train_base_model(X_train, y_train):
    """
    Training Gradient Boosting base model untuk point prediction.

    Fungsi base model dalam CQR:
    → Selain untuk prediksi titik, base model juga digunakan
      untuk membandingkan performa (R², MAE, RMSE) sebagai
      dasar sebelum ditambahkan lapisan conformal.

    Args:
        X_train: Feature matrix training set
        y_train: Target vector training set

    Returns:
        GradientBoostingRegressor: Model yang sudah dilatih
    """
    print("  Training Gradient Boosting base model...")
    print(f"  Parameter: {GBM_PARAMS}")

    start = time.time()
    model = GradientBoostingRegressor(**GBM_PARAMS)
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  ✅ Selesai dalam {elapsed:.1f} detik")
    return model


def train_quantile_model(X_train, y_train, alpha_quantile):
    """
    Training satu model quantile regression.

    Quantile regression memprediksi kuantil tertentu dari
    distribusi target, bukan nilai rata-rata.

    Contoh:
        alpha_quantile=0.05 → prediksi batas bawah 5% distribusi
        alpha_quantile=0.95 → prediksi batas atas 95% distribusi

    Args:
        X_train: Feature matrix
        y_train: Target vector
        alpha_quantile: Quantile target (0.0 – 1.0)

    Returns:
        GradientBoostingRegressor: Model quantile yang sudah dilatih
    """
    params = {**QUANTILE_GBM_PARAMS, 'alpha': alpha_quantile}
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def hitung_conformal_scores(model_low, model_high, X_cal, y_cal):
    """
    Hitung conformal nonconformity scores pada calibration set.

    Formula CQR (Romano et al., 2019):
        score_i = max( q̂_low(x_i) - y_i, y_i - q̂_high(x_i) )

    Interpretasi:
        - score > 0: y_i berada DI LUAR interval prediksi
        - score < 0: y_i berada DI DALAM interval prediksi
        - Semakin besar score, semakin jauh y_i dari interval

    Args:
        model_low : Model quantile bawah (q_α/2)
        model_high: Model quantile atas (q_{1-α/2})
        X_cal     : Feature calibration set
        y_cal     : Target calibration set

    Returns:
        np.ndarray: Array conformal scores
    """
    pred_low  = model_low.predict(X_cal)
    pred_high = model_high.predict(X_cal)
    scores = np.maximum(pred_low - y_cal, y_cal - pred_high)
    return scores


def hitung_q_hat(scores, alpha):
    """
    Hitung nilai penyesuaian konformal (q̂) dari conformal scores.

    Formula:
        q̂ = quantile ke-⌈(n+1)(1-α)⌉/n dari distribusi scores

    Kenapa pakai formula ini?
    → Ini adalah versi finite-sample yang memberikan jaminan coverage
      tepat (1-α) secara teoritis, berbeda dari quantile biasa.

    Args:
        scores: Array conformal scores dari calibration set
        alpha : Level signifikansi (0.10 = 90% coverage)

    Returns:
        float: Nilai q̂ yang akan ditambahkan ke interval
    """
    n = len(scores)
    # Level quantile yang dikoreksi untuk finite-sample guarantee
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # Pastikan tidak melebihi 1.0
    q_hat   = float(np.quantile(scores, q_level))
    return q_hat


def prediksi_interval(model_low, model_high, q_hat, X_new):
    """
    Hasilkan interval prediksi CQR untuk data baru.

    Formula:
        batas_bawah = q̂_low(x) - q̂
        batas_atas  = q̂_high(x) + q̂

    Jaminan:
        P(y ∈ [batas_bawah, batas_atas]) ≥ 1 - α

    Args:
        model_low : Model quantile bawah
        model_high: Model quantile atas
        q_hat     : Penyesuaian konformal
        X_new     : Data baru yang akan diprediksi

    Returns:
        tuple: (batas_bawah, batas_atas)
    """
    low  = model_low.predict(X_new) - q_hat
    high = model_high.predict(X_new) + q_hat

    # Clip batas bawah ke 0 (harga tidak boleh negatif)
    low = np.maximum(low, 0)

    return low, high


def evaluasi_interval(y_true, low, high, alpha):
    """
    Hitung metrik evaluasi interval prediksi CQR.

    Metrik:
        PICP  : Prediction Interval Coverage Probability
                = % sampel yang masuk interval
                Harus mendekati (1-α)

        MPIW  : Mean Prediction Interval Width
                = rata-rata lebar interval
                Semakin kecil semakin informatif

        CWC   : Coverage Width-based Criterion
                = trade-off antara coverage dan lebar
                Semakin kecil semakin baik

    Args:
        y_true: Nilai aktual
        low   : Batas bawah interval
        high  : Batas atas interval
        alpha : Level signifikansi

    Returns:
        dict: Semua metrik evaluasi
    """
    covered = (y_true >= low) & (y_true <= high)
    picp    = float(np.mean(covered))
    widths  = high - low
    mpiw    = float(np.mean(widths))
    median_w = float(np.median(widths))

    # CWC dengan penalty factor eta=50
    eta = 50
    target_cov = 1 - alpha
    cwc = mpiw * (1 + (picp < target_cov) * np.exp(-eta * (picp - target_cov)))

    return {
        'picp'          : picp,
        'mpiw'          : mpiw,
        'median_width'  : median_w,
        'cwc'           : float(cwc),
        'n_covered'     : int(np.sum(covered)),
        'n_total'       : len(y_true),
    }


# ============================================================
# PIPELINE MODELING UTAMA
# ============================================================

def jalankan_modeling():
    """
    Pipeline modeling lengkap: split → train → calibrate → predict → save.
    """
    print("=" * 60)
    print("  TAHAP 2: MODELING — CQR + GRADIENT BOOSTING")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    print("\n📂 Load data bersih...")
    if not os.path.exists(DATA_CLEAN_PATH):
        print(f"❌ ERROR: {DATA_CLEAN_PATH} tidak ditemukan.")
        print("   Jalankan dulu: python 01_preprocessing.py")
        sys.exit(1)

    df = pd.read_csv(DATA_CLEAN_PATH)
    print(f"   ✅ Loaded: {len(df):,} baris, {len(FEATURE_COLS)} fitur")
    print(f"   Fitur: {FEATURE_COLS}")

    # ── Split data ────────────────────────────────────────────
    print("\n✂️  Split data 60/20/20...")
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(df)
    print(f"   Train      : {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Calibration: {len(X_cal):,} ({len(X_cal)/len(df)*100:.1f}%)")
    print(f"   Test       : {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

    # ── Training Base Model ───────────────────────────────────
    print("\n🤖 Training Base Model (Point Prediction)...")
    gbm_base = train_base_model(X_train, y_train)

    # Evaluasi base model
    y_pred_test  = gbm_base.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2   = r2_score(y_test, y_pred_test)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print(f"\n  📊 Evaluasi Base Model (Test Set):")
    print(f"     MAE  : Rp{mae:,.0f}")
    print(f"     RMSE : Rp{rmse:,.0f}")
    print(f"     R²   : {r2:.4f}")
    print(f"     MAPE : {mape:.2f}%")

    # ── Training CQR per Level α ──────────────────────────────
    print("\n🎯 Training Conformalized Quantile Regression (CQR)...")
    print("   Framework: Romano, Patterson & Candès (NeurIPS 2019)")

    hasil_cqr     = {}   # Simpan model per alpha
    hasil_cal     = {}   # Simpan scores & q_hat per alpha
    hasil_pred    = {}   # Simpan prediksi interval per alpha

    for alpha in ALPHA_LEVELS:
        coverage_target = 1 - alpha
        q_lo_val  = alpha / 2
        q_hi_val  = 1 - alpha / 2

        print(f"\n  --- Coverage {coverage_target*100:.0f}% (α={alpha}) ---")

        # Training quantile models
        print(f"  Training quantile {q_lo_val:.3f}...")
        model_low = train_quantile_model(X_train, y_train, q_lo_val)

        print(f"  Training quantile {q_hi_val:.3f}...")
        model_high = train_quantile_model(X_train, y_train, q_hi_val)

        # Kalibrasi konformal pada calibration set
        scores = hitung_conformal_scores(model_low, model_high, X_cal, y_cal)
        q_hat  = hitung_q_hat(scores, alpha)
        print(f"  q̂ (conformal adjustment): Rp{q_hat:,.0f}")

        # Prediksi interval pada test set
        low, high = prediksi_interval(model_low, model_high, q_hat, X_test)

        # Evaluasi interval
        metrics = evaluasi_interval(y_test, low, high, alpha)
        print(f"  PICP   : {metrics['picp']*100:.2f}% (target {coverage_target*100:.0f}%)")
        print(f"  MPIW   : Rp{metrics['mpiw']:,.0f}")
        print(f"  Median : Rp{metrics['median_width']:,.0f}")
        status = "✅ Terpenuhi" if abs(metrics['picp'] - coverage_target) <= 0.015 else "⚠️ Hampir"
        print(f"  Status : {status}")

        # Simpan semua hasil untuk alpha ini
        hasil_cqr[alpha] = {
            'model_low' : model_low,
            'model_high': model_high,
        }
        hasil_cal[alpha] = {
            'scores' : scores,
            'q_hat'  : q_hat,
            'metrics': metrics,
        }
        hasil_pred[alpha] = {
            'low' : low,
            'high': high,
        }

    # ── Simpan Model ──────────────────────────────────────────
    print("\n💾 Menyimpan model...")

    joblib.dump(gbm_base,  MODEL_BASE_PATH)
    print(f"   ✅ Base model: {MODEL_BASE_PATH}")

    joblib.dump(hasil_cqr, MODEL_CQR_PATH)
    print(f"   ✅ CQR models: {MODEL_CQR_PATH}")

    # Simpan kalibrasi (tanpa objek model, hanya data numerik)
    cal_data = {
        alpha: {
            'scores' : data['scores'].tolist(),
            'q_hat'  : data['q_hat'],
            'metrics': data['metrics'],
        }
        for alpha, data in hasil_cal.items()
    }
    joblib.dump(cal_data, MODEL_CAL_PATH)
    print(f"   ✅ Calibration: {MODEL_CAL_PATH}")

    # ── Simpan Hasil Prediksi ─────────────────────────────────
    print("\n📊 Menyimpan hasil prediksi...")

    df_pred = pd.DataFrame({
        'y_actual'          : y_test,
        'y_pred_point'      : y_pred_test,
        # Coverage 90%
        'low_90'            : hasil_pred[0.10]['low'],
        'high_90'           : hasil_pred[0.10]['high'],
        'covered_90'        : (y_test >= hasil_pred[0.10]['low']) & (y_test <= hasil_pred[0.10]['high']),
        'width_90'          : hasil_pred[0.10]['high'] - hasil_pred[0.10]['low'],
        # Coverage 85%
        'low_85'            : hasil_pred[0.15]['low'],
        'high_85'           : hasil_pred[0.15]['high'],
        'covered_85'        : (y_test >= hasil_pred[0.15]['low']) & (y_test <= hasil_pred[0.15]['high']),
        # Coverage 80%
        'low_80'            : hasil_pred[0.20]['low'],
        'high_80'           : hasil_pred[0.20]['high'],
        'covered_80'        : (y_test >= hasil_pred[0.20]['low']) & (y_test <= hasil_pred[0.20]['high']),
    })

    df_pred.to_csv(REPORT_PRED_PATH, index=False)
    print(f"   ✅ Prediksi: {REPORT_PRED_PATH}")

    # ── Simpan Summary Metrics ────────────────────────────────
    summary = {
        'base_model': {
            'MAE' : float(mae),
            'RMSE': float(rmse),
            'R2'  : float(r2),
            'MAPE': float(mape),
        },
        'cqr_results': {
            str(alpha): {
                'coverage_target': 1 - alpha,
                'q_hat'          : hasil_cal[alpha]['q_hat'],
                **hasil_cal[alpha]['metrics'],
            }
            for alpha in ALPHA_LEVELS
        },
        'data_split': {
            'train'      : len(X_train),
            'calibration': len(X_cal),
            'test'       : len(X_test),
        }
    }

    metrics_path = os.path.join(os.path.dirname(REPORT_PRED_PATH), 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Metrics: {metrics_path}")

    # ── Feature Importance ────────────────────────────────────
    print("\n📌 Feature Importance:")
    fi = gbm_base.feature_importances_
    fi_sorted = sorted(zip(FEATURE_COLS, fi), key=lambda x: -x[1])
    for nama, imp in fi_sorted:
        bar = '█' * int(imp * 50)
        novelty = " ⭐ NOVELTY" if 'Depresiasi' in nama else ""
        print(f"   {nama:25s}: {imp:.4f}  {bar}{novelty}")

    # ── Ringkasan Final ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RINGKASAN HASIL MODELING")
    print("=" * 60)
    print(f"  Base Model R² : {r2:.4f}")
    print(f"  Base Model MAE: Rp{mae:,.0f}")
    for alpha in ALPHA_LEVELS:
        m = hasil_cal[alpha]['metrics']
        print(f"  CQR {(1-alpha)*100:.0f}%: PICP={m['picp']*100:.2f}% | MPIW=Rp{m['mpiw']:,.0f}")
    print(f"\n  ✅ MODELING SELESAI!")
    print("=" * 60)

    return gbm_base, hasil_cqr, hasil_cal, df_pred


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    create_dirs()
    gbm, cqr_models, cqr_cal, df_result = jalankan_modeling()
    print(f"\nSample prediksi (10 baris pertama):")
    cols_show = ['y_actual', 'y_pred_point', 'low_90', 'high_90', 'covered_90']
    print(df_result[cols_show].head(10).to_string())
