# 🏛️ Estimasi Interval Harga Limit Lelang Kendaraan Dinas BMN
## Conformalized Quantile Regression (CQR) + Gradient Boosting + Fitur Depresiasi PMK

**Peneliti:** Ahmad Muzanni | NIM: 251012000041  
**Universitas:** Universitas Pamulang | Program S2 Teknik Informatika | 2026

---

## 📁 Struktur Proyek

```
bmn_project/
│
├── README.md                    ← Panduan ini
├── requirements.txt             ← Library yang dibutuhkan
│
├── data/
│   ├── raw/                     ← Data mentah dari DJKN (taruh file Excel di sini)
│   └── processed/               ← Data bersih hasil preprocessing (auto-generated)
│
├── outputs/
│   ├── models/                  ← Model yang sudah dilatih (auto-generated)
│   ├── reports/                 ← Laporan Excel & CSV hasil (auto-generated)
│   └── figures/                 ← Grafik & visualisasi (auto-generated)
│
├── 01_preprocessing.py          ← TAHAP 1: Preprocessing & Feature Engineering
├── 02_modeling.py               ← TAHAP 2: Training Base Model + CQR
├── 03_evaluation.py             ← TAHAP 3: Evaluasi & Visualisasi
├── 04_predict_new.py            ← TAHAP 4: Prediksi kendaraan baru (inference)
└── config.py                    ← Konfigurasi global (path, parameter)
```

---

## 🚀 Cara Menjalankan (Urutan Wajib)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan data
- Taruh file Excel dari DJKN ke folder `data/raw/`
- Rename file menjadi: `Data_Lelang_Kendaraan_DJKN.xlsx`

### 3. Jalankan tahap demi tahap
```bash
# Tahap 1: Preprocessing
python 01_preprocessing.py

# Tahap 2: Modeling (butuh ~3-5 menit)
python 02_modeling.py

# Tahap 3: Evaluasi & Visualisasi
python 03_evaluation.py

# Tahap 4: Prediksi kendaraan baru
python 04_predict_new.py
```

---

## 📊 Output yang Dihasilkan

| File Output | Isi |
|---|---|
| `data/processed/data_bersih.csv` | Dataset bersih 27.573 baris |
| `data/processed/encoding_map.json` | Tabel encoding kategorikal |
| `outputs/models/gbm_base.pkl` | Base model GBM tersimpan |
| `outputs/models/cqr_models.pkl` | Model CQR (q_low + q_high) per α |
| `outputs/models/cqr_calibration.pkl` | Conformal scores & q_hat |
| `outputs/reports/hasil_prediksi.csv` | Prediksi interval test set |
| `outputs/reports/metrics_summary.json` | Semua metrik evaluasi |
| `outputs/figures/` | 6 grafik visualisasi |

---

## 📌 Catatan Penting

- **Jangan ubah urutan** menjalankan file (01 → 02 → 03 → 04)
- Tahap 02 membutuhkan waktu paling lama (~3-5 menit)
- Semua parameter bisa diubah di `config.py`
- Seed random = 42 (reproducible results)
