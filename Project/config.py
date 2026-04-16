# ============================================================
# config.py
# Konfigurasi Global Proyek BMN CQR
# Semua parameter bisa diubah di sini tanpa menyentuh file lain
# ============================================================

import os

# ─────────────────────────────────────────────────────────────
# PATH KONFIGURASI
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input
DATA_RAW_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'Data_Lelang_Kendaraan_DJKN.xlsx')

# Processed data
DATA_CLEAN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'data_bersih.csv')
ENCODING_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'encoding_map.json')

# Model output
MODEL_BASE_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'gbm_base.pkl')
MODEL_CQR_PATH  = os.path.join(BASE_DIR, 'outputs', 'models', 'cqr_models.pkl')
MODEL_CAL_PATH  = os.path.join(BASE_DIR, 'outputs', 'models', 'cqr_calibration.pkl')

# Reports
REPORT_PRED_PATH    = os.path.join(BASE_DIR, 'outputs', 'reports', 'hasil_prediksi.csv')
REPORT_METRICS_PATH = os.path.join(BASE_DIR, 'outputs', 'reports', 'metrics_summary.json')

# Figures
FIGURE_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')

# ─────────────────────────────────────────────────────────────
# KOLOM DATASET
# ─────────────────────────────────────────────────────────────
TARGET_COL   = 'Harga_Numeric'          # Variabel target (yang diprediksi)

# Fitur untuk model (9 fitur input)
FEATURE_COLS = [
    'Objek_Enc',           # Jenis kendaraan: 1=Mobil, 0=Motor
    'Merek_Enc',           # Merek kendaraan (encoded)
    'Tahun_Pembuatan',     # Tahun produksi
    'Tahun_Lelang',        # Tahun pelaksanaan lelang
    'Usia_Saat_Lelang',    # Selisih tahun lelang - tahun pembuatan
    'Faktor_Depresiasi',   # *** NOVELTY: (1-tarif)^usia berdasarkan PMK ***
    'Kat_Lokasi',          # Kategori lokasi KPKNL (1-4)
    'Warna_Enc',           # Warna standar (encoded)
    'Wilayah_Enc',         # Wilayah geografis (encoded)
]

# Kolom informasi (tidak masuk model, hanya untuk referensi)
INFO_COLS = ['Kode', 'Objek', 'KPKNL', 'Wilayah', 'Merek_Std', 'Warna_Std', 'Tipe']

# ─────────────────────────────────────────────────────────────
# PARAMETER PREPROCESSING
# ─────────────────────────────────────────────────────────────
TAHUN_MIN           = 1970     # Batas bawah tahun pembuatan yang valid
TAHUN_MAX           = 2024     # Batas atas tahun pembuatan yang valid
USIA_MIN            = 1        # Usia minimum kendaraan saat dilelang (tahun)
USIA_MAX            = 50       # Usia maksimum kendaraan saat dilelang (tahun)
OUTLIER_LOWER_PCT   = 0.05     # Persentil bawah untuk filter outlier harga
OUTLIER_UPPER_PCT   = 0.95     # Persentil atas untuk filter outlier harga

# ─────────────────────────────────────────────────────────────
# PARAMETER FITUR DEPRESIASI (NOVELTY - berbasis PMK)
# ─────────────────────────────────────────────────────────────
TARIF_DEPRESIASI = {
    'Mobil': 0.20,   # 20% per tahun (Double Declining Balance)
    'Motor': 0.25,   # 25% per tahun (lebih cepat menyusut)
}
NILAI_RESIDU_MIN = 0.05   # 5% nilai awal sebagai batas minimum (nilai sisa)

# ─────────────────────────────────────────────────────────────
# PARAMETER MODEL
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Proporsi split data
TRAIN_SIZE = 0.60    # 60% training
CAL_SIZE   = 0.20    # 20% calibration (untuk CQR)
TEST_SIZE  = 0.20    # 20% test (evaluasi akhir)

# Level coverage CQR yang akan dievaluasi
ALPHA_LEVELS = [0.10, 0.15, 0.20]   # → 90%, 85%, 80% coverage

# Parameter Gradient Boosting (base model)
GBM_PARAMS = {
    'n_estimators'    : 500,
    'learning_rate'   : 0.05,
    'max_depth'       : 6,
    'min_samples_split': 20,
    'min_samples_leaf' : 10,
    'subsample'       : 0.8,
    'max_features'    : 0.8,
    'loss'            : 'huber',   # Robust terhadap outlier
    'alpha'           : 0.9,
    'random_state'    : RANDOM_SEED,
}

# Parameter Quantile GBM (untuk CQR)
QUANTILE_GBM_PARAMS = {
    'n_estimators'    : 300,
    'learning_rate'   : 0.05,
    'max_depth'       : 5,
    'min_samples_leaf': 15,
    'subsample'       : 0.8,
    'max_features'    : 0.8,
    'loss'            : 'quantile',   # Wajib untuk quantile regression
    'random_state'    : RANDOM_SEED,
    # 'alpha' akan diset dinamis per level kuantil
}

# ─────────────────────────────────────────────────────────────
# STANDARISASI KATEGORIKAL
# ─────────────────────────────────────────────────────────────
WARNA_MAP = {
    # key: substring yang dicari (lowercase), value: label standar
    'hitam'  : 'Hitam',
    'putih'  : 'Putih',
    'white'  : 'Putih',
    'merah'  : 'Merah',
    'red'    : 'Merah',
    'biru'   : 'Biru',
    'blue'   : 'Biru',
    'silver' : 'Silver/Abu-abu',
    'perak'  : 'Silver/Abu-abu',
    'abu'    : 'Silver/Abu-abu',
    'hijau'  : 'Hijau',
    'green'  : 'Hijau',
    'kuning' : 'Kuning',
    'yellow' : 'Kuning',
    'coklat' : 'Coklat',
    'cokel'  : 'Coklat',
    'brown'  : 'Coklat',
    'orange' : 'Orange',
    'jingga' : 'Orange',
    'ungu'   : 'Ungu',
    'violet' : 'Ungu',
    'purple' : 'Ungu',
}

MEREK_MAP = {
    'Honda'         : 'Honda',
    'Toyota'        : 'Toyota',
    'Suzuki'        : 'Suzuki',
    'Yamaha'        : 'Yamaha',
    'Mitsubishi'    : 'Mitsubishi',
    'Isuzu'         : 'Isuzu',
    'Daihatsu'      : 'Daihatsu',
    'Nissan'        : 'Nissan',
    'Kia'           : 'Kia',
    'Ford'          : 'Ford',
    'Kawasaki'      : 'Kawasaki',
    'Hino'          : 'Hino',
    'Viar'          : 'Viar',
    'Royal Enfield' : 'Royal Enfield',
    'Chevrolet'     : 'Chevrolet',
    'Mercedes'      : 'Mercedes',
    'Hyundai'       : 'Hyundai',
    'Mazda'         : 'Mazda',
    'BMW'           : 'BMW',
    'Bmw'           : 'BMW',
    'Volvo'         : 'Volvo',
    'Jeep'          : 'Jeep',
    'Fiat'          : 'Fiat',
    'Bajaj'         : 'Bajaj',
    'TVS'           : 'TVS',
    'Tvs'           : 'TVS',
}

WILAYAH_MAP = {
    'Jawa'       : ['jakarta', 'bandung', 'bogor', 'bekasi', 'depok', 'tangerang',
                    'surabaya', 'semarang', 'yogyakarta', 'malang', 'solo', 'kediri',
                    'madiun', 'mojokerto', 'pasuruan', 'probolinggo', 'tegal',
                    'purwokerto', 'serang', 'cirebon', 'tasikmalaya', 'pekalongan',
                    'jember', 'blitar', 'tulungagung', 'banyuwangi', 'pamekasan'],
    'Sumatera'   : ['medan', 'padang', 'pekanbaru', 'batam', 'palembang',
                    'banda aceh', 'bengkulu', 'jambi', 'bandar lampung',
                    'pangkal pinang', 'tanjung pinang', 'pematang siantar',
                    'padang sidempuan', 'lubuk linggau', 'metro', 'kotabumi'],
    'Kalimantan' : ['balikpapan', 'banjarmasin', 'pontianak', 'samarinda',
                    'palangkaraya', 'tarakan', 'singkawang', 'pangkalan bun'],
    'Sulawesi'   : ['makassar', 'manado', 'kendari', 'palu', 'gorontalo',
                    'mamuju', 'palopo', 'parepare', 'luwuk', 'bitung'],
    'Bali-NTT-NTB': ['denpasar', 'mataram', 'kupang', 'maumere', 'ruteng',
                      'ende', 'waingapu', 'singaraja'],
    'Papua-Maluku': ['jayapura', 'ambon', 'sorong', 'manokwari', 'merauke',
                     'ternate', 'biak', 'timika', 'fakfak'],
}

# ─────────────────────────────────────────────────────────────
# HELPER: Buat folder jika belum ada
# ─────────────────────────────────────────────────────────────
def create_dirs():
    """Buat semua folder yang diperlukan."""
    dirs = [
        os.path.join(BASE_DIR, 'data', 'raw'),
        os.path.join(BASE_DIR, 'data', 'processed'),
        os.path.join(BASE_DIR, 'outputs', 'models'),
        os.path.join(BASE_DIR, 'outputs', 'reports'),
        os.path.join(BASE_DIR, 'outputs', 'figures'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Struktur folder siap.")

if __name__ == '__main__':
    create_dirs()
    print("Config loaded successfully.")
    print(f"  Features  : {FEATURE_COLS}")
    print(f"  Target    : {TARGET_COL}")
    print(f"  Alpha     : {ALPHA_LEVELS}")
