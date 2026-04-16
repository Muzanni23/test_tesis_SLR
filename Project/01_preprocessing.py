# ============================================================
# 01_preprocessing.py
# TAHAP 1: Preprocessing & Feature Engineering
#
# Apa yang dilakukan file ini:
#   1. Membaca data mentah Excel dari DJKN
#   2. Membersihkan data (anomali, outlier, missing values)
#   3. Standarisasi kolom kategorikal (warna, merek)
#   4. Ekstrak wilayah dari nama KPKNL
#   5. Feature Engineering: Usia + Faktor Depresiasi (NOVELTY - PMK)
#   6. Label Encoding semua kolom kategorikal
#   7. Simpan dataset bersih + encoding mapping
#
# Cara menjalankan:
#   python 01_preprocessing.py
#
# Output:
#   data/processed/data_bersih.csv
#   data/processed/encoding_map.json
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import sys

# ── Load konfigurasi ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_RAW_PATH, DATA_CLEAN_PATH, ENCODING_PATH,
    TAHUN_MIN, TAHUN_MAX, USIA_MIN, USIA_MAX,
    OUTLIER_LOWER_PCT, OUTLIER_UPPER_PCT,
    TARIF_DEPRESIASI, NILAI_RESIDU_MIN,
    WARNA_MAP, MEREK_MAP, WILAYAH_MAP,
    create_dirs
)

from sklearn.preprocessing import LabelEncoder


# ============================================================
# FUNGSI-FUNGSI HELPER
# ============================================================

def parse_harga(nilai_str):
    """
    Konversi string harga 'Rp3,555,557.00' → float 3555557.0

    Args:
        nilai_str: String harga dalam format Rupiah

    Returns:
        float: Nilai numerik harga
    """
    return (
        str(nilai_str)
        .replace('Rp', '')
        .replace(',', '')
        .strip()
    )


def standarisasi_warna(warna):
    """
    Normalisasi 605 variasi warna → 11 kategori standar.

    Contoh:
        'Hitam Metalik' → 'Hitam'
        'BIRU TUA MET'  → 'Biru'
        '-'             → 'Lainnya'

    Args:
        warna: String warna asli

    Returns:
        str: Kategori warna standar
    """
    if pd.isna(warna) or str(warna).strip() in ['-', '']:
        return 'Lainnya'

    w = str(warna).lower().strip()

    for keyword, label in WARNA_MAP.items():
        if keyword in w:
            return label

    return 'Lainnya'


def standarisasi_merek(merek):
    """
    Normalisasi variasi penulisan merek → nama standar.

    Contoh:
        'HONDA'   → 'Honda'
        'Bmw'     → 'BMW'
        'Ariel Motor' → 'Lainnya'

    Args:
        merek: String merek asli

    Returns:
        str: Nama merek terstandarisasi
    """
    m = str(merek).strip()

    for key, val in MEREK_MAP.items():
        if key.lower() in m.lower():
            return val

    return 'Lainnya'


def ekstrak_wilayah(nama_kpknl):
    """
    Ekstrak wilayah geografis dari nama KPKNL.

    Contoh:
        'KPKNL Jakarta II'   → 'Jawa'
        'KPKNL Denpasar'     → 'Bali-NTT-NTB'
        'KPKNL Ternate'      → 'Papua-Maluku'

    Args:
        nama_kpknl: String nama KPKNL lengkap

    Returns:
        str: Label wilayah geografis
    """
    k = str(nama_kpknl).replace('KPKNL', '').lower().strip()

    for wilayah, keywords in WILAYAH_MAP.items():
        for kw in keywords:
            if kw in k:
                return wilayah

    return 'Lainnya'


def hitung_faktor_depresiasi(row):
    """
    *** NOVELTY UTAMA PENELITIAN ***

    Hitung faktor depresiasi kendaraan berdasarkan regulasi PMK
    menggunakan metode Double Declining Balance (DDB).

    Formula:
        Faktor = max( (1 - tarif_depresiasi) ^ usia, nilai_residu_min )

    Penjelasan:
        - tarif_depresiasi: 20%/tahun untuk Mobil, 25%/tahun untuk Motor
          (berdasarkan standar penilaian kendaraan & PMK)
        - nilai_residu_min: 5% nilai awal (kendaraan selalu punya nilai sisa)
        - DDB dipilih karena lebih realistis: kendaraan menyusut lebih cepat
          di tahun-tahun awal dibanding metode garis lurus

    Contoh:
        Mobil, usia 10 tahun → (1-0.20)^10 = 0.107 → 10.7% nilai awal tersisa
        Motor, usia 5 tahun  → (1-0.25)^5  = 0.237 → 23.7% nilai awal tersisa

    Args:
        row: pandas Series dengan kolom 'Objek' dan 'Usia_Saat_Lelang'

    Returns:
        float: Faktor depresiasi (0.05 hingga 1.0)
    """
    usia = row['Usia_Saat_Lelang']
    objek = row['Objek']

    tarif = TARIF_DEPRESIASI.get(objek, 0.20)  # default 20% jika tidak dikenal
    nilai_sisa = (1 - tarif) ** usia
    nilai_sisa = max(nilai_sisa, NILAI_RESIDU_MIN)  # floor di 5%

    return round(nilai_sisa, 4)


# ============================================================
# PIPELINE PREPROCESSING UTAMA
# ============================================================

def jalankan_preprocessing():
    """
    Pipeline preprocessing lengkap dari raw data → clean data.

    Steps:
        1.  Load data
        2.  Parse harga
        3.  Seleksi & rename kolom
        4.  Hitung usia kendaraan
        5.  Filter anomali tahun
        6.  Hapus outlier harga
        7.  Standarisasi warna
        8.  Standarisasi merek
        9.  Ekstrak wilayah
        10. Feature Engineering: Faktor Depresiasi (NOVELTY)
        11. Label Encoding
        12. Simpan hasil

    Returns:
        pd.DataFrame: Dataset bersih siap modeling
        dict: Mapping encoding
    """

    print("=" * 60)
    print("  TAHAP 1: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    # ── STEP 1: Load data ─────────────────────────────────────
    print("\n📂 STEP 1: Load data mentah...")
    if not os.path.exists(DATA_RAW_PATH):
        print(f"❌ ERROR: File tidak ditemukan: {DATA_RAW_PATH}")
        print("   Taruh file Excel DJKN ke folder data/raw/")
        sys.exit(1)

    df = pd.read_excel(DATA_RAW_PATH)
    print(f"   ✅ Loaded: {len(df):,} baris, {len(df.columns)} kolom")
    print(f"   Kolom: {df.columns.tolist()}")

    # ── STEP 2: Parse harga ───────────────────────────────────
    print("\n💰 STEP 2: Parse kolom Harga Laku...")
    df['Harga_Numeric'] = df['Harga Laku'].apply(parse_harga).astype(float)
    print(f"   ✅ Min: Rp{df['Harga_Numeric'].min():,.0f}")
    print(f"   ✅ Max: Rp{df['Harga_Numeric'].max():,.0f}")

    # ── STEP 3: Seleksi & rename kolom ───────────────────────
    print("\n🗂️  STEP 3: Seleksi dan rename kolom...")
    df = df.drop(columns=['Harga Laku', 'Kat. Lokasi (Semester II 2025)'])
    df = df.rename(columns={'Kat. Lokasi (Semester I 2026)': 'Kat_Lokasi'})
    print(f"   ✅ Kolom setelah seleksi: {df.columns.tolist()}")

    # ── STEP 4: Hitung usia kendaraan ─────────────────────────
    print("\n🕰️  STEP 4: Hitung usia kendaraan saat dilelang...")
    df['Usia_Saat_Lelang'] = df['Tahun Lelang'] - df['Tahun Pembuatan']
    print(f"   ✅ Usia range: {df['Usia_Saat_Lelang'].min()} – {df['Usia_Saat_Lelang'].max()} tahun")

    # ── STEP 5: Filter anomali tahun ─────────────────────────
    print("\n🔍 STEP 5: Filter anomali tahun pembuatan...")
    n_before = len(df)
    mask_valid = (
        (df['Tahun Pembuatan'] >= TAHUN_MIN) &
        (df['Tahun Pembuatan'] <= TAHUN_MAX) &
        (df['Usia_Saat_Lelang'] >= USIA_MIN) &
        (df['Usia_Saat_Lelang'] <= USIA_MAX)
    )
    df = df[mask_valid].copy()
    print(f"   ✅ Dihapus: {n_before - len(df)} baris anomali")
    print(f"   ✅ Tersisa: {len(df):,} baris")

    # ── STEP 6: Hapus outlier harga ──────────────────────────
    print("\n📊 STEP 6: Hapus outlier harga (IQR per Objek)...")
    n_before = len(df)
    hasil_bersih = []

    for objek, grup in df.groupby('Objek'):
        q_lo = grup['Harga_Numeric'].quantile(OUTLIER_LOWER_PCT)
        q_hi = grup['Harga_Numeric'].quantile(OUTLIER_UPPER_PCT)
        iqr  = q_hi - q_lo
        lo   = q_lo - 1.5 * iqr
        hi   = q_hi + 1.5 * iqr
        mask = (grup['Harga_Numeric'] >= lo) & (grup['Harga_Numeric'] <= hi)
        grup_bersih = grup[mask]
        hasil_bersih.append(grup_bersih)
        print(f"   {objek}: {len(grup):,} → {len(grup_bersih):,} "
              f"| range: Rp{grup_bersih['Harga_Numeric'].min():,.0f} – "
              f"Rp{grup_bersih['Harga_Numeric'].max():,.0f}")

    df = pd.concat(hasil_bersih).reset_index(drop=True)
    print(f"   ✅ Dihapus: {n_before - len(df)} outlier")
    print(f"   ✅ Tersisa: {len(df):,} baris")

    # ── STEP 7: Standarisasi warna ────────────────────────────
    print("\n🎨 STEP 7: Standarisasi warna...")
    df['Warna_Std'] = df['Warna'].apply(standarisasi_warna)
    print(f"   Sebelum: {df['Warna'].nunique()} nilai unik")
    print(f"   Sesudah: {df['Warna_Std'].nunique()} nilai unik")
    print(f"   Distribusi:\n{df['Warna_Std'].value_counts().to_string()}")

    # ── STEP 8: Standarisasi merek ────────────────────────────
    print("\n🚗 STEP 8: Standarisasi merek...")
    df['Merek_Std'] = df['Merek'].apply(standarisasi_merek)
    print(f"   Sebelum: {df['Merek'].nunique()} nilai unik")
    print(f"   Sesudah: {df['Merek_Std'].nunique()} nilai unik")
    print(f"   Top 10:\n{df['Merek_Std'].value_counts().head(10).to_string()}")

    # ── STEP 9: Ekstrak wilayah ───────────────────────────────
    print("\n🗺️  STEP 9: Ekstrak wilayah dari nama KPKNL...")
    df['Wilayah'] = df['KPKNL'].apply(ekstrak_wilayah)
    print(f"   Distribusi wilayah:\n{df['Wilayah'].value_counts().to_string()}")

    # ── STEP 10: Feature Engineering — NOVELTY ───────────────
    print("\n⭐ STEP 10: Feature Engineering — Faktor Depresiasi (NOVELTY PMK)...")
    df['Faktor_Depresiasi'] = df.apply(hitung_faktor_depresiasi, axis=1)
    print(f"   Formula: Faktor = max((1-tarif)^usia, {NILAI_RESIDU_MIN})")
    print(f"   Tarif: Mobil={TARIF_DEPRESIASI['Mobil']}, Motor={TARIF_DEPRESIASI['Motor']}")
    print(f"   Statistik Faktor Depresiasi:")
    print(df.groupby('Objek')['Faktor_Depresiasi'].describe().round(4).to_string())

    # ── STEP 11: Rename kolom tahun ───────────────────────────
    df = df.rename(columns={
        'Tahun Pembuatan': 'Tahun_Pembuatan',
        'Tahun Lelang'   : 'Tahun_Lelang',
    })

    # ── STEP 11: Label Encoding ───────────────────────────────
    print("\n🔢 STEP 11: Label Encoding kolom kategorikal...")

    le_merek   = LabelEncoder()
    le_warna   = LabelEncoder()
    le_wilayah = LabelEncoder()

    # Binary encoding untuk Objek
    df['Objek_Enc']   = (df['Objek'] == 'Mobil').astype(int)  # 1=Mobil, 0=Motor

    # Label encoding untuk kategorikal multi-kelas
    df['Merek_Enc']   = le_merek.fit_transform(df['Merek_Std'])
    df['Warna_Enc']   = le_warna.fit_transform(df['Warna_Std'])
    df['Wilayah_Enc'] = le_wilayah.fit_transform(df['Wilayah'])

    # Simpan mapping encoding untuk dipakai saat inference nanti
    encoding_map = {
        'Objek'  : {'Mobil': 1, 'Motor': 0},
        'Merek'  : dict(zip(
            le_merek.classes_.tolist(),
            le_merek.transform(le_merek.classes_).tolist()
        )),
        'Warna'  : dict(zip(
            le_warna.classes_.tolist(),
            le_warna.transform(le_warna.classes_).tolist()
        )),
        'Wilayah': dict(zip(
            le_wilayah.classes_.tolist(),
            le_wilayah.transform(le_wilayah.classes_).tolist()
        )),
    }

    print(f"   Merek  : {list(encoding_map['Merek'].keys())}")
    print(f"   Warna  : {list(encoding_map['Warna'].keys())}")
    print(f"   Wilayah: {list(encoding_map['Wilayah'].keys())}")

    # ── STEP 12: Finalisasi & Simpan ─────────────────────────
    print("\n💾 STEP 12: Finalisasi dan simpan dataset...")

    # Pilih kolom final untuk disimpan
    kolom_final = [
        'Kode', 'Objek', 'Objek_Enc',
        'KPKNL', 'Wilayah', 'Wilayah_Enc',
        'Merek_Std', 'Merek_Enc',
        'Tipe',
        'Tahun_Pembuatan', 'Tahun_Lelang',
        'Usia_Saat_Lelang', 'Faktor_Depresiasi',
        'Warna_Std', 'Warna_Enc',
        'Kat_Lokasi',
        'Harga_Numeric',
    ]
    df_final = df[kolom_final].copy()

    # Simpan dataset bersih
    df_final.to_csv(DATA_CLEAN_PATH, index=False)
    print(f"   ✅ Dataset bersih disimpan: {DATA_CLEAN_PATH}")

    # Simpan encoding mapping
    with open(ENCODING_PATH, 'w', encoding='utf-8') as f:
        json.dump(encoding_map, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Encoding map disimpan: {ENCODING_PATH}")

    # ── Ringkasan Final ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RINGKASAN HASIL PREPROCESSING")
    print("=" * 60)
    print(f"  Total baris bersih : {len(df_final):,}")
    print(f"  Distribusi Objek   : {df_final['Objek'].value_counts().to_dict()}")
    print(f"  Kolom final        : {len(df_final.columns)} kolom")
    print(f"  Harga Mobil median : Rp{df_final[df_final['Objek']=='Mobil']['Harga_Numeric'].median():,.0f}")
    print(f"  Harga Motor median : Rp{df_final[df_final['Objek']=='Motor']['Harga_Numeric'].median():,.0f}")
    print(f"\n  ✅ PREPROCESSING SELESAI!")
    print("=" * 60)

    return df_final, encoding_map


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    create_dirs()
    df_bersih, enc_map = jalankan_preprocessing()
    print(f"\nSample 5 baris pertama:")
    print(df_bersih.head().to_string())
