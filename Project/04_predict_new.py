# ============================================================
# 04_predict_new.py
# TAHAP 4: Prediksi Kendaraan Baru (Inference)
#
# Apa yang dilakukan file ini:
#   Menerima input spesifikasi kendaraan baru dan menghasilkan
#   interval estimasi harga limit lelang menggunakan model
#   yang sudah dilatih.
#
# Cara menjalankan:
#   python 04_predict_new.py
#   (Jalankan SETELAH 02_modeling.py)
#
# Cara Penggunaan:
#   Edit bagian CONTOH KENDARAAN di bawah dengan data kendaraan
#   yang ingin diprediksi, lalu jalankan file ini.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_BASE_PATH, MODEL_CQR_PATH, MODEL_CAL_PATH,
    ENCODING_PATH, FEATURE_COLS, ALPHA_LEVELS,
    TARIF_DEPRESIASI, NILAI_RESIDU_MIN, create_dirs
)


# ============================================================
# KELAS PREDIKTOR
# ============================================================

class BMNPricePredictor:
    """
    Kelas untuk memprediksi interval harga limit lelang kendaraan dinas BMN.

    Cara pakai:
        predictor = BMNPricePredictor()
        predictor.load_models()
        hasil = predictor.predict({...})
        predictor.tampilkan_hasil(hasil)
    """

    def __init__(self):
        self.gbm       = None
        self.cqr_models = None
        self.cal_data  = None
        self.enc_map   = None
        self.loaded    = False

    def load_models(self):
        """Load semua model dan encoding dari file."""
        print("📂 Loading model...")

        for path, nama in [
            (MODEL_BASE_PATH, 'Base model'),
            (MODEL_CQR_PATH,  'CQR models'),
            (MODEL_CAL_PATH,  'Calibration'),
            (ENCODING_PATH,   'Encoding map'),
        ]:
            if not os.path.exists(path):
                print(f"❌ ERROR: {path} tidak ditemukan.")
                print("   Jalankan dulu: python 02_modeling.py")
                sys.exit(1)

        self.gbm        = joblib.load(MODEL_BASE_PATH)
        self.cqr_models = joblib.load(MODEL_CQR_PATH)
        self.cal_data   = joblib.load(MODEL_CAL_PATH)

        with open(ENCODING_PATH, encoding='utf-8') as f:
            self.enc_map = json.load(f)

        self.loaded = True
        print("   ✅ Semua model berhasil di-load!")

    def encode_input(self, kendaraan: dict) -> np.ndarray:
        """
        Konversi input kendaraan (dict) → array numerik untuk model.

        Proses:
            1. Encode Objek: Mobil=1, Motor=0
            2. Encode Merek: lookup dari encoding map
            3. Hitung Usia = Tahun_Lelang - Tahun_Pembuatan
            4. Hitung Faktor_Depresiasi (NOVELTY PMK)
            5. Encode Warna & Wilayah: lookup dari encoding map
            6. Susun sesuai urutan FEATURE_COLS

        Args:
            kendaraan: Dict dengan key-key berikut:
                - objek       : 'Mobil' atau 'Motor'
                - merek       : nama merek terstandar
                - tahun_buat  : tahun pembuatan kendaraan
                - tahun_lelang: tahun akan dilelang
                - kat_lokasi  : kategori lokasi KPKNL (1-4)
                - warna       : warna standar kendaraan
                - wilayah     : wilayah KPKNL

        Returns:
            np.ndarray: Array 1D siap masuk model
        """
        # Encode Objek
        objek_enc = self.enc_map['Objek'].get(kendaraan['objek'], 0)

        # Encode Merek (default ke 'Lainnya' jika tidak dikenal)
        merek_std = kendaraan['merek']
        if merek_std not in self.enc_map['Merek']:
            print(f"   ⚠️  Merek '{merek_std}' tidak dikenal, menggunakan 'Lainnya'")
            merek_std = 'Lainnya'
        merek_enc = self.enc_map['Merek'][merek_std]

        # Hitung usia
        usia = kendaraan['tahun_lelang'] - kendaraan['tahun_buat']

        # Hitung Faktor Depresiasi (NOVELTY)
        tarif = TARIF_DEPRESIASI.get(kendaraan['objek'], 0.20)
        faktor_dep = max((1 - tarif) ** usia, NILAI_RESIDU_MIN)

        # Encode Warna
        warna_std = kendaraan.get('warna', 'Lainnya')
        if warna_std not in self.enc_map['Warna']:
            warna_std = 'Lainnya'
        warna_enc = self.enc_map['Warna'][warna_std]

        # Encode Wilayah
        wilayah = kendaraan.get('wilayah', 'Jawa')
        if wilayah not in self.enc_map['Wilayah']:
            wilayah = 'Jawa'
        wilayah_enc = self.enc_map['Wilayah'][wilayah]

        # Susun sesuai FEATURE_COLS
        # ['Objek_Enc', 'Merek_Enc', 'Tahun_Pembuatan', 'Tahun_Lelang',
        #  'Usia_Saat_Lelang', 'Faktor_Depresiasi', 'Kat_Lokasi',
        #  'Warna_Enc', 'Wilayah_Enc']
        fitur_array = np.array([[
            objek_enc,
            merek_enc,
            kendaraan['tahun_buat'],
            kendaraan['tahun_lelang'],
            usia,
            round(faktor_dep, 4),
            kendaraan.get('kat_lokasi', 3),
            warna_enc,
            wilayah_enc,
        ]])

        return fitur_array, {
            'Objek_Enc'        : objek_enc,
            'Merek_Enc'        : merek_enc,
            'Tahun_Pembuatan'  : kendaraan['tahun_buat'],
            'Tahun_Lelang'     : kendaraan['tahun_lelang'],
            'Usia_Saat_Lelang' : usia,
            'Faktor_Depresiasi': round(faktor_dep, 4),
            'Kat_Lokasi'       : kendaraan.get('kat_lokasi', 3),
            'Warna_Enc'        : warna_enc,
            'Wilayah_Enc'      : wilayah_enc,
        }

    def predict(self, kendaraan: dict, alpha: float = 0.10) -> dict:
        """
        Prediksi interval harga limit lelang untuk satu kendaraan.

        Args:
            kendaraan: Dict spesifikasi kendaraan
            alpha: Level signifikansi (0.10=90%, 0.15=85%, 0.20=80%)

        Returns:
            dict: Hasil prediksi lengkap
        """
        if not self.loaded:
            self.load_models()

        # Encode input
        X, fitur_detail = self.encode_input(kendaraan)

        # Prediksi titik (base model)
        y_pred_point = float(self.gbm.predict(X)[0])

        # Prediksi interval (CQR)
        cqr = self.cqr_models[alpha]
        cal = self.cal_data[alpha]

        low  = float(cqr['model_low'].predict(X)[0])  - cal['q_hat']
        high = float(cqr['model_high'].predict(X)[0]) + cal['q_hat']
        low  = max(low, 0)  # Harga tidak boleh negatif

        return {
            'input'           : kendaraan,
            'fitur_encoded'   : fitur_detail,
            'alpha'           : alpha,
            'coverage_target' : 1 - alpha,
            'y_pred_point'    : y_pred_point,
            'interval_low'    : low,
            'interval_high'   : high,
            'interval_width'  : high - low,
            'q_hat'           : cal['q_hat'],
        }

    def predict_multi_alpha(self, kendaraan: dict) -> list:
        """
        Prediksi interval untuk semua level α sekaligus.

        Args:
            kendaraan: Dict spesifikasi kendaraan

        Returns:
            list: List hasil prediksi untuk setiap alpha
        """
        return [self.predict(kendaraan, alpha) for alpha in ALPHA_LEVELS]

    def tampilkan_hasil(self, hasil_list: list):
        """
        Cetak hasil prediksi ke konsol dengan format yang rapi.

        Args:
            hasil_list: List hasil dari predict_multi_alpha()
        """
        kendaraan = hasil_list[0]['input']

        print("\n" + "=" * 65)
        print("  HASIL ESTIMASI HARGA LIMIT LELANG KENDARAAN DINAS BMN")
        print("=" * 65)
        print(f"\n  Kendaraan : {kendaraan.get('objek','-')} "
              f"{kendaraan.get('merek','-')} {kendaraan.get('tipe','')}")
        print(f"  Tahun     : {kendaraan.get('tahun_buat','-')}")
        print(f"  Warna     : {kendaraan.get('warna','-')}")
        print(f"  Wilayah   : {kendaraan.get('wilayah','-')}")

        # Detail fitur (termasuk novelty)
        detail = hasil_list[0]['fitur_encoded']
        print(f"\n  Usia saat dilelang  : {detail['Usia_Saat_Lelang']} tahun")
        print(f"  Faktor Depresiasi   : {detail['Faktor_Depresiasi']:.4f} "
              f"({detail['Faktor_Depresiasi']*100:.2f}% nilai awal tersisa)  ⭐ NOVELTY PMK")

        print(f"\n  Prediksi Titik      : Rp{hasil_list[0]['y_pred_point']:,.0f}")

        print(f"\n  {'Coverage':10s} {'Batas Bawah':>18s} {'Batas Atas':>18s} "
              f"{'Lebar Interval':>18s} {'Rekomendasi Limit':>20s}")
        print("  " + "-" * 85)

        for hasil in hasil_list:
            cov   = f"{hasil['coverage_target']*100:.0f}%"
            low   = hasil['interval_low']
            high  = hasil['interval_high']
            width = hasil['interval_width']
            # Rekomendasi: tetapkan limit di batas bawah interval
            rekomendasi = low
            print(f"  {cov:10s} {f'Rp{low:,.0f}':>18s} {f'Rp{high:,.0f}':>18s} "
                  f"{f'Rp{width:,.0f}':>18s} {f'Rp{rekomendasi:,.0f}':>20s}")

        print(f"\n  📌 INTERPRETASI (Coverage 90%):")
        h90 = hasil_list[0]
        print(f"     Dengan keyakinan 90%, kendaraan ini akan terjual")
        print(f"     antara Rp{h90['interval_low']:,.0f} hingga Rp{h90['interval_high']:,.0f}")
        print(f"\n  📌 REKOMENDASI PENETAPAN HARGA LIMIT:")
        print(f"     Tetapkan Harga Limit ≈ Rp{h90['interval_low']:,.0f}")
        print(f"     (= batas bawah interval 90% agar lelang tidak gagal)")
        print("=" * 65)


# ============================================================
# CONTOH PENGGUNAAN
# ============================================================

# ─────────────────────────────────────────────────────────────
# EDIT BAGIAN INI: Masukkan data kendaraan yang ingin diprediksi
# ─────────────────────────────────────────────────────────────

DAFTAR_KENDARAAN = [
    {
        'nama_referensi': 'Contoh 1: Toyota Innova 2010 di KPKNL Jakarta',
        'objek'         : 'Mobil',
        'merek'         : 'Toyota',
        'tipe'          : 'Innova',
        'tahun_buat'    : 2010,
        'tahun_lelang'  : 2025,
        'warna'         : 'Hitam',
        'kat_lokasi'    : 2,              # 2 = Jakarta/kota besar
        'wilayah'       : 'Jawa',
    },
    {
        'nama_referensi': 'Contoh 2: Honda Beat 2019 di KPKNL Makassar',
        'objek'         : 'Motor',
        'merek'         : 'Honda',
        'tipe'          : 'Beat',
        'tahun_buat'    : 2019,
        'tahun_lelang'  : 2025,
        'warna'         : 'Biru',
        'kat_lokasi'    : 3,              # 3 = kota menengah
        'wilayah'       : 'Sulawesi',
    },
    {
        'nama_referensi': 'Contoh 3: Mitsubishi Pajero 2015 di KPKNL Surabaya',
        'objek'         : 'Mobil',
        'merek'         : 'Mitsubishi',
        'tipe'          : 'Pajero Sport',
        'tahun_buat'    : 2015,
        'tahun_lelang'  : 2025,
        'warna'         : 'Hitam',
        'kat_lokasi'    : 2,
        'wilayah'       : 'Jawa',
    },
    {
        'nama_referensi': 'Contoh 4: Yamaha Mio 2012 di KPKNL Pontianak',
        'objek'         : 'Motor',
        'merek'         : 'Yamaha',
        'tipe'          : 'Mio',
        'tahun_buat'    : 2012,
        'tahun_lelang'  : 2025,
        'warna'         : 'Merah',
        'kat_lokasi'    : 3,
        'wilayah'       : 'Kalimantan',
    },
]


def prediksi_batch(daftar_kendaraan: list) -> pd.DataFrame:
    """
    Prediksi untuk banyak kendaraan sekaligus dan simpan ke CSV.

    Args:
        daftar_kendaraan: List dict kendaraan

    Returns:
        pd.DataFrame: Hasil prediksi semua kendaraan
    """
    predictor = BMNPricePredictor()
    predictor.load_models()

    rows = []
    for kend in daftar_kendaraan:
        hasil_list = predictor.predict_multi_alpha(kend)
        row = {
            'Nama Referensi'      : kend.get('nama_referensi', '-'),
            'Objek'               : kend['objek'],
            'Merek'               : kend['merek'],
            'Tipe'                : kend.get('tipe', '-'),
            'Tahun Pembuatan'     : kend['tahun_buat'],
            'Tahun Lelang'        : kend['tahun_lelang'],
            'Usia (tahun)'        : kend['tahun_lelang'] - kend['tahun_buat'],
            'Wilayah'             : kend.get('wilayah', '-'),
            'Faktor Depresiasi'   : hasil_list[0]['fitur_encoded']['Faktor_Depresiasi'],
            'Prediksi Titik (Rp)' : hasil_list[0]['y_pred_point'],
            'Low 90% (Rp)'        : hasil_list[0]['interval_low'],
            'High 90% (Rp)'       : hasil_list[0]['interval_high'],
            'Width 90% (Rp)'      : hasil_list[0]['interval_width'],
            'Low 85% (Rp)'        : hasil_list[1]['interval_low'],
            'High 85% (Rp)'       : hasil_list[1]['interval_high'],
            'Low 80% (Rp)'        : hasil_list[2]['interval_low'],
            'High 80% (Rp)'       : hasil_list[2]['interval_high'],
            'Rekomendasi Limit'   : hasil_list[0]['interval_low'],
        }
        rows.append(row)

    df_hasil = pd.DataFrame(rows)
    return df_hasil


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    create_dirs()

    print("=" * 60)
    print("  TAHAP 4: PREDIKSI KENDARAAN BARU (INFERENCE)")
    print("=" * 60)

    # Inisialisasi prediktor
    predictor = BMNPricePredictor()
    predictor.load_models()

    # Prediksi setiap kendaraan
    for kend in DAFTAR_KENDARAAN:
        print(f"\n{'─'*60}")
        print(f"  {kend['nama_referensi']}")
        hasil_list = predictor.predict_multi_alpha(kend)
        predictor.tampilkan_hasil(hasil_list)

    # Prediksi batch → simpan ke CSV
    print("\n\n📊 Membuat laporan batch...")
    df_batch = prediksi_batch(DAFTAR_KENDARAAN)
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'outputs', 'reports', 'prediksi_kendaraan_baru.csv'
    )
    df_batch.to_csv(output_path, index=False)
    print(f"   ✅ Laporan batch disimpan: {output_path}")
    print("\n" + df_batch.to_string())
