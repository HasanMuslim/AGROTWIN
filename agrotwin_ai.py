# ============================================================================
#
#   ___  ___  ____  ____  ____  __    __  ____  _  _
#  / __)/ __)(  _ \(  _ \(_  _)/  \  (_  )(_  _)( \/ )
# ( (_ \) _)  )   / )   / _)( ( () )  )(   )(   )  (
#  \___/(____)(_)\_)(_)\_)(____)\__/  (__) (__) (_/\_)
#
#  AGROTWIN — Master Node (Sistem Beyni)
#  Topraksız Tarım için Yapay Zekâ Destekli Dijital İkiz
#
#  Versiyon  : 2.0.0
#  Tarih     : 2025
#  Python    : 3.9+
#
#  Gerekli Kütüphaneler (pip install ...):
#    paho-mqtt        — MQTT istemcisi
#    scikit-learn     — Makine öğrenmesi (DecisionTreeClassifier)
#    pandas           — CSV veri yönetimi
#    requests         — EPİAŞ API HTTP çağrıları
#    eptr2            — EPİAŞ şeffaflık platformu (alternatif)
#
#  Standart kütüphaneler (kurulum gerektirmez):
#    sqlite3, csv, os, json, logging, threading, datetime
#
# ============================================================================

import json
import csv
import os
import logging
import sqlite3
import threading
import time
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import paho.mqtt.client as mqtt
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import Dropout

# ============================================================================
# BÖLÜM 1: YAPILANDIRMA SABİTLERİ
# ============================================================================

# --- MQTT Broker Ayarları (HiveMQ Cloud veya yerel broker) ---
MQTT_SUNUCU    = "broker.hivemq.com"   # <<< HiveMQ Cloud adresinizi girin
MQTT_PORT      = 1883
MQTT_KULLANICI = ""                    # <<< Gerekiyorsa HiveMQ kullanıcı adı
MQTT_SIFRE     = ""                    # <<< Gerekiyorsa HiveMQ şifresi
MQTT_CLIENT_ID = "AGROTWIN-MasterNode-01"

# --- MQTT Kanal (Topic) Tanımları — ESP32 koduyla birebir eşleşir ---
TOPIC_SENSORLER   = "agrotwin/sensorler"   # ESP32'den gelen sensör verileri
TOPIC_KOMUTLAR    = "agrotwin/komutlar"    # ESP32'ye gönderilen röle komutları
TOPIC_AI_LOG      = "agrotwin/ai_log"      # Çiftçiye insan dilinde karar açıklamaları
TOPIC_CHAT_SORU   = "agrotwin/chat/soru"   # Flutter'dan gelen chatbot soruları
TOPIC_CHAT_CEVAP  = "agrotwin/chat/cevap"  # Flutter'a gönderilen chatbot cevapları

# --- Karar Motoru Eşik Değerleri ---
ESIK_HAVA_SICAKLIK_MAX   = 28.0   # °C — Üstünde fan açılmayı tetikler
ESIK_SU_SICAKLIK_MIN     = 20.0   # °C — Altında ısıtıcı açılmayı tetikler
ESIK_SU_MESAFE_KRITIK    = 20.0   # cm — Üstünde pompa kapanır (tank boşluğu)
ESIK_ELEKTRIK_PAHALI     = 2000.0 # TL/MWh — EPİAŞ PTF üstünde "pahalı" sayılır
VARSAYILAN_ELEKTRIK_FIYAT = 1500.0 # TL/MWh — API çekemezse kullanılır

# --- Makine Öğrenmesi Eşiği ---
ML_VERI_ESIGI = 50  # Bu kadar satır dolunca AI otonom moda geçer

# --- Dosya Yolları ---
DATASET_DOSYA  = "C:\\Users\\hasan\\Desktop\\AGROTWİN\\IoTProcessed_Data.csv"   # ML eğitim verisi
VERITABANI     = "agrotwin_data.db"       # SQLite Grafana veritabanı

# --- CSV Sütun Başlıkları ---
CSV_SUTUNLAR = [
    "timestamp", "T_ortam", "H_ortam", "T_su",
    "Su_Mesafe_cm", "Isik_Analog", "elektrik_fiyati",
    "pompa_karar", "fan_karar", "isitici_karar"
]


# ============================================================================
# BÖLÜM 2: LOGLAMA YAPILANDIRMASI
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                        # Terminale yaz
        logging.FileHandler("agrotwin_master.log",      # Dosyaya da yaz
                            encoding="utf-8")
    ]
)
log = logging.getLogger("AGROTWIN")

# ============================================================================
# BÖLÜM 3: PAYLAŞILAN DURUM DEĞİŞKENLERİ (Thread-Safe)
# ============================================================================

# Thread'ler arası veri paylaşımı için kilit (mutex)
veri_kilidi = threading.Lock()

# Son okunan sensör verileri (ESP32'den gelen JSON anahtarlarıyla birebir)
son_sensor_verisi: dict = {
    "T_ortam"     : 25.0,
    "H_ortam"     : 60.0,
    "T_su"        : 22.0,
    "Su_Mesafe_cm": 10.0,
    "Isik_Analog" : 2000,
}

# Son geçerli elektrik fiyatı
son_elektrik_fiyati: float = VARSAYILAN_ELEKTRIK_FIYAT

# Son alınan karar durumları (SQLite ve chatbot için)
son_kararlar: dict = {
    "pompa"  : "BILINMIYOR",
    "fan"    : "BILINMIYOR",
    "isitici": "BILINMIYOR",
}

# ML modeli ve encoder (global — thread içinden de erişilecek)
ml_modeli: Sequential | None = None
ml_encoder_pompa   = LabelEncoder()
ml_encoder_fan     = LabelEncoder()
ml_encoder_isitici = LabelEncoder()
ml_hazir: bool = False  # True olduğunda model.predict() kullanılır
predict_buffer = []

# MQTT istemcisi (global — fonksiyonlarda publish için erişilecek)
mqtt_istemci: mqtt.Client | None = None

# ============================================================================
# BÖLÜM 4: VERİTABANI (SQLite) — BAŞLATMA VE KAYIT
# ============================================================================

def veritabani_baslat() -> None:
    """
    SQLite veritabanını ve sensor_logs tablosunu oluşturur.
    Tablo zaten varsa dokunmaz (CREATE TABLE IF NOT EXISTS).
    Grafana bu tablodaki verileri sorgular.
    """
    with sqlite3.connect(VERITABANI) as baglanti:
        baglanti.execute("""
            CREATE TABLE IF NOT EXISTS sensor_logs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                T_ortam          REAL,
                H_ortam          REAL,
                T_su             REAL,
                Su_Mesafe_cm     REAL,
                Isik_Analog      INTEGER,
                elektrik_fiyati  REAL,
                pompa_karar      TEXT,
                fan_karar        TEXT,
                isitici_karar    TEXT
            )
        """)
        baglanti.commit()
    log.info("[DB] SQLite veritabanı hazır: %s", VERITABANI)


def veritabanina_kaydet(sensor: dict, fiyat: float, kararlar: dict) -> None:
    """
    Sensör verilerini, elektrik fiyatını ve AI kararlarını
    sensor_logs tablosuna kaydeder.

    Args:
        sensor  : ESP32'den gelen filtrelenmiş sensör dict'i
        fiyat   : EPİAŞ'tan alınan güncel elektrik fiyatı (TL/MWh)
        kararlar: Pompa/fan/ısıtıcı için "ON"/"OFF" kararları
    """
    try:
        with sqlite3.connect(VERITABANI) as baglanti:
            baglanti.execute("""
                INSERT INTO sensor_logs
                    (timestamp, T_ortam, H_ortam, T_su,
                     Su_Mesafe_cm, Isik_Analog, elektrik_fiyati,
                     pompa_karar, fan_karar, isitici_karar)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                sensor.get("T_ortam"),
                sensor.get("H_ortam"),
                sensor.get("T_su"),
                sensor.get("Su_Mesafe_cm"),
                sensor.get("Isik_Analog"),
                fiyat,
                kararlar.get("pompa"),
                kararlar.get("fan"),
                kararlar.get("isitici"),
            ))
            baglanti.commit()
    except sqlite3.Error as hata:
        log.error("[DB] Kayıt hatası: %s", hata)

# ============================================================================
# BÖLÜM 5: CSV VERİ KAYDI (ML Eğitim Seti)
# ============================================================================

def csv_kaydet(sensor: dict, fiyat: float, kararlar: dict) -> None:
    """
    Her karar döngüsünden sonra durumu CSV'ye ekler.
    Dosya yoksa başlık satırını da yazar.

    Args:
        sensor  : Sensör verisi dict'i
        fiyat   : Güncel elektrik fiyatı
        kararlar: Alınan röle kararları
    """
    dosya_var = os.path.isfile(DATASET_DOSYA)
    try:
        with open(DATASET_DOSYA, "a", newline="", encoding="utf-8") as dosya:
            yazar = csv.DictWriter(dosya, fieldnames=CSV_SUTUNLAR)
            if not dosya_var:
                yazar.writeheader()  # İlk açılışta başlık yaz
            yazar.writerow({
                "timestamp"      : datetime.now().isoformat(timespec="seconds"),
                "T_ortam"        : sensor.get("T_ortam"),
                "H_ortam"        : sensor.get("H_ortam"),
                "T_su"           : sensor.get("T_su"),
                "Su_Mesafe_cm"   : sensor.get("Su_Mesafe_cm"),
                "Isik_Analog"    : sensor.get("Isik_Analog"),
                "elektrik_fiyati": fiyat,
                "pompa_karar"    : kararlar.get("pompa"),
                "fan_karar"      : kararlar.get("fan"),
                "isitici_karar"  : kararlar.get("isitici"),
            })
    except OSError as hata:
        log.error("[CSV] Yazma hatası: %s", hata)

# ============================================================================
# BÖLÜM 6: EPİAŞ ELEKTRİK FİYATI ÇEKME
# ============================================================================

def epias_fiyat_cek() -> float:
    """Doğrudan EPİAŞ Şeffaflık Platformu REST API'sine POST isteği atar."""
    try:
        simdi = datetime.now()
        tarih_str = simdi.strftime("%Y-%m-%dT00:00:00+03:00")
        
        url = "https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html#_mcp-export"
        headers = {
            "x-epias-token": "https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html#_mcp-export",
            "Content-Type": "application/json"
        }
        payload = {
            "startDate": tarih_str,
            "endDate": tarih_str,
            "region": "TR1"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        veri = response.json()
        
        su_anki_saat = simdi.strftime("%Y-%m-%dT%H:00:00+03:00")
        for item in veri.get("items", []):
            if item.get("date") == su_anki_saat:
                return float(item.get("price"))
        return VARSAYILAN_ELEKTRIK_FIYAT
    except Exception as e:
        print(f"[HATA] EPİAŞ API Bağlantısı: {e}")
        return VARSAYILAN_ELEKTRIK_FIYAT
    
# ============================================================================
# BÖLÜM 7: MQTT KOMUT GÖNDERME
# ============================================================================

def role_komutu_gonder(cihaz: str, durum: str) -> None:
    """
    ESP32'ye röle komutu gönderir.

    ESP32 kodundaki beklenen JSON formatı:
        {"cihaz": "pompa", "durum": "ON"}
    Geçerli cihazlar : "pompa", "fan", "isitici"
    Geçerli durumlar : "ON", "OFF"

    Args:
        cihaz: Kontrol edilecek cihaz adı
        durum: "ON" veya "OFF"
    """
    if mqtt_istemci is None or not mqtt_istemci.is_connected():
        log.warning("[MQTT] Bağlantı yok, komut gönderilemedi: %s -> %s", cihaz, durum)
        return

    mesaj = json.dumps({"cihaz": cihaz, "durum": durum}, ensure_ascii=False)
    mqtt_istemci.publish(TOPIC_KOMUTLAR, mesaj, qos=1)
    log.info("[MQTT ↑] Komut gönderildi: %s", mesaj)


def ai_log_gonder(metin: str) -> None:
    """
    Kararın insan diline çevrilmiş açıklamasını Flutter'a gönderir.

    Args:
        metin: Çiftçiye gösterilecek Türkçe açıklama
    """
    if mqtt_istemci and mqtt_istemci.is_connected():
        mqtt_istemci.publish(TOPIC_AI_LOG, metin, qos=0)
        log.info("[AI LOG] %s", metin)

# ============================================================================
# BÖLÜM 8: MAKİNE ÖĞRENMESİ — MODEL EĞİTİMİ VE TAHMİN
# ============================================================================

# --- LSTM Yapılandırması ---
LOOKBACK_WINDOW = 12  # Model, karar vermek için son 12 saatlik trende bakar.
scaler = MinMaxScaler(feature_range=(0, 1))

def ml_model_egit_lstm():
    global ml_modeli, ml_hazir, scaler
    dosya = "C:\\Users\\hasan\\Downloads\\archive (2)\\IoTProcessed_Data.csv" if os.path.exists("C:\\Users\\hasan\\Downloads\\archive (2)\\IoTProcessed_Data.csv") else DATASET_DOSYA
    if not os.path.exists(dosya): return False
    try:
        df = pd.read_csv(dosya)
        if len(df) < ML_VERI_ESIGI: return False

        # Zaman Özelliği Ekleme
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        ozellikler = ["T_ortam", "H_ortam", "T_su", "Su_Mesafe_cm", "Isik_Analog", "elektrik_fiyati", "hour"]
        
        # Etiketleme (0=OFF, 1=ON)
        y_p = (df["pompa_karar"] == "ON").astype(int)
        y_f = (df["fan_karar"] == "ON").astype(int)
        y_i = (df["isitici_karar"] == "ON").astype(int)
        Y = np.column_stack([y_p, y_f, y_i])

        # Ölçeklendirme
        data_scaled = scaler.fit_transform(df[ozellikler])

        X_train, y_train = [], []
        for i in range(LOOKBACK_WINDOW, len(data_scaled)):
            X_train.append(data_scaled[i-LOOKBACK_WINDOW:i])
            y_train.append(Y[i])

        X_train, y_train = np.array(X_train), np.array(y_train)

        # LSTM Mimarisi
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(3, activation='sigmoid') # 3 Çıkış: Pompa, Fan, Isıtıcı
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        ml_modeli = model
        ml_hazir = True
        log.info("[LSTM] Derin Öğrenme Modeli Başarıyla Eğitildi.")
        return True
    except Exception as e:
        log.error(f"Eğitim hatası: {e}")
        return False

def ml_tahmin_yap_lstm(sensor, fiyat):
    """LSTM tahmini için anlık veriyi işler."""
    global predict_buffer
    saat = datetime.now().hour
    yeni_satir = [sensor["T_ortam"], sensor["H_ortam"], sensor["T_su"], sensor["Su_Mesafe_cm"], sensor["Isik_Analog"], fiyat, saat]
    
    predict_buffer.append(yeni_satir)
    if len(predict_buffer) < LOOKBACK_WINDOW: 
        log.info(f"[LSTM] Veri birikiyor: {len(predict_buffer)}/{LOOKBACK_WINDOW}")
        return None 
    
    recent_data = np.array(predict_buffer[-LOOKBACK_WINDOW:])
    recent_scaled = scaler.transform(recent_data)
    X_input = np.reshape(recent_scaled, (1, LOOKBACK_WINDOW, 7))
    
    pred = ml_modeli.predict(X_input, verbose=0)[0]
    return {
        "pompa": "ON" if pred[0] > 0.5 else "OFF",
        "fan": "ON" if pred[1] > 0.5 else "OFF",
        "isitici": "ON" if pred[2] > 0.5 else "OFF"
    }

# ============================================================================
# BÖLÜM 9: KURAL TABANLI KARAR MOTORU (Faz 3 — Cold-Start)
# ============================================================================

# ============================================================================
# BÖLÜM 9: KURAL TABANLI KARAR MOTORU (Faz 3 — Cold-Start & Yedek)
# ============================================================================

def kural_bazli_karar_al(sensor: dict, fiyat: float) -> tuple[dict, list[str]]:
    """
    EPİAŞ borsa fiyatı ve sensör verilerine dayalı deterministik karar motoru.
    ML modeli henüz 50 veriye ulaşmadan önce (Cold-Start) sistemi yönetir.

    Karar Stratejisi:
      1. Güvenlik (Pompa): Su biterse maliyete bakmaksızın durdur.
      2. Optimizasyon (Fan/Isıtıcı): Eşik aşılsa bile elektrik pahalıysa 
         bitkiyi güvenli sınırda tutup cihazı çalıştırma (Tasarruf Modu).
    """
    kararlar    = {}
    aciklamalar = []

    # Elektrik pahalı mı? (EPİAŞ fiyatı > Belirlenen Eşik)
    elektrik_pahali = fiyat > ESIK_ELEKTRIK_PAHALI

    # --- 1. POMPA KONTROLÜ (Güvenlik Öncelikli) ---
    # Su mesafesi kritik eşikten büyükse (Örn: 20cm), tankta su azalmıştır.
    if sensor["Su_Mesafe_cm"] > ESIK_SU_MESAFE_KRITIK:
        kararlar["pompa"] = "OFF"
        aciklamalar.append(
            f"KRİTİK: Su seviyesi çok düşük ({sensor['Su_Mesafe_cm']:.1f} cm). "
            "Pompa motoru korumak için kapatıldı."
        )
    else:
        kararlar["pompa"] = "ON"
        aciklamalar.append("Su seviyesi yeterli, pompa çalışmaya devam ediyor.")

    # --- 2. FAN KONTROLÜ (Sıcaklık ve Maliyet Odaklı) ---
    if sensor["T_ortam"] > ESIK_HAVA_SICAKLIK_MAX:
        if not elektrik_pahali:
            kararlar["fan"] = "ON"
            aciklamalar.append(
                f"Hava sıcak ({sensor['T_ortam']:.1f}°C) ve elektrik ucuz ({fiyat:.0f} TL). "
                "Soğutma başlatıldı."
            )
        else:
            kararlar["fan"] = "OFF"
            aciklamalar.append(
                f"Hava sıcak ({sensor['T_ortam']:.1f}°C) ancak elektrik PAHALI ({fiyat:.0f} TL). "
                "Maliyet optimizasyonu için fan kapalı tutuluyor."
            )
    else:
        kararlar["fan"] = "OFF"
        aciklamalar.append(f"Hava sıcaklığı ({sensor['T_ortam']:.1f}°C) normal seviyede.")

    # --- 3. ISITICI KONTROLÜ (Su Isısı ve Maliyet Odaklı) ---
    if sensor["T_su"] < ESIK_SU_SICAKLIK_MIN:
        if not elektrik_pahali:
            kararlar["isitici"] = "ON"
            aciklamalar.append(
                f"Su soğuk ({sensor['T_su']:.1f}°C) ve elektrik ucuz ({fiyat:.0f} TL). "
                "Isıtıcı (Sarı Ampul) aktif edildi."
            )
        else:
            kararlar["isitici"] = "OFF"
            aciklamalar.append(
                f"Su soğuk ({sensor['T_su']:.1f}°C) ancak elektrik PAHALI ({fiyat:.0f} TL). "
                "Isıtma erteleniyor."
            )
    else:
        kararlar["isitici"] = "OFF"
        aciklamalar.append(f"Su ısısı ({sensor['T_su']:.1f}°C) bitki gelişimi için ideal.")

    return kararlar, aciklamalar

    # --- POMPA KARARI ---
    if sensor["Su_Mesafe_cm"] > ESIK_SU_MESAFE_KRITIK:
        kararlar["pompa"] = "OFF"
        aciklamalar.append(
            f"Su seviyesi kritik ({sensor['Su_Mesafe_cm']:.1f} cm). "
            "Pompa kapatıldı, motor korunuyor."
        )
    else:
        kararlar["pompa"] = "ON"
        aciklamalar.append("Su seviyesi normal. Pompa çalışıyor.")

    # --- FAN KARARI ---
    if sensor["T_ortam"] > ESIK_HAVA_SICAKLIK_MAX:
        if not elektrik_pahali:
            kararlar["fan"] = "ON"
            aciklamalar.append(
                f"Hava sıcak ({sensor['T_ortam']:.1f}°C) ve elektrik "
                f"uygun ({fiyat:.0f} TL). Fan açıldı."
            )
        else:
            kararlar["fan"] = "OFF"
            aciklamalar.append(
                f"Hava sıcak ama elektrik pahalı ({fiyat:.0f} TL/MWh). "
                "Tasarruf modundayız, fan kapatıldı."
            )
    else:
        kararlar["fan"] = "OFF"
        aciklamalar.append(
            f"Ortam sıcaklığı ({sensor['T_ortam']:.1f}°C) normal. "
            "Fan gerekmiyor."
        )

    # --- ISITICI KARARI ---
    if sensor["T_su"] < ESIK_SU_SICAKLIK_MIN:
        if not elektrik_pahali:
            kararlar["isitici"] = "ON"
            aciklamalar.append(
                f"Su soğuk ({sensor['T_su']:.1f}°C) ve elektrik "
                f"uygun ({fiyat:.0f} TL). Isıtıcı açıldı."
            )
        else:
            kararlar["isitici"] = "OFF"
            aciklamalar.append(
                f"Su soğuk ama elektrik pahalı ({fiyat:.0f} TL/MWh). "
                "Isıtıcı beklemeye alındı."
            )
    else:
        kararlar["isitici"] = "OFF"
        aciklamalar.append(
            f"Su sıcaklığı ({sensor['T_su']:.1f}°C) yeterli. "
            "Isıtıcı gerekmiyor."
        )

    return kararlar, aciklamalar

# ============================================================================
# BÖLÜM 10: ANA KARAR DÖNGÜSÜ
# ============================================================================

def karar_dongusu_calistir(sensor: dict, fiyat: float) -> None:
    """
    Gelen sensör verisi ve elektrik fiyatıyla tam bir karar döngüsü yürütür:
      1. Kural tabanlı veya ML kararı al
      2. MQTT üzerinden ESP32'ye komutları gönder
      3. AI log mesajını Flutter'a ilet
      4. CSV'ye kaydet (ML eğitim seti için)
      5. SQLite'a kaydet (Grafana için)

    Args:
        sensor: Güncel filtrelenmiş sensör verisi
        fiyat : Güncel elektrik fiyatı
    """
    global son_kararlar

    # --- Karar Al ---
    if ml_hazir and ml_modeli is not None:
        # AI otonom modu — model tahmin ediyor
        kararlar    = ml_tahmin_yap_lstm(sensor, fiyat)
        aciklamalar = ["[AI Otonom Mod] Model kararlarını uyguluyor."]
        log.info("[KARAR] AI modeli kullanıldı.")
    else:
        # Cold-Start — kural tabanlı karar motoru
        kararlar, aciklamalar = kural_bazli_karar_al(sensor, fiyat)
        veri_sayisi = _csv_satir_sayisi()
        log.info("[KARAR] Kural motoru. Veri: %d/%d", veri_sayisi, ML_VERI_ESIGI)

    # --- Kararları ESP32'ye Gönder ---
    for cihaz, durum in kararlar.items():
        role_komutu_gonder(cihaz, durum)

    # --- AI Log Mesajını Oluştur ve Gönder ---
    log_mesaji = " | ".join(aciklamalar)
    ai_log_gonder(log_mesaji)

    # --- Küresel Karar Durumunu Güncelle (chatbot için) ---
    with veri_kilidi:
        son_kararlar.update(kararlar)

    # --- CSV ve SQLite'a Kaydet ---
    csv_kaydet(sensor, fiyat, kararlar)
    veritabanina_kaydet(sensor, fiyat, kararlar)

    # --- Her Döngüde ML Modelini Yeniden Eğitmeyi Dene ---
    if not ml_hazir:
        ml_model_egit_lstm()


def _csv_satir_sayisi() -> int:
    """CSV dosyasındaki veri satırı sayısını döner (başlık hariç)."""
    if not os.path.isfile(DATASET_DOSYA):
        return 0
    try:
        with open(DATASET_DOSYA, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)  # başlık satırını çıkar
    except OSError:
        return 0

# ============================================================================
# BÖLÜM 11: CHATBOT — DOĞAL DİL İŞLEME (NLP)
# ============================================================================

def chatbot_cevap_uret(soru: str) -> str:
    """
    Çiftçinin Flutter uygulamasından gönderdiği soruyu anahtar kelime
    tabanlı NLP ile analiz eder ve Türkçe cevap üretir.

    Desteklenen soru kategorileri:
      - Maliyet / fatura / elektrik
      - Genel sistem durumu
      - Su / pompa durumu
      - Sıcaklık / nem bilgisi
      - Yapay zeka / AI modu

    Args:
        soru: Çiftçinin gönderdiği ham metin

    Returns:
        str: Türkçe cevap metni
    """
    soru_kucuk = soru.lower().strip()

    with veri_kilidi:
        sensor = dict(son_sensor_verisi)
        fiyat  = son_elektrik_fiyati
        karar  = dict(son_kararlar)

    # --- Kategori 1: Maliyet / Elektrik ---
    if any(k in soru_kucuk for k in ["maliyet", "fatura", "elektrik", "enerji", "fiyat", "para"]):
        durum_metni = "uygun" if fiyat <= ESIK_ELEKTRIK_PAHALI else "pahalı"
        return (
            f"Şu anki elektrik fiyatı {fiyat:.0f} TL/MWh — bu fiyat {durum_metni}. "
            f"Fan {'açık' if karar['fan'] == 'ON' else 'kapalı'}, "
            f"ısıtıcı {'açık' if karar['isitici'] == 'ON' else 'kapalı'}. "
            "Sisteminiz fatura optimizasyonunu otomatik yapıyor."
        )

    # --- Kategori 2: Genel Durum ---
    if any(k in soru_kucuk for k in ["durum", "nasıl", "iyi mi", "sorun", "rapor", "özet"]):
        return (
            f"Sistem durumu:\n"
            f"  • Hava sıcaklığı : {sensor['T_ortam']:.1f}°C\n"
            f"  • Ortam nemi     : {sensor['H_ortam']:.1f}%\n"
            f"  • Su sıcaklığı   : {sensor['T_su']:.1f}°C\n"
            f"  • Su mesafesi    : {sensor['Su_Mesafe_cm']:.1f} cm\n"
            f"  • Pompa          : {karar['pompa']}\n"
            f"  • Fan            : {karar['fan']}\n"
            f"  • Isıtıcı        : {karar['isitici']}\n"
            f"  • Elektrik       : {fiyat:.0f} TL/MWh"
        )

    # --- Kategori 3: Su / Pompa ---
    if any(k in soru_kucuk for k in ["su", "pompa", "tank", "seviye", "sulama"]):
        kritik = sensor["Su_Mesafe_cm"] > ESIK_SU_MESAFE_KRITIK
        return (
            f"Su mesafesi {sensor['Su_Mesafe_cm']:.1f} cm "
            f"({'KRİTİK — tank azalıyor!' if kritik else 'normal seviyede'}).\n"
            f"Pompa şu an {'kapalı (motor koruması)' if karar['pompa'] == 'OFF' else 'çalışıyor'}."
        )

    # --- Kategori 4: Sıcaklık / Nem ---
    if any(k in soru_kucuk for k in ["sıcaklık", "sıcak", "soğuk", "nem", "hava", "ısı"]):
        return (
            f"Ortam sıcaklığı {sensor['T_ortam']:.1f}°C, "
            f"nem %{sensor['H_ortam']:.1f}.\n"
            f"Su sıcaklığı {sensor['T_su']:.1f}°C. "
            f"Isıtıcı {'açık' if karar['isitici'] == 'ON' else 'kapalı'}, "
            f"fan {'açık' if karar['fan'] == 'ON' else 'kapalı'}."
        )

    # --- Kategori 5: AI / Yapay Zeka Modu ---
    if any(k in soru_kucuk for k in ["ai", "yapay", "zeka", "öğren", "model", "otonom"]):
        veri_sayisi = _csv_satir_sayisi()
        if ml_hazir:
            return (
                f"Sistem tam otonom AI modunda çalışıyor. "
                f"Model {veri_sayisi} örnekle eğitildi ve kararları kendisi alıyor."
            )
        else:
            return (
                f"Sistem şu an kural tabanlı modda. "
                f"AI için {veri_sayisi}/{ML_VERI_ESIGI} veri toplandı. "
                f"Yakında otonom moda geçilecek!"
            )

    # --- Kategori 6: Bilinmeyen Soru ---
    return (
        "Merhaba! Şu konularda yardımcı olabilirim: "
        "elektrik maliyeti, sistem durumu, su seviyesi, "
        "sıcaklık/nem veya AI modu. Sorunuzu biraz farklı yazabilir misiniz?"
    )

# ============================================================================
# BÖLÜM 12: MQTT — CALLBACK FONKSİYONLARI
# ============================================================================

def mqtt_baglandinda(client, userdata, flags, reason_code, properties) -> None:
    """Broker bağlantısı kurulduğunda çalışır."""
    if reason_code == 0:
        log.info("[MQTT] Broker'a bağlandı: %s:%d", MQTT_SUNUCU, MQTT_PORT)
        # Gerekli kanallara abone ol
        client.subscribe(TOPIC_SENSORLER, qos=1)
        client.subscribe(TOPIC_CHAT_SORU, qos=1)
        log.info("[MQTT] Abonelikler: %s, %s", TOPIC_SENSORLER, TOPIC_CHAT_SORU)
    else:
        log.error("[MQTT] Bağlantı reddedildi. Kod: %s", reason_code)


def mqtt_kesildiginde(client, userdata, disconnect_flags, reason_code, properties) -> None:
    """Broker bağlantısı kesildiğinde çalışır."""
    log.warning("[MQTT] Bağlantı kesildi. Kod: %s. Yeniden bağlanılıyor...", reason_code)


def mqtt_mesaj_alindi(client, userdata, mesaj) -> None:
    """
    Abone olunan kanallardan mesaj geldiğinde çalışır.
    İşleme thread'de yapılır; callback hızlı döner.
    """
    konu = mesaj.topic
    try:
        ham_veri = mesaj.payload.decode("utf-8")
    except UnicodeDecodeError:
        log.error("[MQTT] Mesaj çözme hatası.")
        return

    if konu == TOPIC_SENSORLER:
        # Sensör verisini işlemeyi ayrı bir thread'de yap (callback'i bloke etme)
        threading.Thread(
            target=_sensor_mesajini_isle,
            args=(ham_veri,),
            daemon=True
        ).start()

    elif konu == TOPIC_CHAT_SORU:
        # Chatbot sorusunu ayrı thread'de yanıtla
        threading.Thread(
            target=_chatbot_sorusunu_isle,
            args=(ham_veri,),
            daemon=True
        ).start()


def _sensor_mesajini_isle(ham_veri: str) -> None:
    """
    ESP32'den gelen sensör JSON'ını ayrıştırır,
    elektrik fiyatını çeker ve karar döngüsünü başlatır.

    Beklenen JSON: {"T_ortam":25.3,"H_ortam":62.1,"T_su":21.4,
                    "Su_Mesafe_cm":8.5,"Isik_Analog":2100}
    """
    global son_sensor_verisi, son_elektrik_fiyati

    try:
        veri = json.loads(ham_veri)
    except json.JSONDecodeError as hata:
        log.error("[SENSÖR] JSON ayrıştırma hatası: %s | Ham veri: %s", hata, ham_veri)
        return

    # Beklenen anahtarları doğrula
    zorunlu_anahtarlar = {"T_ortam", "H_ortam", "T_su", "Su_Mesafe_cm", "Isik_Analog"}
    eksik = zorunlu_anahtarlar - veri.keys()
    if eksik:
        log.warning("[SENSÖR] Eksik anahtarlar: %s", eksik)
        return

    log.info("[SENSÖR] ←  T_ortam=%.1f°C  H=%.1f%%  T_su=%.1f°C  "
             "Mesafe=%.1fcm  Işık=%d",
             veri["T_ortam"], veri["H_ortam"], veri["T_su"],
             veri["Su_Mesafe_cm"], veri["Isik_Analog"])

    # Küresel sensör verisini güncelle (thread-safe)
    with veri_kilidi:
        son_sensor_verisi.update(veri)

    # Güncel elektrik fiyatını çek
    fiyat = epias_fiyat_cek()
    with veri_kilidi:
        son_elektrik_fiyati = fiyat

    # Karar döngüsünü çalıştır
    karar_dongusu_calistir(veri, fiyat)


def _chatbot_sorusunu_isle(soru: str) -> None:
    """
    Chatbot sorusunu alır, cevap üretir ve Flutter'a gönderir.

    Args:
        soru: Flutter uygulamasından gelen ham soru metni
    """
    log.info("[CHATBOT] Soru: %s", soru)
    cevap = chatbot_cevap_uret(soru)
    log.info("[CHATBOT] Cevap: %s", cevap)

    if mqtt_istemci and mqtt_istemci.is_connected():
        mqtt_istemci.publish(TOPIC_CHAT_CEVAP, cevap, qos=1)

# ============================================================================
# BÖLÜM 13: MQTT İSTEMCİSİ BAŞLATMA
# ============================================================================

def mqtt_istemci_olustur() -> mqtt.Client:
    """
    paho-mqtt v2 uyumlu MQTT istemcisini oluşturur ve yapılandırır.

    Returns:
        mqtt.Client: Yapılandırılmış MQTT istemci nesnesi
    """
    istemci = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=MQTT_CLIENT_ID,
        clean_session=True,
    )

    # Kimlik bilgileri varsa ayarla
    if MQTT_KULLANICI:
        istemci.username_pw_set(MQTT_KULLANICI, MQTT_SIFRE)

    # Callback'leri bağla
    istemci.on_connect    = mqtt_baglandinda
    istemci.on_disconnect = mqtt_kesildiginde
    istemci.on_message    = mqtt_mesaj_alindi

    # Otomatik yeniden bağlanma
    istemci.reconnect_delay_set(min_delay=2, max_delay=30)

    return istemci

# ============================================================================
# BÖLÜM 14: GİRİŞ NOKTASI (Main)
# ============================================================================

def main() -> None:
    """
    AGROTWIN Master Node ana giriş noktası.
    Altyapıyı başlatır ve MQTT döngüsünü kesintisiz çalıştırır.
    """
    global mqtt_istemci

    log.info("=" * 60)
    log.info("  AGROTWIN Master Node v2.0.0 başlatılıyor...")
    log.info("=" * 60)

    # --- Altyapı Başlatma ---
    veritabani_baslat()

    # Mevcut CSV verisiyle modeli önceden eğitmeyi dene (sıcak başlangıç)
    if ml_model_egit_lstm():
        log.info("[ML] Mevcut veriyle model önceden yüklendi.")
    else:
        log.info("[ML] Cold-Start modu. %d örnek toplanması bekleniyor.",
                 ML_VERI_ESIGI)

    # --- MQTT İstemcisi ---
    mqtt_istemci = mqtt_istemci_olustur()

    # Broker'a bağlan
    try:
        mqtt_istemci.connect(MQTT_SUNUCU, MQTT_PORT, keepalive=60)
    except (OSError, ConnectionRefusedError) as hata:
        log.error("[MQTT] İlk bağlantı başarısız: %s", hata)
        log.info("[MQTT] Arka planda yeniden deneme yapılacak...")

    # --- Sonsuz MQTT Döngüsü (Non-Blocking, thread-safe) ---
    log.info("[SİSTEM] MQTT döngüsü başladı. Sensörler bekleniyor...")
    try:
        mqtt_istemci.loop_forever()  # Bağlantı kopunca otomatik yeniden bağlanır
    except KeyboardInterrupt:
        log.info("[SİSTEM] Kullanıcı tarafından durduruldu (Ctrl+C).")
    finally:
        mqtt_istemci.disconnect()
        log.info("[SİSTEM] AGROTWIN Master Node kapatıldı.")


if __name__ == "__main__":
    main()

# ============================================================================
#  AGROTWIN Master Node — Son
#  "Topraksız tarımda, yapay zekâ hiç uyumaz." ™
# ============================================================================
