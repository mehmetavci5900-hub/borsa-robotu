import datetime
import json
import os
import re
import logging
from typing import Optional, Dict, Any, List

import requests
import urllib.parse
import feedparser
import pandas as pd
import ta

from fastapi import FastAPI, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from google import genai


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("borsa_robotu_api")


# =========================
# 1) GEMINI API KURULUMU (GÜVENLİK: ENV)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY ortam değişkeni set edilmedi. Lütfen export/ENV ile ekleyin.")

ai_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# 2) VERİTABANI KURULUMU
# =========================
DATABASE_URL = "sqlite:///./borsa_robotu.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalizKaydi(Base):
    __tablename__ = "analizler"

    id = Column(Integer, primary_key=True, index=True)
    sembol = Column(String)
    mevcut_fiyat = Column(Float)
    beklenen_yon = Column(String)
    hedef_fiyat = Column(Float)
    stop_loss = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


# =========================
# 3) FASTAPI + CORS
# =========================
app = FastAPI(title="Borsa Robotu API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# 4) SYMBOL CLEANER
# =========================
def sembol_temizle(sembol: str) -> str:
    """
    Binance symbol formatı genelde: BTCUSDT, ETHUSDT vb.
    Kullanıcı: BTC/USDT, BTC-USDT, BTC, BTCUSDT ... gönderebilir.
    """
    if not sembol:
        raise ValueError("sembol boş olamaz.")

    clean = (
        sembol.upper()
        .replace("/", "")
        .replace("%2F", "")
        .replace("-", "")
        .replace(" ", "")
        .strip()
    )

    # Eğer zaten bir quote ile bitiyorsa olduğu gibi bırak
    bilinen_uzantilar = (
        "USDT", "TRY", "EUR", "GBP", "JPY", "BUSD",
        "FDUSD", "BTC", "ETH", "BNB"
    )
    if any(clean.endswith(ext) for ext in bilinen_uzantilar):
        return clean

    # Diğer durumda varsayılan USDT quote
    return clean + "USDT"


# =========================
# 5) BINANCE HELPER
# =========================
def binance_get_json(url: str, params: Optional[dict] = None, timeout: int = 8) -> dict:
    try:
        res = requests.get(url, params=params, timeout=timeout)
        # Binance bazen 4xx döndürür; raise et
        res.raise_for_status()
        data = res.json()
        return data
    except requests.RequestException as e:
        logger.error(f"Binance HTTP hatası: {e} | url={url} params={params}")
        raise
    except ValueError as e:
        logger.error(f"Binance JSON parse hatası: {e} | url={url} params={params}")
        raise


# =========================
# 6) CANLI HABER ÇEKME
# =========================
@app.get("/api/v1/haberler")
def haberleri_getir(sembol: str = "BTC"):
    try:
        clean_coin = sembol.upper()
        for cut in ["USDT", "TRY", "GBP", "EUR", "JPY"]:
            clean_coin = clean_coin.replace(cut, "")

        query = urllib.parse.quote(f"{clean_coin} kripto OR bitcoin")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"

        feed = feedparser.parse(rss_url)
        if getattr(feed, "status", None) not in (None, 200):
            # feedparser bazen status vermez; ama olağan dışıysa logla
            logger.warning(f"RSS beklenmedik status: {getattr(feed, 'status', None)}")

        haberler: List[Dict[str, Any]] = []
        for entry in feed.entries[:10]:
            haberler.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "Bugün"),
                "source": entry.get("source", {}).get("title", "Piyasa Basını"),
            })

        return {"status": "success", "coin": clean_coin, "haberler": haberler}
    except Exception as e:
        logger.exception("haberleri_getir hata aldı")
        return {"status": "error", "message": str(e)}


# =========================
# 7) CANLI FİYAT SORGULAMA
# =========================
@app.get("/api/v1/fiyat")
def canlı_fiyat_getir(sembol: str = "BTCUSDT"):
    try:
        clean_symbol = sembol_temizle(sembol)
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = binance_get_json(url, params={"symbol": clean_symbol})

        last_price = data.get("lastPrice")
        if last_price is None:
            raise ValueError(f"Binance yanıtında lastPrice yok. Yanıt: {data}")

        return {"status": "success", "sembol": clean_symbol, "son_fiyat": float(last_price)}
    except Exception as e:
        logger.exception("canlı_fiyat_getir hata aldı")
        return {"status": "error", "message": str(e)}


# =========================
# 8) GRAFİK VERİSİ
# =========================
@app.get("/api/v1/grafik-verisi")
def grafik_verisi_getir(sembol: str = "BTCUSDT", zaman_dilimi: str = "1m", limit: int = 100):
    """
    Binance klines:
    /api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100
    """
    try:
        clean_symbol = sembol_temizle(sembol)

        # limit kontrol
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="limit 1 ile 1000 arasında olmalı.")

        url = "https://api.binance.com/api/v3/klines"
        raw_data = binance_get_json(url, params={
            "symbol": clean_symbol,
            "interval": zaman_dilimi,
            "limit": limit
        })

        if not isinstance(raw_data, list):
            raise ValueError(f"Beklenen liste (klines) gelmedi. Gelen: {type(raw_data)} - {raw_data}")

        mumlar: List[Dict[str, Any]] = []
        for item in raw_data:
            # item formatı: [openTime, open, high, low, close, volume, ...]
            if len(item) < 6:
                continue
            open_time_ms = item[0]

            mumlar.append({
                "time": int(open_time_ms / 1000),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            })

        if not mumlar:
            raise ValueError("Binance'dan veri gelmedi (mumlar boş). Sembol/interval kontrol et.")

        return {"status": "success", "sembol": clean_symbol, "grafik_verileri": mumlar}

    except Exception as e:
        logger.exception("grafik_verisi_getir hata aldı")
        return {"status": "error", "message": str(e)}


# =========================
# 9) TEKNİK ANALİZ (SERVICE) + ENDPOINT
# =========================
def teknik_analiz_hesapla_service(sembol: str = "BTCUSDT", zaman_dilimi: str = "1h", limit: int = 100) -> Dict[str, Any]:
    clean_symbol = sembol_temizle(sembol)

    url = "https://api.binance.com/api/v3/klines"
    raw_data = binance_get_json(url, params={
        "symbol": clean_symbol,
        "interval": zaman_dilimi,
        "limit": limit
    })

    if not isinstance(raw_data, list) or len(raw_data) < 60:
        raise ValueError("Yetersiz veri geldi. Limit/interval/simbol kontrol et.")

    df = pd.DataFrame(raw_data, columns=[
        'ts', 'op', 'hi', 'lo', 'close', 'vol', 'ct', 'qav', 'not', 'tbba', 'tbqa', 'ig'
    ])

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # RSI & SMA
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

    # NaN temizle (özellikle başlangıç dönemleri)
    df = df.dropna(subset=['rsi', 'sma_50']).reset_index(drop=True)
    if df.empty:
        raise ValueError("RSI/SMA hesaplanamadı (NaN oluştu).")

    son_fiyat = float(df['close'].iloc[-1])
    son_rsi = float(df['rsi'].iloc[-1])
    son_sma50 = float(df['sma_50'].iloc[-1])

    # Sinyaller
    rsi_durum = (
        "AŞIRI SATIM (AL SİNYALİ)" if son_rsi < 30
        else ("AŞIRI ALIM (SAT SİNYALİ)" if son_rsi > 70
              else "NÖTR")
    )
    trend = "YÜKSELİŞ TRENDİ" if son_fiyat >= son_sma50 else "DÜŞÜŞ TRENDİ"

    return {
        "sembol": clean_symbol,
        "son_fiyat": son_fiyat,
        "rsi_14": round(son_rsi, 2),
        "rsi_sinyali": rsi_durum,
        "genel_trend": trend
    }


@app.get("/api/v1/teknik-analiz")
def teknik_analiz_hesapla(sembol: str = "BTCUSDT", zaman_dilimi: str = "1h"):
    try:
        data = teknik_analiz_hesapla_service(sembol=sembol, zaman_dilimi=zaman_dilimi, limit=100)
        return {"status": "success", **data}
    except Exception as e:
        logger.exception("teknik_analiz_hesapla hata aldı")
        return {"status": "error", "message": str(e)}


# =========================
# 10) GEMINI JSON PARSER (SAĞLAM)
# =========================
def extract_first_json_object(text: str) -> Optional[dict]:
    """
    Model bazen açıklama ekleyebiliyor.
    En basit sağlam yöntem: ilk '{...}' blok yakalamak.
    """
    if not text:
        return None

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    json_candidate = match.group(0).strip()

    try:
        return json.loads(json_candidate)
    except Exception:
        return None


# =========================
# 11) HİBRİT AI ANALİZ (DÜZELTİLDİ)
# =========================
@app.post("/api/v1/haber-analiz")
def haber_analiz_et(
    sembol: str = Form(...),
    mevcut_fiyat: float = Form(...),
    haber_metni: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        clean_symbol = sembol_temizle(sembol)
        fiyat = float(mevcut_fiyat)

        # 1) Teknik veriyi service ile al
        try:
            teknik_veri = teknik_analiz_hesapla_service(sembol=clean_symbol, zaman_dilimi="1h", limit=100)
            rsi_bilgi = float(teknik_veri.get("rsi_14", 50.0))
            rsi_sinyali = teknik_veri.get("rsi_sinyali", "NÖTR")
            genel_trend = teknik_veri.get("genel_trend", "NÖTR")
        except Exception as e:
            logger.warning(f"Teknik analiz alınamadı: {e}")
            teknik_veri = {}
            rsi_bilgi, rsi_sinyali, genel_trend = 50.0, "NÖTR", "NÖTR"

        ai_data: Optional[Dict[str, Any]] = None

        # 2) Gemini Prompt
        prompt = f"""
Kripto Analiz Görevi:
Sembol: {clean_symbol}
Mevcut Fiyat: {fiyat}
Haber Metni: {haber_metni}
Teknik:
- RSI(14): {rsi_bilgi} ({rsi_sinyali})
- Trend: {genel_trend}

ÇIKTI ZORUNLU:
Sadece geçerli JSON döndür.
JSON alanları:
{{
  "karar": "AL" | "TUT" | "SAT",
  "tahmini_sure": string,
  "beklenen_yon": "YÜKSELİŞ" | "YATAY" | "DÜŞÜŞ",
  "hedef_fiyat": number,
  "stop_loss": number,
  "ai_ozeti": string
}}

Ek açıklama, markdown, kod bloğu (```), veya başka metin ekleme.
"""
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            text = getattr(response, "text", None) or str(response)
            ai_data = extract_first_json_object(text)

            if not ai_data or not isinstance(ai_data, dict):
                ai_data = None
            else:
                karar = ai_data.get("karar")
                yon = ai_data.get("beklenen_yon")

                if karar not in ("AL", "TUT", "SAT"):
                    ai_data = None
                else:
                    if karar == "AL" and yon not in ("YÜKSELİŞ",):
                        ai_data = None
                    if karar == "SAT" and yon not in ("DÜŞÜŞ",):
                        ai_data = None
                    if karar == "TUT" and yon not in ("YATAY",):
                        ai_data = None

        except Exception as e:
            logger.warning(f"Gemini servis hatası: {e}")
            ai_data = None

        # 3) YEDEK MOTOR
        if not ai_data:
            karar = "SAT" if rsi_bilgi > 70 else ("AL" if rsi_bilgi < 40 else "TUT")
            yon = "DÜŞÜŞ" if karar == "SAT" else ("YÜKSELİŞ" if karar == "AL" else "YATAY")

            ai_data = {
                "karar": karar,
                "tahmini_sure": "12-24 Saat İçinde",
                "beklenen_yon": yon,
                "hedef_fiyat": round(fiyat * (1.05 if karar == "AL" else 0.95), 2),
                "stop_loss": round(fiyat * (0.95 if karar == "AL" else 1.03), 2),
                "ai_ozeti": f"RSI(14) = {rsi_bilgi} ve teknik trend ({genel_trend}) doğrultusunda {karar} pozisyonu öngörülmektedir."
            }

        # 4) Veritabanına kayıt
        try:
            yeni_kayit = AnalizKaydi(
                sembol=clean_symbol,
                mevcut_fiyat=fiyat,
                beklenen_yon=str(ai_data.get("karar", "TUT")),
                hedef_fiyat=float(ai_data.get("hedef_fiyat", fiyat)),
                stop_loss=float(ai_data.get("stop_loss", fiyat))
            )
            db.add(yeni_kayit)
            db.
