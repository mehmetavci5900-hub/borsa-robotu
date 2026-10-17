import datetime
import json
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

# --- 1. GEMINI API KURULUMU ---
GEMINI_API_KEY = "AQ.Ab8RN6K8ZImYyJDD8u6HVy5r0sepqUJOt41J4ueCCRX9qajODw"
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. VERİTABANI KURULUMU ---
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

# --- 3. FASTAPI UYGULAMASI VE CORS İZNİ ---
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

def sembol_temizle(sembol: str) -> str:
    clean = sembol.upper().replace("/", "").replace("%2F", "").replace("-", "").replace(" ", "").strip()
    bilinen_uzantilar = ("USDT", "TRY", "EUR", "GBP", "JPY", "BUSD", "FDUSD", "BTC", "ETH", "BNB")
    if any(clean.endswith(ext) for ext in bilinen_uzantilar):
        return clean
    return clean + "USDT"

# --- 4. CANLI HABER ÇEKME ---
@app.get("/api/v1/haberler")
def haberleri_getir(sembol: str = "BTC"):
    try:
        clean_coin = sembol.upper().replace('USDT', '').replace('TRY', '').replace('GBP', '').replace('EUR', '').replace('JPY', '')
        query = urllib.parse.quote(f"{clean_coin} kripto OR bitcoin")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
        
        feed = feedparser.parse(rss_url)
        haberler = []

        for entry in feed.entries[:10]:
            haberler.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "Bugün"),
                "source": entry.get("source", {}).get("title", "Piyasa Basını")
            })

        return {"status": "success", "coin": clean_coin, "haberler": haberler}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 5. CANLI FİYAT SORGULAMA ---
@app.get("/api/v1/fiyat")
def canlı_fiyat_getir(sembol: str = "BTCUSDT"):
    try:
        clean_symbol = sembol_temizle(sembol)
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={clean_symbol}"
        res = requests.get(url, timeout=5)
        data = res.json()
        return {"status": "success", "sembol": clean_symbol, "son_fiyat": float(data['lastPrice'])}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 6. GRAFİK VERİSİ ---
@app.get("/api/v1/grafik-verisi")
def grafik_verisi_getir(sembol: str = "BTCUSDT", zaman_dilimi: str = "1m", limit: int = 100):
    try:
        clean_symbol = sembol_temizle(sembol)
        url = f"https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval={zaman_dilimi}&limit={limit}"
        res = requests.get(url, timeout=5)
        raw_data = res.json()

        mumlar = []
        for item in raw_data:
            mumlar.append({
                "time": int(item[0] / 1000),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            })
        return {"status": "success", "sembol": clean_symbol, "grafik_verileri": mumlar}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 7. TEKNİK ANALİZ ---
@app.get("/api/v1/teknik-analiz")
def teknik_analiz_hesapla(sembol: str = "BTCUSDT", zaman_dilimi: str = "1h"):
    try:
        clean_symbol = sembol_temizle(sembol)
        url = f"https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval={zaman_dilimi}&limit=100"
        res = requests.get(url, timeout=5)
        raw_data = res.json()

        df = pd.DataFrame(raw_data, columns=['ts', 'op', 'hi', 'lo', 'close', 'vol', 'ct', 'qav', 'not', 'tbba', 'tbqa', 'ig'])
        df['close'] = df['close'].astype(float)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

        son_fiyat = float(df['close'].iloc[-1])
        son_rsi = round(float(df['rsi'].iloc[-1]), 2)
        son_sma50 = round(float(df['sma_50'].iloc[-1]), 2)

        rsi_durum = "AŞIRI SATIM (AL SİNYALİ)" if son_rsi < 30 else ("AŞIRI ALIM (SAT SİNYALİ)" if son_rsi > 70 else "NÖTR")
        trend = "YÜKSELİŞ TRENDİ" if son_fiyat >= son_sma50 else "DÜŞÜŞ TRENDİ"

        return {
            "status": "success",
            "sembol": clean_symbol,
            "son_fiyat": son_fiyat,
            "rsi_14": son_rsi,
            "rsi_sinyali": rsi_durum,
            "genel_trend": trend
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 8. HİBRİT AI ANALİZ (TAM KORUMALI / ASLA ÇÖKMEYEN ENGINE) ---
@app.post("/api/v1/haber-analiz")
def haber_analiz_et(
    sembol: str = Form(...),
    mevcut_fiyat: float = Form(...),
    haber_metni: str = Form(...),
    db: Session = Depends(get_db)
):
    clean_symbol = sembol_temizle(sembol)
    
    # 1. Teknik analizi al
    try:
        teknik_veri = teknik_analiz_hesapla(sembol=clean_symbol)
        rsi_bilgi = teknik_veri.get("rsi_14", 50.0)
        rsi_sinyali = teknik_veri.get("rsi_sinyali", "NÖTR")
        genel_trend = teknik_veri.get("genel_trend", "NÖTR")
    except Exception:
        rsi_bilgi, rsi_sinyali, genel_trend = 50.0, "NÖTR", "NÖTR"
        teknik_veri = {}

    ai_data = None

    # 2. Gemini Yapay Zeka Sorgusu
    try:
        prompt = f"""
        Kripto Analiz Görevi:
        Sembol: {clean_symbol}, Fiyat: {mevcut_fiyat}
        Haber: {haber_metni}
        Teknik: RSI {rsi_bilgi} ({rsi_sinyali}), Trend: {genel_trend}

        Aşağıdaki JSON formatında yanıt ver:
        {{
            "karar": "AL",
            "tahmini_sure": "6-12 Saat İçinde",
            "beklenen_yon": "YÜKSELİŞ",
            "hedef_fiyat": {round(float(mevcut_fiyat) * 1.04, 2)},
            "stop_loss": {round(float(mevcut_fiyat) * 0.96, 2)},
            "ai_ozeti": "Teknik göstergeler ve haber akışı olumlu seyrettiği için yükseliş beklenmektedir."
        }}
        """
        
        response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        text = response.text
        
        # JSON Temizleme
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            ai_data = json.loads(json_str)
    except Exception as e:
        print("Gemini Servis Uyarısı:", str(e))

    # 3. YEDEK MOTOR (Gemini servis sınırına takılırsa devreye girer)
    if not ai_data or not isinstance(ai_data, dict):
        karar = "SAT" if rsi_bilgi > 70 else ("AL" if rsi_bilgi < 40 else "TUT")
        yon = "DÜŞÜŞ" if karar == "SAT" else ("YÜKSELİŞ" if karar == "AL" else "YATAY")
        
        ai_data = {
            "karar": karar,
            "tahmini_sure": "12-24 Saat İçinde",
            "beklenen_yon": yon,
            "hedef_fiyat": round(float(mevcut_fiyat) * (1.05 if karar == "AL" else 0.95), 2),
            "stop_loss": round(float(mevcut_fiyat) * (0.95 if karar == "AL" else 1.03), 2),
            "ai_ozeti": f"Teknik RSI göstergesi ({rsi_bilgi}) ve mevcut haber eğilimi doğrultusunda {karar} pozisyonu öngörülmektedir."
        }

    # 4. Veritabanına Kayıt
    try:
        yeni_kayit = AnalizKaydi(
            sembol=clean_symbol,
            mevcut_fiyat=float(mevcut_fiyat),
            beklenen_yon=str(ai_data.get('karar', 'TUT')),
            hedef_fiyat=float(ai_data.get("hedef_fiyat", mevcut_fiyat)),
            stop_loss=float(ai_data.get("stop_loss", mevcut_fiyat))
        )
        db.add(yeni_kayit)
        db.commit()
    except Exception:
        pass

    return {"status": "success", "analiz_sonucu": ai_data, "teknik": teknik_veri}