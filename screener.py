"""
Taiwan Stock Screener (Root copy for easy upload to GitHub)
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import normalize_symbol, fetch_stock_history, fetch_stock_fundamentals
from indicators import compute_all_indicators

EXPANDED_UNIVERSE = [
    {"code": "2330", "name": "台積電", "sector": "晶圓代工"},
    {"code": "2317", "name": "鴻海", "sector": "AI伺服器/組裝"},
    {"code": "2454", "name": "聯發科", "sector": "IC設計"},
    {"code": "2382", "name": "廣達", "sector": "AI伺服器"},
    {"code": "2308", "name": "台達電", "sector": "電源/散熱"},
    {"code": "3231", "name": "緯創", "sector": "AI伺服器"},
    {"code": "2603", "name": "長榮", "sector": "貨櫃航運"},
    {"code": "2609", "name": "陽明", "sector": "貨櫃航運"},
    {"code": "2881", "name": "富邦金", "sector": "金控"},
    {"code": "2882", "name": "國泰金", "sector": "金控"},
    {"code": "2891", "name": "中信金", "sector": "金控"},
    {"code": "2412", "name": "中華電", "sector": "電信"},
    {"code": "3008", "name": "大立光", "sector": "光學鏡頭"},
    {"code": "3443", "name": "創意", "sector": "ASIC/IP"},
    {"code": "3661", "name": "世芯-KY", "sector": "ASIC/IP"},
    {"code": "2376", "name": "技嘉", "sector": "AI伺服器/主機板"},
    {"code": "6669", "name": "緯穎", "sector": "雲端伺服器"},
    {"code": "3034", "name": "聯詠", "sector": "驅動IC"},
    {"code": "2357", "name": "華碩", "sector": "PC/伺服器"},
    {"code": "3711", "name": "日月光投控", "sector": "封測/CoWoS"},
    {"code": "2002", "name": "中鋼", "sector": "鋼鐵"},
    {"code": "1301", "name": "台塑", "sector": "塑化"},
    {"code": "1303", "name": "南亞", "sector": "塑化"},
    {"code": "0050", "name": "元大台灣50", "sector": "市值型ETF"},
    {"code": "0056", "name": "元大高股息", "sector": "高股息ETF"},
    {"code": "00878", "name": "國泰永續高股息", "sector": "高股息ETF"},
    {"code": "00919", "name": "群益精選高息", "sector": "高股息ETF"},
    {"code": "00929", "name": "復華科技優息", "sector": "科技高息ETF"},
    {"code": "8069", "name": "元太", "sector": "電子紙/上櫃"},
    {"code": "3293", "name": "鈊象", "sector": "遊戲軟體/上櫃"}
]

def screen_stocks(strategy: str = "all_signals") -> pd.DataFrame:
    results = []
    for item in EXPANDED_UNIVERSE:
        code = item["code"]
        name = item["name"]
        sector = item["sector"]
        norm_sym, _ = normalize_symbol(code)

        try:
            df = fetch_stock_history(norm_sym, period="6mo", interval="1d")
            if df.empty or len(df) < 25:
                continue

            df_ind = compute_all_indicators(df)
            latest = df_ind.iloc[-1]
            prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest

            close = float(latest["Close"])
            prev_close = float(prev["Close"])
            change = close - prev_close
            pct_change = (change / prev_close * 100.0) if prev_close > 0 else 0.0

            volume = int(latest["Volume"])
            vol_ma5 = float(latest.get("Vol_MA5", volume))
            vol_ratio = (volume / vol_ma5) if vol_ma5 > 0 else 1.0

            ma5 = float(latest.get("MA5", 0))
            ma20 = float(latest.get("MA20", 0))
            ma60 = float(latest.get("MA60", 0))
            k_val = float(latest.get("K", 50))
            d_val = float(latest.get("D", 50))
            k_prev = float(prev.get("K", 50))
            d_prev = float(prev.get("D", 50))
            rsi6 = float(latest.get("RSI_6", 50))

            match = False
            tags = []

            is_ma_bull = (close > ma5 > ma20 > ma60)
            if is_ma_bull:
                tags.append("均線多頭")

            is_breakout = (close > ma20 and prev_close <= ma20 and vol_ratio >= 1.3 and pct_change > 1.0)
            if is_breakout:
                tags.append("帶量突破月線")

            is_kd_cross = (k_val > d_val and k_prev <= d_prev and k_val < 70)
            if is_kd_cross:
                tags.append("KD黃金交叉")

            is_vol_surge = (vol_ratio >= 2.0 and pct_change >= 2.5)
            if is_vol_surge:
                tags.append("爆量長紅")

            is_rsi_rebound = (rsi6 < 35 or (prev.get("RSI_6", 50) < 30 and rsi6 >= 30))
            if is_rsi_rebound:
                tags.append("RSI低檔反彈")

            div_val = None
            if strategy in ["high_dividend", "all_signals"]:
                fund = fetch_stock_fundamentals(norm_sym)
                div_val = fund.get("dividend_yield")
                if div_val and div_val >= 4.5:
                    tags.append(f"高殖利率({div_val:.1f}%)")

            if strategy == "ma_bull" and is_ma_bull:
                match = True
            elif strategy == "vol_breakout" and is_breakout:
                match = True
            elif strategy == "kd_golden" and is_kd_cross:
                match = True
            elif strategy == "rsi_oversold" and is_rsi_rebound:
                match = True
            elif strategy == "vol_surge" and is_vol_surge:
                match = True
            elif strategy == "high_dividend" and (div_val and div_val >= 4.5):
                match = True
            elif strategy == "all_signals" and len(tags) > 0:
                match = True

            if match:
                results.append({
                    "股票代碼": code,
                    "股票名稱": name,
                    "所屬產業": sector,
                    "最新收盤價": f"NT${close:,.2f}",
                    "今日漲跌幅 (%)": f"{pct_change:+.2f}%",
                    "今日成交量": f"{volume:,}",
                    "今日量比 (倍)": f"{vol_ratio:.2f}x",
                    "KD(9,3)": f"K:{k_val:.1f} D:{d_val:.1f}",
                    "RSI(6)": f"{rsi6:.1f}",
                    "觸發訊號標籤": " | ".join(tags) if tags else "多頭符合",
                    "_raw_pct": pct_change,
                    "_tag_count": len(tags)
                })
        except Exception:
            continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.sort_values(by=["_tag_count", "_raw_pct"], ascending=[False, False], inplace=True)
        res_df.drop(columns=["_raw_pct", "_tag_count"], inplace=True)
    return res_df
