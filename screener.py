"""
Taiwan Stock Screener (Root copy for easy upload to GitHub)
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import normalize_symbol, fetch_stock_history, fetch_stock_fundamentals, POPULAR_TW_STOCKS, TAIWAN_STOCK_UNIVERSE_260
from indicators import compute_all_indicators

EXPANDED_UNIVERSE = POPULAR_TW_STOCKS

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
