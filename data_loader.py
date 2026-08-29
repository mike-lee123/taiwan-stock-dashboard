"""
Data Loader Module for Taiwan Stock & Global Market Analysis Dashboard
Provides functions to fetch and normalize Taiwan stocks, global macro indices,
ADR premium calculations, and fundamental financial metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from typing import Dict, List, Tuple, Optional, Any

# 熱門台股預設清單 (代碼, 名稱, 類別, 市場)
POPULAR_TW_STOCKS = [
    {"code": "2330", "name": "台積電", "sector": "半導體", "market": "TW"},
    {"code": "2317", "name": "鴻海", "sector": "電子代工/AI伺服器", "market": "TW"},
    {"code": "2454", "name": "聯發科", "sector": "IC設計", "market": "TW"},
    {"code": "2382", "name": "廣達", "sector": "AI伺服器/電腦", "market": "TW"},
    {"code": "2308", "name": "台達電", "sector": "電源/散熱/綠能", "market": "TW"},
    {"code": "3231", "name": "緯創", "sector": "AI伺服器/電腦", "market": "TW"},
    {"code": "2603", "name": "長榮", "sector": "航運", "market": "TW"},
    {"code": "2881", "name": "富邦金", "sector": "金融保險", "market": "TW"},
    {"code": "2882", "name": "國泰金", "sector": "金融保險", "market": "TW"},
    {"code": "2412", "name": "中華電", "sector": "通信網路/電信", "market": "TW"},
    {"code": "0050", "name": "元大台灣50", "sector": "市值型ETF", "market": "TW"},
    {"code": "0056", "name": "元大高股息", "sector": "高股息ETF", "market": "TW"},
    {"code": "00878", "name": "國泰永續高股息", "sector": "高股息ETF", "market": "TW"},
    {"code": "00919", "name": "群益台灣精選高息", "sector": "高股息ETF", "market": "TW"},
    {"code": "00929", "name": "復華台灣科技優息", "sector": "科技高息ETF", "market": "TW"},
    {"code": "006208", "name": "富邦台50", "sector": "市值型ETF", "market": "TW"},
    {"code": "3008", "name": "大立光", "sector": "光電/鏡頭", "market": "TW"},
    {"code": "3443", "name": "創意", "sector": "ASIC/IP", "market": "TW"},
    {"code": "3661", "name": "世芯-KY", "sector": "ASIC/IP", "market": "TW"},
    {"code": "6547", "name": "高端疫苗", "sector": "生技醫療", "market": "TWO"},
    {"code": "8069", "name": "元太", "sector": "電子紙/光電", "market": "TWO"},
    {"code": "3293", "name": "鈊象", "sector": "遊戲軟體", "market": "TWO"}
]

# 國際與連動總經指標代碼對照表
MACRO_BENCHMARKS = {
    "費城半導體 (SOX)": {"symbol": "^SOX", "desc": "全球半導體景氣風向球", "category": "美股指數"},
    "道瓊工業指數 (DJI)": {"symbol": "^DJI", "desc": "美國藍籌權值股代表", "category": "美股指數"},
    "那斯達克 (IXIC)": {"symbol": "^IXIC", "desc": "美國科技與成長股代表", "category": "美股指數"},
    "標普500 (S&P 500)": {"symbol": "^GSPC", "desc": "美國標竿綜合大盤指數", "category": "美股指數"},
    "美元兌台幣 (USD/TWD)": {"symbol": "USDTWD=X", "desc": "外資資金匯出入與匯率走勢", "category": "外匯匯率"},
    "韓國綜合指數 (KOSPI)": {"symbol": "^KS11", "desc": "亞洲半導體/科技競爭市場", "category": "亞洲股市"},
    "台積電 ADR (TSM)": {"symbol": "TSM", "desc": "台積電美股存託憑證", "category": "連動ADR"},
    "MSCI台灣ETF (EWT)": {"symbol": "EWT", "desc": "美股交易時段台股連動風向球", "category": "夜盤/海外ETF"},
    "台股加權指數 (TAIEX)": {"symbol": "^TWII", "desc": "台灣集中市場大盤指數", "category": "台股大盤"}
}


def normalize_symbol(user_input: str) -> Tuple[str, str]:
    """
    將使用者輸入的股票代號或名稱解析為正確的 yfinance Ticker 與顯示名稱。
    例如: '2330' -> ('2330.TW', '台積電')
          '台積電' -> ('2330.TW', '台積電')
          'TSM' -> ('TSM', '台積電 ADR')
          'AAPL' -> ('AAPL', 'AAPL')
    """
    cleaned = user_input.strip()
    if not cleaned:
        return "2330.TW", "台積電"

    # 先在熱門清單中搜尋
    for item in POPULAR_TW_STOCKS:
        if cleaned.upper() == item["code"] or cleaned == item["name"]:
            suffix = ".TWO" if item["market"] == "TWO" else ".TW"
            return f"{item['code']}{suffix}", f"{item['name']} ({item['code']})"

    # 檢查是否為純數字 (台股 4 或 5 或 6 位代碼)
    if cleaned.isdigit():
        return f"{cleaned}.TW", f"台股 {cleaned}"

    # 若已經帶有 .TW 或 .TWO
    if cleaned.upper().endswith(".TW") or cleaned.upper().endswith(".TWO"):
        return cleaned.upper(), cleaned.upper()

    # 檢查是否為總經指標
    for name, data in MACRO_BENCHMARKS.items():
        if cleaned.upper() == data["symbol"].upper() or cleaned in name:
            return data["symbol"], name

    # 其它美股或英文字符號
    return cleaned.upper(), cleaned.upper()


def fetch_stock_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[datetime.date] = None,
    end: Optional[datetime.date] = None
) -> pd.DataFrame:
    """
    抓取指定代碼的歷史 OHLCV 資料，並進行欄位標準化與清理。
    """
    try:
        t = yf.Ticker(ticker)
        if start and end:
            df = t.history(start=start, end=end, interval=interval)
        else:
            df = t.history(period=period, interval=interval)

        # 若 .TW 抓不到且是 4 碼數字，自動嘗試 .TWO (櫃買)
        if (df.empty or len(df.dropna()) == 0) and ticker.endswith(".TW"):
            alt_ticker = ticker.replace(".TW", ".TWO")
            t_alt = yf.Ticker(alt_ticker)
            if start and end:
                df = t_alt.history(start=start, end=end, interval=interval)
            else:
                df = t_alt.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # 處理 MultiIndex 欄位 (若 yfinance 回傳多層欄位)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # 確保必要欄位存在
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()

        # 去除時區資訊，利於 Plotly 與日期比對
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[required_cols].copy()
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_macro_data() -> Dict[str, Dict[str, Any]]:
    """
    抓取所有國際總經指數、匯率、海外連動與主要市場的最新報價與近期漲跌幅。
    """
    results = {}
    for name, info in MACRO_BENCHMARKS.items():
        sym = info["symbol"]
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="1mo", interval="1d")
            if not hist.empty:
                # 處理時區
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)

                clean_close = hist["Close"].dropna()
                if len(clean_close) >= 1:
                    latest_close = float(clean_close.iloc[-1])
                    prev_close = float(clean_close.iloc[-2]) if len(clean_close) > 1 else latest_close
                    change = latest_close - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0

                    results[name] = {
                        "symbol": sym,
                        "desc": info["desc"],
                        "category": info["category"],
                        "price": latest_close,
                        "change": change,
                        "pct_change": pct_change,
                        "history": clean_close,
                        "high_52w": float(hist["High"].dropna().max()) if not hist["High"].dropna().empty else latest_close,
                        "low_52w": float(hist["Low"].dropna().min()) if not hist["Low"].dropna().empty else latest_close,
                        "latest_date": clean_close.index[-1].strftime("%Y-%m-%d")
                    }
                    continue

            # Fallback
            results[name] = {
                "symbol": sym,
                "desc": info["desc"],
                "category": info["category"],
                "price": np.nan,
                "change": 0.0,
                "pct_change": 0.0,
                "history": pd.Series(dtype=float),
                "high_52w": np.nan,
                "low_52w": np.nan,
                "latest_date": "N/A"
            }
        except Exception as e:
            print(f"Error loading macro {sym}: {e}")
            results[name] = {
                "symbol": sym,
                "desc": info["desc"],
                "category": info["category"],
                "price": np.nan,
                "change": 0.0,
                "pct_change": 0.0,
                "history": pd.Series(dtype=float),
                "high_52w": np.nan,
                "low_52w": np.nan,
                "latest_date": "N/A"
            }
    return results


def calculate_adr_premium(period: str = "6mo") -> pd.DataFrame:
    """
    計算台積電 ADR (TSM) 相對於台積電現股 (2330.TW) 的溢價率 (Premium %)。
    換算公式:
      1 單位 TSM ADR = 5 股 2330.TW 普通股
      ADR 折合台幣每股價格 = (TSM 收盤價(USD) * 美元兌台幣匯率) / 5
      溢價率(%) = ((ADR每股折合台幣 - 2330現股收盤價) / 2330現股收盤價) * 100%
    """
    try:
        tsm_df = fetch_stock_history("TSM", period=period)
        tw_df = fetch_stock_history("2330.TW", period=period)
        fx_df = fetch_stock_history("USDTWD=X", period=period)

        if tsm_df.empty or tw_df.empty or fx_df.empty:
            return pd.DataFrame()

        # 整合三者歷史收盤價
        merged = pd.DataFrame({
            "TSM_USD": tsm_df["Close"].dropna(),
            "TW_2330": tw_df["Close"].dropna(),
            "USD_TWD": fx_df["Close"].dropna()
        }).dropna()

        # ADR 換算成台幣每股價值 (1 ADR = 5 普通股)
        merged["ADR_TWD_Equiv"] = (merged["TSM_USD"] * merged["USD_TWD"]) / 5.0
        # 計算溢價率 %
        merged["Premium_Pct"] = ((merged["ADR_TWD_Equiv"] - merged["TW_2330"]) / merged["TW_2330"]) * 100.0
        # 價差 (TWD)
        merged["Spread_TWD"] = merged["ADR_TWD_Equiv"] - merged["TW_2330"]

        return merged
    except Exception as e:
        print(f"Error calculating ADR premium: {e}")
        return pd.DataFrame()


def fetch_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    抓取個股基本面、估值指標與歷年配息歷史。
    """
    info_dict = {
        "name": ticker,
        "pe_ratio": None,
        "forward_pe": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "market_cap": None,
        "eps": None,
        "beta": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "profit_margin": None,
        "roe": None,
        "revenue_growth": None,
        "dividends": pd.Series(dtype=float),
        "financials_summary": {}
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        info_dict["name"] = info.get("longName") or info.get("shortName") or ticker
        info_dict["pe_ratio"] = info.get("trailingPE")
        info_dict["forward_pe"] = info.get("forwardPE")
        info_dict["pb_ratio"] = info.get("priceToBook")
        div_rate = info.get("dividendYield")
        info_dict["dividend_yield"] = (div_rate * 100) if div_rate is not None else None
        info_dict["market_cap"] = info.get("marketCap")
        info_dict["eps"] = info.get("trailingEps")
        info_dict["beta"] = info.get("beta")
        info_dict["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
        info_dict["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
        info_dict["profit_margin"] = (info.get("profitMargins") * 100) if info.get("profitMargins") is not None else None
        info_dict["roe"] = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") is not None else None
        info_dict["revenue_growth"] = (info.get("revenueGrowth") * 100) if info.get("revenueGrowth") is not None else None

        # 歷年股利
        divs = t.dividends
        if divs is not None and not divs.empty:
            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            info_dict["dividends"] = divs

    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")

    return info_dict


def calculate_pe_bands(df: pd.DataFrame, eps: Optional[float] = None) -> pd.DataFrame:
    """
    計算本益比河流圖 (P/E River Bands)
    若無外部 EPS，則依據移動收盤推算基礎倍數。
    """
    if df.empty:
        return df

    result_df = df.copy()
    if eps and eps > 0:
        base_eps = eps
    else:
        # 推估參考每股盈餘 (以歷史平均中位數換算 18 倍本益比為基準)
        base_eps = result_df["Close"].median() / 18.0

    # 常用本益比倍數區間：12x, 15x, 18x, 22x, 26x
    multipliers = [12, 15, 18, 22, 26]
    for m in multipliers:
        result_df[f"PE_{m}X"] = base_eps * m

    return result_df
