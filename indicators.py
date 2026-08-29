"""
Technical Indicators Calculation Module
Implements moving averages, Bollinger Bands, KD, RSI, MACD, BIAS, ATR, and Volume indicators.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """計算常用均線 (SMA 5, 10, 20, 60, 120, 240) 與 EMA (12, 26)"""
    res = df.copy()
    for period in [5, 10, 20, 60, 120, 240]:
        res[f"MA{period}"] = res["Close"].rolling(window=period).mean()

    res["EMA12"] = res["Close"].ewm(span=12, adjust=False).mean()
    res["EMA26"] = res["Close"].ewm(span=26, adjust=False).mean()
    return res


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """計算布林通道 (上軌、中軌、下軌、通道寬度、%B)"""
    res = df.copy()
    middle = res["Close"].rolling(window=window).mean()
    std = res["Close"].rolling(window=window).std()

    res["BB_Middle"] = middle
    res["BB_Upper"] = middle + (num_std * std)
    res["BB_Lower"] = middle - (num_std * std)
    res["BB_Width"] = ((res["BB_Upper"] - res["BB_Lower"]) / middle) * 100
    res["BB_PctB"] = (res["Close"] - res["BB_Lower"]) / (res["BB_Upper"] - res["BB_Lower"])
    return res


def add_kd(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    """
    計算台股常用 KD 指標 (標準 9, 3, 3 週期平滑)
    RSV = (Close - Low_N) / (High_N - Low_N) * 100
    K = 2/3 * K_prev + 1/3 * RSV
    D = 2/3 * D_prev + 1/3 * K
    """
    res = df.copy()
    low_n = res["Low"].rolling(window=n).min()
    high_n = res["High"].rolling(window=n).max()

    # 計算 RSV
    rsv = (res["Close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    # 遞推計算 K 與 D
    k_vals = []
    d_vals = []
    k_cur = 50.0
    d_cur = 50.0

    for val in rsv:
        if np.isnan(val):
            k_vals.append(np.nan)
            d_vals.append(np.nan)
        else:
            k_cur = (2.0 / 3.0) * k_cur + (1.0 / 3.0) * val
            d_cur = (2.0 / 3.0) * d_cur + (1.0 / 3.0) * k_cur
            k_vals.append(k_cur)
            d_vals.append(d_cur)

    res["K"] = k_vals
    res["D"] = d_vals
    res["RSV"] = rsv
    return res


def add_rsi(df: pd.DataFrame, periods: list = [6, 12, 14]) -> pd.DataFrame:
    """計算相對強弱指標 RSI (常用 6, 12, 14 天)"""
    res = df.copy()
    delta = res["Close"].diff()

    for p in periods:
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        res[f"RSI_{p}"] = rsi.fillna(50)

    return res


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    計算指數平滑異同移動平均線 (MACD)
    DIF = EMA(12) - EMA(26)
    MACD (Signal) = EMA(DIF, 9)
    OSC (Histogram) = (DIF - MACD) * 2
    """
    res = df.copy()
    ema_fast = res["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = res["Close"].ewm(span=slow, adjust=False).mean()

    res["MACD_DIF"] = ema_fast - ema_slow
    res["MACD_Signal"] = res["MACD_DIF"].ewm(span=signal, adjust=False).mean()
    res["MACD_Hist"] = (res["MACD_DIF"] - res["MACD_Signal"]) * 2.0
    return res


def add_bias(df: pd.DataFrame, periods: list = [5, 20, 60]) -> pd.DataFrame:
    """計算乖離率 (BIAS % = (Close - MA) / MA * 100)"""
    res = df.copy()
    for p in periods:
        ma = res["Close"].rolling(window=p).mean()
        res[f"BIAS_{p}"] = ((res["Close"] - ma) / ma) * 100
    return res


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """計算真實波幅均值 (ATR)"""
    res = df.copy()
    high = res["High"]
    low = res["Low"]
    close = res["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["TR"] = tr
    res["ATR"] = tr.rolling(window=period).mean()
    return res


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算成交量均線 (Vol MA5, Vol MA20) 與 OBV"""
    res = df.copy()
    res["Vol_MA5"] = res["Volume"].rolling(window=5).mean()
    res["Vol_MA20"] = res["Volume"].rolling(window=20).mean()

    # OBV
    direction = np.sign(res["Close"].diff()).fillna(0)
    res["OBV"] = (direction * res["Volume"]).cumsum()
    return res


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次計算所有核心技術指標"""
    if df.empty:
        return df

    res = df.copy()
    res = add_moving_averages(res)
    res = add_bollinger_bands(res)
    res = add_kd(res)
    res = add_rsi(res)
    res = add_macd(res)
    res = add_bias(res)
    res = add_atr(res)
    res = add_volume_indicators(res)
    return res
