"""
Backtest Engine Module
Provides vector-based and event-driven backtesting for technical trading strategies.
Computes Total Return, CAGR, Max Drawdown, Sharpe Ratio, Win Rate, and Trade Logs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def run_strategy_backtest(
    df: pd.DataFrame,
    strategy: str = "Dual_MA",
    initial_capital: float = 100000.0,
    **kwargs
) -> Tuple[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
    """
    執行策略回測
    回傳:
      - metrics: 績效指標摘要字典
      - equity_curve: 包含策略與基準權益走勢之 DataFrame
      - trades: 交易紀錄明細清單
    """
    if df.empty or len(df) < 30:
        return {}, pd.DataFrame(), []

    data = df.copy()
    signals = pd.Series(0, index=data.index)

    # 1. 產生進出場訊號 (1: 買進持有, 0: 空手)
    if strategy == "Dual_MA":
        fast_p = kwargs.get("fast_period", 5)
        slow_p = kwargs.get("slow_period", 20)
        data[f"MA_fast"] = data["Close"].rolling(fast_p).mean()
        data[f"MA_slow"] = data["Close"].rolling(slow_p).mean()
        signals = np.where(data["MA_fast"] > data["MA_slow"], 1, 0)

    elif strategy == "RSI":
        rsi_p = kwargs.get("rsi_period", 14)
        buy_threshold = kwargs.get("buy_threshold", 30)
        sell_threshold = kwargs.get("sell_threshold", 70)
        if f"RSI_{rsi_p}" not in data.columns:
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
            rs = gain / loss.replace(0, np.nan)
            data["RSI_val"] = 100 - (100 / (1 + rs))
        else:
            data["RSI_val"] = data[f"RSI_{rsi_p}"]

        # 狀態機產生持倉訊號
        pos = 0
        sig_list = []
        for val in data["RSI_val"]:
            if np.isnan(val):
                sig_list.append(0)
            elif val < buy_threshold:
                pos = 1
                sig_list.append(pos)
            elif val > sell_threshold:
                pos = 0
                sig_list.append(pos)
            else:
                sig_list.append(pos)
        signals = np.array(sig_list)

    elif strategy == "MACD":
        if "MACD_DIF" not in data.columns:
            ema12 = data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = data["Close"].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dem = dif.ewm(span=9, adjust=False).mean()
        else:
            dif = data["MACD_DIF"]
            dem = data["MACD_Signal"]

        signals = np.where(dif > dem, 1, 0)

    elif strategy == "Bollinger":
        if "BB_Lower" not in data.columns:
            mid = data["Close"].rolling(20).mean()
            std = data["Close"].rolling(20).std()
            lower = mid - 2.0 * std
            upper = mid + 2.0 * std
        else:
            lower = data["BB_Lower"]
            upper = data["BB_Upper"]

        pos = 0
        sig_list = []
        for c, l, u in zip(data["Close"], lower, upper):
            if np.isnan(l) or np.isnan(u):
                sig_list.append(0)
            elif c < l:
                pos = 1  # 破下軌買進
                sig_list.append(pos)
            elif c > u:
                pos = 0  # 觸上軌賣出
                sig_list.append(pos)
            else:
                sig_list.append(pos)
        signals = np.array(sig_list)

    data["Signal"] = signals
    data["Position"] = data["Signal"].shift(1).fillna(0)  # 次日開盤/以昨日訊號執行

    # 計算每日報酬
    data["Market_Return"] = data["Close"].pct_change().fillna(0)
    data["Strategy_Return"] = data["Position"] * data["Market_Return"]

    # 權益曲線
    data["Buy_Hold_Equity"] = initial_capital * (1 + data["Market_Return"]).cumprod()
    data["Strategy_Equity"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()

    # 最大回撤 (Drawdown)
    cum_max = data["Strategy_Equity"].cummax()
    data["Drawdown"] = (data["Strategy_Equity"] - cum_max) / cum_max
    max_drawdown = float(data["Drawdown"].min()) * 100.0

    # 績效統計
    total_return = float((data["Strategy_Equity"].iloc[-1] / initial_capital - 1) * 100.0)
    buy_hold_return = float((data["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1) * 100.0)

    # 年化報酬率 CAGR
    total_days = (data.index[-1] - data.index[0]).days
    cagr = 0.0
    if total_days > 0:
        cagr = float(((data["Strategy_Equity"].iloc[-1] / initial_capital) ** (365.0 / total_days) - 1) * 100.0)

    # 夏普比率 (年化 252 交易日，假設無風險利率 2%)
    rf_daily = 0.02 / 252.0
    excess_ret = data["Strategy_Return"] - rf_daily
    if data["Strategy_Return"].std() > 0:
        sharpe = float(np.sqrt(252) * (excess_ret.mean() / data["Strategy_Return"].std()))
    else:
        sharpe = 0.0

    # 產生交易明細 (Trades Log)
    trades = []
    in_pos = False
    entry_date = None
    entry_price = 0.0

    for i in range(len(data)):
        pos = data["Position"].iloc[i]
        date = data.index[i]
        price = data["Close"].iloc[i]

        if not in_pos and pos == 1:
            in_pos = True
            entry_date = date
            entry_price = price
        elif in_pos and pos == 0:
            in_pos = False
            exit_date = date
            exit_price = price
            ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
            holding_days = (exit_date - entry_date).days
            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "return_pct": ret_pct,
                "holding_days": holding_days,
                "is_win": ret_pct > 0
            })

    # 若結束時仍在持倉
    if in_pos:
        exit_date = data.index[-1]
        exit_price = data["Close"].iloc[-1]
        ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
        holding_days = (exit_date - entry_date).days
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "return_pct": ret_pct,
            "holding_days": holding_days,
            "is_win": ret_pct > 0
        })

    # 勝率與獲利因子
    total_trades = len(trades)
    win_trades = [t for t in trades if t["is_win"]]
    win_rate = (len(win_trades) / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(t["return_pct"] for t in win_trades)
    gross_loss = abs(sum(t["return_pct"] for t in trades if not t["is_win"]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

    metrics = {
        "initial_capital": initial_capital,
        "final_capital": float(data["Strategy_Equity"].iloc[-1]),
        "total_return_pct": total_return,
        "buy_hold_return_pct": buy_hold_return,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor
    }

    return metrics, data[["Strategy_Equity", "Buy_Hold_Equity", "Drawdown"]], trades
