"""
Charts Module for Taiwan Stock & Global Market Analysis Dashboard
Builds rich, interactive Plotly visualizations for K-line, technical indicators,
ADR premium trends, macro comparison, PE rivers, order book (五檔), and backtesting results.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def get_color_scheme(tw_style: bool = True):
    """
    色彩配置:
    tw_style=True (台股習慣: 紅漲綠跌)
    tw_style=False (美股/國際習慣: 綠漲紅跌)
    """
    if tw_style:
        return {
            "up_color": "#FF4D4D",      # 紅 (漲)
            "down_color": "#00CC66",    # 綠 (跌)
            "up_fill": "#FF4D4D",
            "down_fill": "#00CC66",
            "vol_up": "rgba(255, 77, 77, 0.6)",
            "vol_down": "rgba(0, 204, 102, 0.6)"
        }
    else:
        return {
            "up_color": "#00CC66",      # 綠 (漲)
            "down_color": "#FF4D4D",    # 紅 (跌)
            "up_fill": "#00CC66",
            "down_fill": "#FF4D4D",
            "vol_up": "rgba(0, 204, 102, 0.6)",
            "vol_down": "rgba(255, 77, 77, 0.6)"
        }


def plot_stock_candlestick(
    df: pd.DataFrame,
    symbol_name: str,
    selected_mas: List[int] = [5, 10, 20, 60],
    show_bbands: bool = False,
    sub_indicator: str = "KD",
    tw_style: bool = True,
    chart_type: str = "Candlestick"
) -> go.Figure:
    """
    繪製互動式技術分析 K 線圖 + 成交量 + 副圖指標
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="查無資料，請確認代碼或日期範圍", showarrow=False, font=dict(size=18))
        return fig

    colors = get_color_scheme(tw_style)

    # 決定子圖數量與高度比例
    has_sub = sub_indicator in ["KD", "RSI", "MACD", "BIAS"]
    if has_sub:
        rows = 3
        row_heights = [0.60, 0.20, 0.20]
        subplot_titles = [f"{symbol_name} 股價走勢", "成交量 (Volume)", f"技術指標 ({sub_indicator})"]
    else:
        rows = 2
        row_heights = [0.75, 0.25]
        subplot_titles = [f"{symbol_name} 股價走勢", "成交量 (Volume)"]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # 1. 主圖：K線或收盤線
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K線",
                increasing_line_color=colors["up_color"],
                decreasing_line_color=colors["down_color"],
                increasing_fillcolor=colors["up_fill"],
                decreasing_fillcolor=colors["down_fill"]
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="收盤價",
                line=dict(color="#3867d6", width=2)
            ),
            row=1, col=1
        )

    # 疊加均線
    ma_colors = {
        5: "#ff9f43",
        10: "#0abde3",
        20: "#ee5253",
        60: "#10ac84",
        120: "#5f27cd",
        240: "#341f97"
    }
    for period in selected_mas:
        col_name = f"MA{period}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode="lines",
                    name=f"MA{period}",
                    line=dict(width=1.5, color=ma_colors.get(period, "#888888"))
                ),
                row=1, col=1
            )

    # 疊加布林通道
    if show_bbands and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                mode="lines",
                name="布林上軌",
                line=dict(width=1, color="rgba(75, 123, 236, 0.7)", dash="dot")
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                mode="lines",
                name="布林下軌",
                fill="tonexty",
                fillcolor="rgba(75, 123, 236, 0.08)",
                line=dict(width=1, color="rgba(75, 123, 236, 0.7)", dash="dot")
            ),
            row=1, col=1
        )

    # 2. 第二圖：成交量
    vol_colors = np.where(
        df["Close"] >= df["Open"],
        colors["vol_up"],
        colors["vol_down"]
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="成交量",
            marker_color=vol_colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    if "Vol_MA5" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Vol_MA5"],
                mode="lines",
                name="5日均量",
                line=dict(width=1.2, color="#f39c12")
            ),
            row=2, col=1
        )
    if "Vol_MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Vol_MA20"],
                mode="lines",
                name="20日均量",
                line=dict(width=1.2, color="#9b59b6")
            ),
            row=2, col=1
        )

    # 3. 第三圖：副指標
    if has_sub:
        if sub_indicator == "KD" and "K" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["K"], mode="lines", name="K(9,3)", line=dict(color="#f39c12", width=1.5)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["D"], mode="lines", name="D(9,3)", line=dict(color="#2980b9", width=1.5)),
                row=3, col=1
            )
            fig.add_hline(y=80, line_dash="dot", line_color="rgba(231, 76, 60, 0.5)", row=3, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)", row=3, col=1)

        elif sub_indicator == "RSI" and "RSI_6" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["RSI_6"], mode="lines", name="RSI(6)", line=dict(color="#e74c3c", width=1.5)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["RSI_12"], mode="lines", name="RSI(12)", line=dict(color="#3498db", width=1.5)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dot", line_color="rgba(231, 76, 60, 0.5)", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)", row=3, col=1)

        elif sub_indicator == "MACD" and "MACD_DIF" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["MACD_DIF"], mode="lines", name="DIF", line=dict(color="#e67e22", width=1.5)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="MACD (DEM)", line=dict(color="#2980b9", width=1.5)),
                row=3, col=1
            )
            hist_colors = np.where(df["MACD_Hist"] >= 0, colors["up_color"], colors["down_color"])
            fig.add_trace(
                go.Bar(x=df.index, y=df["MACD_Hist"], name="OSC 柱狀", marker_color=hist_colors, opacity=0.7),
                row=3, col=1
            )
            fig.add_hline(y=0, line_color="#7f8c8d", line_width=1, row=3, col=1)

        elif sub_indicator == "BIAS" and "BIAS_20" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["BIAS_5"], mode="lines", name="5日乖離", line=dict(color="#e74c3c", width=1.2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["BIAS_20"], mode="lines", name="20日乖離", line=dict(color="#3498db", width=1.5)),
                row=3, col=1
            )
            fig.add_hline(y=0, line_color="#7f8c8d", line_width=1, row=3, col=1)

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])
        ]
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=720,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_best5_orderbook(snapshot: Dict[str, Any], tw_style: bool = True) -> go.Figure:
    """
    繪製永豐 Shioaji 最佳五檔買賣委託價量圖 (Best 5 Bids & Asks Order Book Depth)
    """
    fig = go.Figure()
    if not snapshot or "bids" not in snapshot or "asks" not in snapshot:
        fig.add_annotation(text="無五檔報價數據", showarrow=False)
        return fig

    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    
    if not bids and not asks:
        fig.add_annotation(text="目前無委託掛單", showarrow=False)
        return fig

    # 準備買盤資料 (由高到低排)
    bid_prices = [f"買 {b['price']:.2f}" for b in bids]
    bid_vols = [b["volume"] for b in bids]

    # 準備賣盤資料 (由低到高排)
    ask_prices = [f"賣 {a['price']:.2f}" for a in asks]
    ask_vols = [a["volume"] for a in asks]

    colors = get_color_scheme(tw_style)

    # 買單 (左側/紅或綠)
    if bids:
        fig.add_trace(go.Bar(
            y=bid_prices[::-1],
            x=bid_vols[::-1],
            orientation="h",
            name="委買量 (Bids)",
            marker_color=colors["up_color"],
            text=[f"{v:,} 張" for v in bid_vols[::-1]],
            textposition="auto"
        ))

    # 賣單 (右側/綠或紅)
    if asks:
        fig.add_trace(go.Bar(
            y=ask_prices[::-1],
            x=ask_vols[::-1],
            orientation="h",
            name="委賣量 (Asks)",
            marker_color=colors["down_color"],
            text=[f"{v:,} 張" for v in ask_vols[::-1]],
            textposition="auto"
        ))

    stock_title = f"{snapshot.get('name', '')} ({snapshot.get('code', '')}) 最佳五檔委託即時報價"
    fig.update_layout(
        title=stock_title,
        xaxis_title="委託張數 / 口數",
        yaxis_title="報價檔位",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_adr_premium(df: pd.DataFrame) -> go.Figure:
    """
    繪製台積電 ADR (TSM) 溢價率走勢與折合台幣對照圖
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="ADR 溢價率資料載入中或無法取得", showarrow=False)
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=["台積電 ADR (折合每股NTD) vs 台積電現股價格 (2330)", "ADR 溢價率走勢 (%)"]
    )

    # 主圖：ADR 折算價格 vs 2330 價格
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ADR_TWD_Equiv"],
            mode="lines",
            name="TSM ADR (折合每股NTD)",
            line=dict(color="#3867d6", width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["TW_2330"],
            mode="lines",
            name="台積電現股 (2330.TW)",
            line=dict(color="#eb3b5a", width=2)
        ),
        row=1, col=1
    )

    # 副圖：溢價率長條與折線圖
    bar_colors = np.where(df["Premium_Pct"] >= 0, "#eb3b5a", "#20bf6b")
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Premium_Pct"],
            name="溢價率 (%)",
            marker_color=bar_colors,
            opacity=0.65
        ),
        row=2, col=1
    )
    premium_ma20 = df["Premium_Pct"].rolling(20).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=premium_ma20,
            mode="lines",
            name="溢價率 20日均線",
            line=dict(color="#fa8231", width=1.5)
        ),
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#777777", row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        height=580,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_macro_comparison(macro_dfs: Dict[str, pd.Series]) -> go.Figure:
    """
    繪製國際指標與台股連動標準化累積漲跌幅走勢比較 (歸一化 0%)
    """
    fig = go.Figure()
    if not macro_dfs:
        fig.add_annotation(text="無比較資料", showarrow=False)
        return fig

    colors = ["#2ed573", "#1e90ff", "#ff4757", "#ffa502", "#a55eea", "#2bcbba", "#ff6348", "#5352ed"]
    c_idx = 0

    for name, series in macro_dfs.items():
        if series is not None and not series.empty:
            s_clean = series.dropna()
            if len(s_clean) > 0:
                base_val = s_clean.iloc[0]
                normalized = ((s_clean - base_val) / base_val) * 100.0
                fig.add_trace(
                    go.Scatter(
                        x=normalized.index,
                        y=normalized,
                        mode="lines",
                        name=name,
                        line=dict(width=2, color=colors[c_idx % len(colors)])
                    )
                )
                c_idx += 1

    fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    fig.update_layout(
        title="國際總經指標與主要市場累積報酬率走勢比較 (基準點歸一化 %)",
        yaxis_title="累積漲跌幅 (%)",
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_pe_river(df: pd.DataFrame, symbol_name: str) -> go.Figure:
    """
    繪製本益比河流圖 (P/E Bands Chart)
    """
    fig = go.Figure()
    if df.empty or "PE_12X" not in df.columns:
        fig.add_annotation(text="無本益比河流圖資料", showarrow=False)
        return fig

    fig.add_trace(go.Scatter(x=df.index, y=df["PE_26X"], mode="lines", name="26X 本益比", line=dict(color="rgba(235, 59, 90, 0.5)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["PE_22X"], mode="lines", name="22X 本益比", line=dict(color="rgba(250, 130, 49, 0.5)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["PE_18X"], mode="lines", name="18X 本益比", line=dict(color="rgba(46, 213, 115, 0.5)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["PE_15X"], mode="lines", name="15X 本益比", line=dict(color="rgba(47, 86, 233, 0.5)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["PE_12X"], mode="lines", name="12X 本益比", line=dict(color="rgba(30, 144, 255, 0.5)", width=1, dash="dot")))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=f"{symbol_name} 收盤價",
            line=dict(color="#2c3e50", width=2.5)
        )
    )

    fig.update_layout(
        title=f"{symbol_name} 本益比評價河流圖 (P/E River Valuation)",
        yaxis_title="股價 (TWD)",
        hovermode="x unified",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_backtest_chart(
    equity_curve: pd.DataFrame,
    trades: List[Dict[str, Any]],
    price_df: pd.DataFrame,
    strategy_name: str
) -> go.Figure:
    """
    繪製量化策略回測結果（權益曲線、買賣點標記、最大回撤）
    """
    if equity_curve.empty:
        fig = go.Figure()
        fig.add_annotation(text="無回測資料", showarrow=False)
        return fig

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=[f"策略交易訊號點 ({strategy_name})", "累計權益曲線 (Portfolio Equity) vs 買入持有 (Buy & Hold)", "策略回撤幅度 (Drawdown %)"]
    )

    # 1. 價格與買賣標記
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df["Close"],
            mode="lines",
            name="標的股價",
            line=dict(color="#747d8c", width=1.5)
        ),
        row=1, col=1
    )

    buy_dates = [t["entry_date"] for t in trades if t["entry_date"] in price_df.index]
    buy_prices = [price_df.loc[d, "Close"] for d in buy_dates]
    sell_dates = [t["exit_date"] for t in trades if t.get("exit_date") and t["exit_date"] in price_df.index]
    sell_prices = [price_df.loc[d, "Close"] for d in sell_dates]

    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="買進 (Buy)",
                marker=dict(symbol="triangle-up", color="#eb3b5a", size=10)
            ),
            row=1, col=1
        )
    if sell_dates:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                name="賣出 (Sell)",
                marker=dict(symbol="triangle-down", color="#20bf6b", size=10)
            ),
            row=1, col=1
        )

    # 2. 累計權益曲線
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Strategy_Equity"],
            mode="lines",
            name="策略權益 (Strategy)",
            line=dict(color="#3867d6", width=2)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Buy_Hold_Equity"],
            mode="lines",
            name="買入持有基準 (Buy & Hold)",
            line=dict(color="#a4b0be", width=1.5, dash="dash")
        ),
        row=2, col=1
    )

    # 3. 回撤 (Drawdown)
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Drawdown"] * 100,
            mode="lines",
            name="策略回撤 (%)",
            fill="tozeroy",
            fillcolor="rgba(235, 59, 90, 0.2)",
            line=dict(color="#eb3b5a", width=1)
        ),
        row=3, col=1
    )

    fig.update_layout(
        hovermode="x unified",
        height=750,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_options_payoff(
    prices: np.ndarray,
    payoffs: np.ndarray,
    spot_price: float,
    strategy_name: str,
    breakevens: list = [],
    point_value: int = 50
) -> go.Figure:
    """
    繪製選擇權策略到期損益圖 (Payoff Diagram)
    :param prices: 標的到期價格陣列
    :param payoffs: 到期損益點數陣列
    :param spot_price: 當前現貨價格
    :param strategy_name: 策略名稱
    :param breakevens: 損益兩平點清單
    :param point_value: 契約每點價值 (台指選擇權為 50 元/點)
    """
    fig = go.Figure()

    # 分割正負獲利區域上色
    payoff_twd = payoffs * point_value

    # 損益主曲線
    fig.add_trace(
        go.Scatter(
            x=prices,
            y=payoffs,
            mode="lines",
            name="到期損益 (點數)",
            line=dict(color="#3867d6", width=3)
        )
    )

    # 零軸基準線 (0 點)
    fig.add_hline(y=0, line_dash="dash", line_color="#747d8c", annotation_text="損益兩平 (0點)", annotation_position="bottom right")

    # 現價垂直線
    fig.add_vline(
        x=spot_price,
        line_dash="dot",
        line_color="#ff9f43",
        annotation_text=f"當前現價: {spot_price:,.1f}",
        annotation_position="top left"
    )

    # 標註損益兩平點 (BEP)
    for bep in breakevens:
        fig.add_vline(
            x=bep,
            line_dash="dash",
            line_color="#eb3b5a",
            annotation_text=f"BEP: {bep:,.1f}",
            annotation_position="top right"
        )

    # 綠色獲利區與紅色虧損區著色
    fig.add_trace(
        go.Scatter(
            x=prices,
            y=np.maximum(payoffs, 0),
            fill="tozeroy",
            fillcolor="rgba(46, 213, 115, 0.25)",
            mode="none",
            name="獲利區間 (Profit)",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices,
            y=np.minimum(payoffs, 0),
            fill="tozeroy",
            fillcolor="rgba(255, 71, 87, 0.25)",
            mode="none",
            name="虧損區間 (Loss)",
            showlegend=True
        )
    )

    fig.update_layout(
        title=f"🎯 【{strategy_name}】到期損益模擬圖 (Payoff Diagram)",
        xaxis_title="標的到期結算價格 (Underlying Spot Price at Expiry)",
        yaxis_title="到期損益點數 (Points)",
        hovermode="x unified",
        height=480,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

