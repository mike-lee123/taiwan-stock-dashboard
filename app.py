"""
Taiwan Stock & Global Market Analysis Dashboard (台灣股市與國際連動分析儀表板)
Main Streamlit Application with SinoPac Shioaji API Integration
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 引入自訂模組
from data_loader import (
    POPULAR_TW_STOCKS,
    THEMATIC_STOCK_GROUPS,
    MACRO_BENCHMARKS,
    normalize_symbol,
    fetch_stock_history,
    fetch_macro_data,
    calculate_adr_premium,
    fetch_stock_fundamentals,
    calculate_pe_bands
)
from indicators import compute_all_indicators
from charts import (
    plot_stock_candlestick,
    plot_adr_premium,
    plot_macro_comparison,
    plot_pe_river,
    plot_backtest_chart,
    plot_best5_orderbook,
    plot_options_payoff
)
from backtest import run_strategy_backtest
from shioaji_client import ShioajiManager

# 嘗試載入 Capital API 本機交易模組 (支援 Windows 本機與雲端環境無痛相容)
try:
    capital_sub_dir = os.path.join(os.path.dirname(__file__), "Capital API Desktop Trading")
    if capital_sub_dir not in sys.path:
        sys.path.insert(0, capital_sub_dir)
    from capital_client import CapitalManager
    from capital_trading_view import render_capital_trading_desk
    HAS_CAPITAL = True
except Exception:
    HAS_CAPITAL = False
    CapitalManager = None
    render_capital_trading_desk = None
from options import (
    black_scholes_pricing_and_greeks,
    get_weekly_txo_status,
    generate_txo_strike_ladder,
    calculate_strategy_payoff,
    recommend_options_play
)

# 設定頁面配置
st.set_page_config(
    page_title="台股與國際連動分析儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 4px solid #3867d6;
        margin-bottom: 8px;
    }
    .sub-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
        color: #2c3e50;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f2f6;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3867d6 !important;
        color: white !important;
    }
    .shioaji-badge-connected {
        background-color: #2ed573;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .shioaji-badge-disconnected {
        background-color: #747d8c;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 初始化券商 API 管理員
sj_mgr = ShioajiManager.get_instance()
cap_mgr = CapitalManager.get_instance() if (HAS_CAPITAL and CapitalManager is not None) else None

# 快取資料抓取函數
@st.cache_data(ttl=300)
def get_cached_stock_data(ticker: str, period: str, interval: str):
    return fetch_stock_history(ticker, period=period, interval=interval)


@st.cache_data(ttl=300)
def get_cached_macro_data():
    return fetch_macro_data()


@st.cache_data(ttl=300)
def get_cached_adr_premium(period: str):
    return calculate_adr_premium(period=period)


@st.cache_data(ttl=600)
def get_cached_fundamentals(ticker: str):
    return fetch_stock_fundamentals(ticker)


# ==============================================================================
# 側邊欄控制面板 (Sidebar)
# ==============================================================================
st.sidebar.title("📊 台股分析控制台")

# 0. 群益金融 API (Capital API) 本機連線設定
if cap_mgr is not None:
    with st.sidebar.expander("💼 群益金融 API (Capital API) 本機設定", expanded=not cap_mgr.is_connected()):
        cap_env_st = cap_mgr.get_env_status()
        if cap_mgr.is_connected():
            cap_env_label = "模擬沙盒" if cap_mgr.is_simulation() else "正式實盤"
            st.markdown(f'<span style="background-color:#2ed573; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">🟢 群益已連線 ({cap_env_label})</span>', unsafe_allow_html=True)
            st.caption(f"帳號: `{cap_mgr.get_user_id()}`")
            if st.button("中斷群益連線", key="btn_logout_cap_main"):
                cap_mgr.logout()
                st.rerun()
        else:
            st.markdown(
                '<span style="background-color:#747d8c; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">⚪ 群益未連線</span> '
                '<span style="background-color:#1e90ff; color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:bold;">🌐 Yahoo Finance 即時連線</span>',
                unsafe_allow_html=True
            )
            st.caption("💡 提示：未連線實盤時，極速交易室自動以 Yahoo Finance 提供現貨、台指期與海期即時行情。")
            in_cap_uid = st.text_input("身分證字號", value=os.getenv("CAPITAL_USER_ID", "A123456789"), key="in_cap_uid_main")
            in_cap_pwd = st.text_input("登入密碼", value=os.getenv("CAPITAL_PASSWORD", "sim_pass"), type="password", key="in_cap_pwd_main")
            in_cap_sim = st.checkbox("啟用模擬交易沙盒 (免憑證立即測試)", value=True, key="chk_cap_sim_main")
            
            if not cap_env_st["skcom_registered"]:
                st.caption("ℹ️ 本機未註冊 SKCOM.dll，將以高擬真模擬沙盒運行。")
                
            if st.button("🚀 連線/啟動群益 API", key="btn_login_cap_main"):
                u_val = in_cap_uid if in_cap_uid else "A123456789"
                p_val = in_cap_pwd if in_cap_pwd else "sim_pass"
                succ, msg = cap_mgr.login(user_id=u_val, password=p_val, simulation=in_cap_sim)
                if succ:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

# 1. 永豐 Shioaji API 連線設定
with st.sidebar.expander("🔑 永豐證券 Shioaji API 設定", expanded=not sj_mgr.is_connected()):
    if sj_mgr.is_connected():
        env_label = "模擬環境" if sj_mgr.is_simulation() else "正式環境"
        st.markdown(f'<span class="shioaji-badge-connected">🟢 Shioaji 已連線 ({env_label})</span>', unsafe_allow_html=True)
        if st.button("中斷 Shioaji 連線"):
            sj_mgr.logout()
            st.session_state["shioaji_connected"] = False
            st.rerun()
    else:
        st.markdown('<span class="shioaji-badge-disconnected">⚪ Shioaji 未連線 (使用公開歷史行情)</span>', unsafe_allow_html=True)
        input_api_key = st.text_input("Shioaji API Key", value=os.getenv("SHIOAJI_API_KEY", ""), type="password")
        input_secret_key = st.text_input("Shioaji Secret Key", value=os.getenv("SHIOAJI_SECRET_KEY", ""), type="password")
        is_sim = st.checkbox("使用模擬環境 (Simulation)", value=True)
        
        if st.button("🚀 連線至永豐 Shioaji"):
            if not input_api_key or not input_secret_key:
                st.error("請輸入 API Key 與 Secret Key (或建立 .env 檔案)")
            else:
                with st.spinner("正在登入永豐金證券 API 並下載合約檔..."):
                    success, msg = sj_mgr.login(
                        api_key=input_api_key,
                        secret_key=input_secret_key,
                        simulation=is_sim
                    )
                    if success:
                        st.session_state["shioaji_connected"] = True
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

st.sidebar.markdown("---")

# 2. 股票搜尋與選擇
st.sidebar.subheader("🎯 標的選擇")
preset_options = ["-- 手動輸入代碼 --"] + [f"{item['code']} {item['name']} ({item['sector']})" for item in POPULAR_TW_STOCKS]
selected_preset = st.sidebar.selectbox("熱門台股快捷清單", preset_options, index=1)

if selected_preset == "-- 手動輸入代碼 --":
    user_input = st.sidebar.text_input("輸入台股代碼或美股代碼 (例: 2330, 2454, TSM, AAPL)", value="2330")
else:
    code_part = selected_preset.split(" ")[0]
    user_input = code_part

target_symbol, target_name = normalize_symbol(user_input)
st.sidebar.info(f"當前分析標的：**{target_name}** (`{target_symbol}`)")

# 3. 時間範圍與週期
st.sidebar.subheader("📅 時間範圍與週期")
period_options = {
    "1個月": "1mo",
    "3個月": "3mo",
    "6個月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "3年": "3y",
    "5年": "5y",
    "年初至今 (YTD)": "ytd"
}
selected_period_label = st.sidebar.selectbox("歷史資料長度", list(period_options.keys()), index=3)
selected_period = period_options[selected_period_label]

interval_options = {"日K (1日)": "1d", "週K (1週)": "1wk", "月K (1月)": "1mo"}
selected_interval_label = st.sidebar.selectbox("K線週期", list(interval_options.keys()), index=0)
selected_interval = interval_options[selected_interval_label]

# 4. 圖表視覺與指標設定
st.sidebar.subheader("🎨 圖表與指標設定")
color_style_choice = st.sidebar.radio(
    "漲跌配色模式",
    ["🔴 台股慣用 (紅漲 綠跌)", "🟢 國際慣用 (綠漲 紅跌)"],
    index=0
)
tw_color_style = "紅漲" in color_style_choice

chart_type = st.sidebar.radio("主圖類型", ["Candlestick", "Line (收盤線)"], index=0)

show_ma = st.sidebar.multiselect(
    "顯示均線 (SMA)",
    [5, 10, 20, 60, 120, 240],
    default=[5, 10, 20, 60]
)

show_bbands = st.sidebar.checkbox("顯示布林通道 (Bollinger Bands 20, 2σ)", value=False)

sub_indicator = st.sidebar.selectbox(
    "副圖技術指標",
    ["KD", "RSI", "MACD", "BIAS", "無"],
    index=0
)

# 重新整理按鈕
if st.sidebar.button("🔄 重新整理所有即時資料"):
    st.cache_data.clear()
    st.rerun()

# ==============================================================================
# 頂部即時總經行情跑馬卡片 (Macro Benchmark Ribbon)
# ==============================================================================
macro_data = get_cached_macro_data()

st.markdown("### 🌐 國際關鍵指標與台股連動即時看板")
macro_cols = st.columns(len(macro_data))

for idx, (m_name, m_info) in enumerate(macro_data.items()):
    col = macro_cols[idx]
    with col:
        p = m_info["price"]
        chg = m_info["change"]
        pct = m_info["pct_change"]
        if np.isnan(p):
            col.metric(label=m_name.split(" ")[0], value="N/A")
        else:
            delta_str = f"{chg:+.2f} ({pct:+.2f}%)"
            short_name = m_name.split(" ")[0]
            col.metric(label=short_name, value=f"{p:,.2f}", delta=delta_str)

st.markdown("---")

# ==============================================================================
# 主要核心分頁 (Tabs)
# ==============================================================================
tabs = st.tabs([
    "🎯 選擇權策略與決策中心",
    "💼 群益本機極速交易室",
    "⚡ 永豐即時五檔與台指期",
    "🦅 多維策略選股獵鷹",
    "🌐 國際與總經連動",
    "📈 個股技術分析",
    "🏢 基本面與估值河流圖",
    "👥 籌碼與法人動態",
    "⚔️ 多股報酬比較",
    "🧪 量化策略回測實驗室"
])

# 抓取個股資料與指標
raw_df = get_cached_stock_data(target_symbol, selected_period, selected_interval)
if not raw_df.empty:
    df_with_ind = compute_all_indicators(raw_df)
else:
    df_with_ind = pd.DataFrame()


# ------------------------------------------------------------------------------
# 分頁 0: 🎯 選擇權策略與決策中心 (Options Strategy & Decision Hub)
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="sub-header-title">🎯 選擇權標的選擇、進出場決策與損益模擬中心 (Options Strategy Hub)</div>', unsafe_allow_html=True)
    
    # 頂部：週選擇權 (TXO) 生命週期與即時大盤情勢
    txo_info = get_weekly_txo_status()
    taiex_current = macro_data.get("台股加權指數 (TAIEX)", {}).get("price", 45975.22)
    if np.isnan(taiex_current):
        taiex_current = 45975.22

    col_txo1, col_txo2, col_txo3, col_txo4 = st.columns(4)
    with col_txo1:
        st.metric("當前加權指數 (現價)", f"{taiex_current:,.2f} 點")
    with col_txo2:
        st.metric("今日週期位置", f"{txo_info['day_name']}")
    with col_txo3:
        st.metric("週選結算倒數", f"{txo_info['days_to_wed']} 天")
    with col_txo4:
        st.metric("週選生命週期階段", txo_info['stage'].split(" ")[1] if " " in txo_info['stage'] else txo_info['stage'])

    st.info(f"💡 **今日週選操盤戰略 ({txo_info['stage']})**：{txo_info['strategy_tip']} (Theta 狀態: `{txo_info['theta_desc']}` | Gamma 狀態: `{txo_info['gamma_desc']}`)")

    st.markdown("---")

    # 子分頁：五大核心功能
    opt_sub_tabs = st.tabs([
        "🧭 智能策略推薦與進出場判斷",
        "📈 策略到期損益模擬圖 (Payoff)",
        "🧮 Black-Scholes & Greeks 計算機",
        "🪜 台指選擇權履約價鏈階梯表",
        "💰 下單口數與資金部位試算 (Position Sizer)"
    ])

    # 1. 智能策略推薦與進出場判斷
    with opt_sub_tabs[0]:
        st.subheader("🧭 選擇權標的與進出場決策引導矩陣")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            user_dir = st.selectbox(
                "1. 行情方向與動能預期",
                [
                    "🚀 強烈看大漲 (帶量突破/長紅發動)",
                    "📈 溫和看多 (沿均線上漲/上升通道)",
                    "⚖️ 盤整震盪 / 區間橫盤 (做賣方收租)",
                    "📉 溫和看空 (反彈遇阻/量縮陰跌)",
                    "💥 強烈看大跌 (長黑跌破/空頭吞噬)",
                    "⚡ 雙向大暴走 (重大事件/結算日急拉急殺)"
                ],
                index=0
            )
        with col_in2:
            user_iv = st.selectbox(
                "2. 隱含波動率 (IV Rank) 環境",
                [
                    "🟢 低波動 (IV Rank < 35% 權利金便宜，適合買方)",
                    "🟡 中波動 (IV Rank 35% ~ 65% 適合垂直價差)",
                    "🔴 高波動 (IV Rank > 65% 權利金昂貴，適合賣方收租)"
                ],
                index=0
            )
        with col_in3:
            user_horizon = st.selectbox(
                "3. 操作週期與天期 (DTE)",
                [
                    "⚡ 週選擇權極短線 / 當沖 (0 ~ 7 天)",
                    "🗓️ 月選擇權波段 (30 ~ 60 天)"
                ],
                index=0
            )

        # 計算推薦策略
        rec = recommend_options_play(user_dir, user_iv, user_horizon)

        st.markdown(f"""
        <div style="background-color: #f1f6ff; border-left: 5px solid #3867d6; padding: 16px; border-radius: 8px; margin: 15px 0;">
            <h4 style="color: #2c3e50; margin-top: 0;">🎯 系統推薦首選策略：<span style="color: #3867d6;">{rec['strategy']}</span> <span style="font-size: 0.85rem; background: #3867d6; color: white; padding: 2px 8px; border-radius: 10px;">{rec['type']}</span></h4>
            <p><strong>💡 策略依據</strong>：{rec['reason']}</p>
            <p><strong>🎯 建議履約價選取 (Strike & Delta)</strong>：<code>{rec['strike_advice']}</code></p>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"**⚡ 進場觸發訊號 (Entry Triggers)**<br>{rec['entry_trigger']}", unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"**🛑 嚴格停損條件 (Stop-Loss Rules)**<br><span style='color:#eb3b5a; font-weight:bold;'>{rec['stop_loss']}</span>", unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"**🎯 獲利停利規則 (Take-Profit Rules)**<br><span style='color:#20bf6b; font-weight:bold;'>{rec['take_profit']}</span>", unsafe_allow_html=True)

        # 檢核清單
        with st.expander("📋 下單前 5 步自我檢核清單 (Trader's Checklist)", expanded=False):
            st.markdown("""
            - [ ] **流動性確認**：近月合約 OI 是否充足？買賣價差 (Spread) 是否小於 3%？
            - [ ] **IV 匹對**：買方是否避開高 IV 追價？賣方是否選在 IV 高檔與關鍵支撐壓力外？
            - [ ] **停損設定**：下單當下是否已設好停損價位或時間停損（3~5 天未動即撤退）？
            - [ ] **停利分批**：買方翻倍是否已規劃好先收回 50% 本金？
            - [ ] **資金規模**：單筆部位本金是否控制在總資金的 **2% ~ 5%** 以內？
            """)

    # 2. 策略到期損益模擬圖
    with opt_sub_tabs[1]:
        st.subheader("📈 選擇權策略到期損益圖 (Payoff Diagram Simulator)")
        
        col_p1, col_p2 = st.columns([0.4, 0.6])
        with col_p1:
            selected_payoff_strat = st.selectbox(
                "選擇欲模擬之策略",
                [
                    "Long Call (買進買權)",
                    "Long Put (買進賣權)",
                    "Bull Call Spread (買權牛市價差)",
                    "Bear Put Spread (賣權熊市價差)",
                    "Bull Put Spread (賣權牛市信用價差)",
                    "Iron Condor (鐵鷹價差/雙向收租)",
                    "Long Straddle (買進跨式/結算日雙買)"
                ],
                index=0
            )

            # 依據策略動態呈現輸入欄位
            param_dict = {}
            if selected_payoff_strat in ["Long Call (買進買權)", "Long Put (買進賣權)"]:
                param_dict["k1"] = st.number_input("履約價 (Strike K)", value=float(round(taiex_current / 100) * 100), step=100.0)
                param_dict["prem1"] = st.number_input("付出權利金 (點數)", value=50.0, step=5.0)
            elif selected_payoff_strat in ["Bull Call Spread (買權牛市價差)", "Bear Put Spread (賣權熊市價差)"]:
                c_k1, c_k2 = st.columns(2)
                param_dict["k1"] = c_k1.number_input("主要買進履約價 K1", value=float(round(taiex_current / 100) * 100), step=100.0)
                param_dict["k2"] = c_k2.number_input("保護賣出履約價 K2", value=float(round(taiex_current / 100) * 100 + 200), step=100.0)
                c_p1, c_p2 = st.columns(2)
                param_dict["prem1"] = c_p1.number_input("買進權利金", value=80.0, step=5.0)
                param_dict["prem2"] = c_p2.number_input("賣出權利金", value=30.0, step=5.0)
            elif selected_payoff_strat == "Bull Put Spread (賣權牛市信用價差)":
                c_k1, c_k2 = st.columns(2)
                param_dict["k1"] = c_k1.number_input("賣出高履約價 Put (K_sell)", value=float(round(taiex_current / 100) * 100 - 100), step=100.0)
                param_dict["k2"] = c_k2.number_input("買進更低履約價 Put (K_buy)", value=float(round(taiex_current / 100) * 100 - 300), step=100.0)
                c_p1, c_p2 = st.columns(2)
                param_dict["prem1"] = c_p1.number_input("賣出所得權利金", value=60.0, step=5.0)
                param_dict["prem2"] = c_p2.number_input("買進保護權利金", value=20.0, step=5.0)
            elif selected_payoff_strat == "Iron Condor (鐵鷹價差/雙向收租)":
                ic_c1, ic_c2 = st.columns(2)
                param_dict["put_sell"] = ic_c1.number_input("賣出 Put", value=float(round(taiex_current / 100) * 100 - 200), step=100.0)
                param_dict["put_buy"] = ic_c2.number_input("買進保護 Put", value=float(round(taiex_current / 100) * 100 - 400), step=100.0)
                ic_c3, ic_c4 = st.columns(2)
                param_dict["call_sell"] = ic_c3.number_input("賣出 Call", value=float(round(taiex_current / 100) * 100 + 200), step=100.0)
                param_dict["call_buy"] = ic_c4.number_input("買進保護 Call", value=float(round(taiex_current / 100) * 100 + 400), step=100.0)
                param_dict["prem_put_s"] = 40.0
                param_dict["prem_put_b"] = 15.0
                param_dict["prem_call_s"] = 40.0
                param_dict["prem_call_b"] = 15.0
            elif selected_payoff_strat == "Long Straddle (買進跨式/結算日雙買)":
                param_dict["k1"] = st.number_input("價平履約價 (Strike K)", value=float(round(taiex_current / 100) * 100), step=100.0)
                c_s1, c_s2 = st.columns(2)
                param_dict["prem1"] = c_s1.number_input("Call 權利金", value=45.0, step=5.0)
                param_dict["prem2"] = c_s2.number_input("Put 權利金", value=45.0, step=5.0)

            # 計算損益曲線
            p_prices, p_payoffs, max_p, max_l, p_bep, p_summary = calculate_strategy_payoff(
                selected_payoff_strat, taiex_current, param_dict
            )

            st.markdown(f"**💡 策略損益摘要**：\n{p_summary}")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("理論最大獲利", max_p)
            with m2:
                st.metric("理論最大風險", max_l)

        with col_p2:
            fig_payoff = plot_options_payoff(
                prices=p_prices,
                payoffs=p_payoffs,
                spot_price=taiex_current,
                strategy_name=selected_payoff_strat,
                breakevens=p_bep,
                point_value=50
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

    # 3. Black-Scholes 與 Greeks 計算機
    with opt_sub_tabs[2]:
        st.subheader("🧮 Black-Scholes 選擇權理論定價與 Greeks 計算機")
        
        g_c1, g_c2, g_c3, g_c4, g_c5 = st.columns(5)
        with g_c1:
            g_spot = st.number_input("標的現價 S", value=float(taiex_current), step=50.0)
        with g_c2:
            g_strike = st.number_input("履約價 K", value=float(round(taiex_current / 100) * 100), step=100.0)
        with g_c3:
            g_dte = st.number_input("到期天數 (DTE)", value=5, min_value=1, max_value=365, step=1)
        with g_c4:
            g_iv = st.number_input("隱含波動率 IV (%)", value=18.0, min_value=1.0, max_value=200.0, step=1.0) / 100.0
        with g_c5:
            g_r = st.number_input("無風險利率 r (%)", value=1.5, step=0.1) / 100.0

        c_res = black_scholes_pricing_and_greeks(g_spot, g_strike, g_dte, g_r, g_iv, "call")
        p_res = black_scholes_pricing_and_greeks(g_spot, g_strike, g_dte, g_r, g_iv, "put")

        greeks_table = pd.DataFrame([
            {
                "指標類型": "理論權利金 (Price)",
                "Call 買權數值": f"{c_res['price']:.2f} 點 (約 NT${c_res['price']*50:,.0f})",
                "Put 賣權數值": f"{p_res['price']:.2f} 點 (約 NT${p_res['price']*50:,.0f})",
                "涵義解釋": "依 Black-Scholes 模型計算之理論公允價格"
            },
            {
                "指標類型": "Delta (Δ)",
                "Call 買權數值": f"{c_res['delta']:+.3f}",
                "Put 賣權數值": f"{p_res['delta']:+.3f}",
                "涵義解釋": "標的每漲 1 點，選擇權價格變動點數（亦代表到期成為價內的機率）"
            },
            {
                "指標類型": "Gamma (Γ)",
                "Call 買權數值": f"{c_res['gamma']:.5f}",
                "Put 賣權數值": f"{p_res['gamma']:.5f}",
                "涵義解釋": "標的每變動 1 點，Delta 的加速度變化量（臨近結算時急劇放大）"
            },
            {
                "指標類型": "Theta (Θ 每日時間耗損)",
                "Call 買權數值": f"{c_res['theta_day']:.2f} 點/日",
                "Put 賣權數值": f"{p_res['theta_day']:.2f} 點/日",
                "涵義解釋": "每過一個日曆日，選擇權自然蒸發的時間價值（買方損失 / 賣方賺取）"
            },
            {
                "指標類型": "Vega (𝒱 波動率敏感度)",
                "Call 買權數值": f"{c_res['vega_1pct']:.2f} 點",
                "Put 賣權數值": f"{p_res['vega_1pct']:.2f} 點",
                "涵義解釋": "當隱含波動率 IV 上升 1% 時，選擇權價格上升的點數"
            }
        ])

        st.dataframe(greeks_table, use_container_width=True, hide_index=True)

    # 4. 台指選擇權履約價鏈階梯表
    with opt_sub_tabs[3]:
        st.subheader(f"🪜 台指選擇權 (TXO) 近月/週選 履約價階梯表 (基準現價: {taiex_current:,.0f} 點)")
        
        ladder_c1, ladder_c2 = st.columns(2)
        ladder_iv = ladder_c1.slider("模擬波動率 (IV %)", min_value=10.0, max_value=50.0, value=18.0, step=1.0) / 100.0
        ladder_dte = ladder_c2.slider("模擬到期天數 (DTE)", min_value=1, max_value=30, value=txo_info['days_to_wed'] if txo_info['days_to_wed'] > 0 else 1)

        ladder_df = generate_txo_strike_ladder(taiex_current, step=100, num_steps=5, iv=ladder_iv, dte=ladder_dte)
        st.dataframe(ladder_df, use_container_width=True, hide_index=True)
        st.caption("註：以上報價為基於 Black-Scholes 模型與無風險利率 1.5% 推估之理論值，實際交易報價請以期交所盤面為準。")

    # 5. 下單口數與資金部位試算 (Position Sizer)
    with opt_sub_tabs[4]:
        st.subheader("💰 選擇權、指數期貨與個股期貨 (台積電/聯發科/大立光) 下單口數與風控試算機")
        st.markdown("支援**台指選擇權、指數期貨 (大台/小台/微台)** 以及 **熱門個股期貨 (台積電、聯發科、大立光、鴻海等)**，依據您的可用資金與單筆最大風險上限，精確推算建議安全口數與停損停利。")

        # 預設股價參考字典
        STOCK_PRICE_REF = {
            "台積電": 2420.0,
            "聯發科": 3985.0,
            "大立光": 7065.0,
            "鴻海": 253.0,
            "廣達": 332.5,
            "長榮": 231.5,
            "京元電": 118.5,
            "世芯": 4065.0
        }

        col_sz1, col_sz2, col_sz3 = st.columns(3)
        with col_sz1:
            product_type = st.selectbox(
                "1. 交易商品類別",
                [
                    "台指選擇權 (TXO) - 買方權利金 (NT$ 50 / 點)",
                    "微型台指期貨 (TMF) - 保證金 (NT$ 10 / 點)",
                    "小型台指期貨 (MXF) - 保證金 (NT$ 50 / 點)",
                    "大型台指期貨 (TXF) - 保證金 (NT$ 200 / 點)",
                    "台積電期貨 (2330 / CDO) - 一口 2,000 股 (2張現股)",
                    "小型台積電期貨 (2330 / QFF) - 一口 100 股 (零股型)",
                    "聯發科期貨 (2454 / DHO) - 一口 2,000 股 (2張現股)",
                    "小型聯發科期貨 (2454) - 一口 100 股 (零股型)",
                    "大立光期貨 (3008 / CAO) - 一口 2,000 股 (2張現股)",
                    "小型大立光期貨 (3008) - 一口 100 股 (零股型)",
                    "鴻海期貨 (2317 / DHF) - 一口 2,000 股 (2張現股)",
                    "廣達期貨 (2382 / JNF) - 一口 2,000 股 (2張現股)",
                    "長榮期貨 (2603 / CZF) - 一口 2,000 股 (2張現股)",
                    "京元電子期貨 (2449) - 一口 2,000 股 (2張現股)",
                    "世芯-KY期貨 (3661) - 一口 2,000 股 (2張現股)",
                    "自訂其他個股期貨 - 一口 2,000 股"
                ],
                index=0
            )
            total_account_fund = st.number_input(
                "可用總資金 / 保證金 (NTD)",
                min_value=10000,
                max_value=100000000,
                value=500000,
                step=50000
            )

        # 判定商品類型
        is_option = "選擇權" in product_type
        is_stock_fut = any(k in product_type for k in ["台積電", "聯發科", "大立光", "鴻海", "廣達", "長榮", "京元", "世芯", "個股期"])
        is_mini_stock = "小型" in product_type

        # 決定預設進場價格
        if is_option:
            def_price = 50.0
        elif is_stock_fut:
            def_price = 100.0
            for name, p in STOCK_PRICE_REF.items():
                if name in product_type:
                    def_price = p
                    break
        else:
            def_price = float(round(taiex_current, 0))

        with col_sz2:
            max_risk_pct = st.slider(
                "2. 單筆最大風險容忍比例 (%)",
                min_value=1.0,
                max_value=20.0,
                value=3.0,
                step=0.5,
                help="建議每筆交易虧損控制在總本金的 2% ~ 5% 以內，長線立於不敗之地。"
            )
            entry_pt = st.number_input(
                "進場價格 / 點數 (NTD)",
                min_value=0.1,
                max_value=100000.0,
                value=def_price,
                step=1.0 if def_price >= 10 else 0.1
            )

        with col_sz3:
            if is_option:
                sl_mode = st.selectbox("3. 停損機制", ["跌幅 -40% 嚴格停損", "跌幅 -50% 停損", "自訂停損點數"], index=0)
                if "40%" in sl_mode:
                    stop_pt = round(entry_pt * 0.6, 1)
                elif "50%" in sl_mode:
                    stop_pt = round(entry_pt * 0.5, 1)
                else:
                    stop_pt = st.number_input("自訂停損點數", min_value=0.0, max_value=entry_pt, value=round(entry_pt * 0.5, 1))

                tp_mode = st.selectbox("4. 停利目標", ["翻倍停利 (+100%)", "三倍噴出 (+200%)", "自訂停利點數"], index=0)
                if "100%" in tp_mode:
                    target_pt = round(entry_pt * 2.0, 1)
                elif "200%" in tp_mode:
                    target_pt = round(entry_pt * 3.0, 1)
                else:
                    target_pt = st.number_input("自訂停利點數", min_value=entry_pt, max_value=100000.0, value=round(entry_pt * 2.0, 1))
            elif is_stock_fut:
                sl_mode = st.selectbox("3. 停損機制", ["跌幅 -3% 停損 (極短線)", "跌幅 -5% 停損 (標準波段)", "跌幅 -7% 停損", "自訂停損價格"], index=1)
                if "-3%" in sl_mode:
                    stop_pt = round(entry_pt * 0.97, 1)
                elif "-5%" in sl_mode:
                    stop_pt = round(entry_pt * 0.95, 1)
                elif "-7%" in sl_mode:
                    stop_pt = round(entry_pt * 0.93, 1)
                else:
                    stop_pt = st.number_input("自訂停損價格", min_value=0.1, max_value=entry_pt, value=round(entry_pt * 0.95, 1))

                tp_mode = st.selectbox("4. 停利目標", ["波段目標 +10% (盈虧比 2:1)", "波段目標 +15% (盈虧比 3:1)", "波段目標 +20%", "自訂停利價格"], index=0)
                if "+10%" in tp_mode:
                    target_pt = round(entry_pt * 1.10, 1)
                elif "+15%" in tp_mode:
                    target_pt = round(entry_pt * 1.15, 1)
                elif "+20%" in tp_mode:
                    target_pt = round(entry_pt * 1.20, 1)
                else:
                    target_pt = st.number_input("自訂停利價格", min_value=entry_pt, max_value=500000.0, value=round(entry_pt * 1.10, 1))
            else:
                sl_mode = st.selectbox("3. 期貨停損點數", ["停損 40 點", "停損 60 點", "停損 80 點", "自訂停損點數"], index=1)
                sl_val = 60.0 if "60" in sl_mode else (40.0 if "40" in sl_mode else (80.0 if "80" in sl_mode else 50.0))
                stop_pt = round(entry_pt - sl_val, 0)

                tp_mode = st.selectbox("4. 期貨停利目標", ["停利 120 點 (盈虧比 2:1)", "停利 180 點 (盈虧比 3:1)", "自訂停利點數"], index=0)
                tp_val = 120.0 if "120" in tp_mode else (180.0 if "180" in tp_mode else 100.0)
                target_pt = round(entry_pt + tp_val, 0)

        # 數值乘數與契約規格
        if is_option:
            point_multiplier = 50.0
            margin_rate = 1.0  # 買方付全額權利金
            cost_per_lot_twd = entry_pt * point_multiplier
        elif is_stock_fut:
            point_multiplier = 100.0 if is_mini_stock else 2000.0
            margin_rate = 0.135  # 期交所標準個股期保證金成數 13.5% (約 7.4 倍槓桿)
            cost_per_lot_twd = entry_pt * point_multiplier * margin_rate
        else:
            point_multiplier = 10.0 if "TMF" in product_type else (50.0 if "MXF" in product_type else 200.0)
            cost_per_lot_twd = 13000.0 if "TMF" in product_type else (65000.0 if "MXF" in product_type else 260000.0)

        risk_dollar_max = total_account_fund * (max_risk_pct / 100.0)
        loss_per_lot_pts = abs(entry_pt - stop_pt)
        profit_per_lot_pts = abs(target_pt - entry_pt)

        loss_per_lot_twd = loss_per_lot_pts * point_multiplier
        profit_per_lot_twd = profit_per_lot_pts * point_multiplier

        # 依風險與資金控管計算建議口數
        suggested_lots = int(risk_dollar_max // loss_per_lot_twd) if loss_per_lot_twd > 0 else 1
        max_afford_lots = int(total_account_fund // cost_per_lot_twd) if cost_per_lot_twd > 0 else 1
        final_lots = max(1, min(suggested_lots, max_afford_lots))

        total_cost_invested = final_lots * cost_per_lot_twd
        total_loss_at_stop = final_lots * loss_per_lot_twd
        total_profit_at_target = final_lots * profit_per_lot_twd
        rr_ratio = (profit_per_lot_pts / loss_per_lot_pts) if loss_per_lot_pts > 0 else 0.0

        st.markdown("---")
        st.markdown('<div class="sub-header-title">📊 試算結果與下單作戰計畫表</div>', unsafe_allow_html=True)

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.metric("🎯 建議下單口數", f"{final_lots} 口", f"契約規模: {point_multiplier:,.0f} 股/點")
        with res_c2:
            st.metric("💵 應備總權利金 / 保證金", f"NT${total_cost_invested:,.0f}", f"佔總資金 {(total_cost_invested/total_account_fund)*100:.1f}%")
        with res_c3:
            st.metric("🛑 觸發停損最大虧損", f"NT$ -{total_loss_at_stop:,.0f}", f"佔總資金 {(total_loss_at_stop/total_account_fund)*100:.1f}%", delta_color="inverse")
        with res_c4:
            st.metric("🚀 達成目標預期獲利", f"NT$ +{total_profit_at_target:,.0f}", f"報酬率 +{(total_profit_at_target/total_cost_invested)*100:.1f}%")

        plan_df = pd.DataFrame([{
            "交易商品": product_type.split(" - ")[0],
            "契約規格": f"{point_multiplier:,.0f} 股/點 (槓桿 ~7.4x)" if is_stock_fut else f"{point_multiplier:,.0f} 元/點",
            "進場點位/價格": f"NT$ {entry_pt:,.1f}",
            "停損出場點位": f"NT$ {stop_pt:,.1f} (-{loss_per_lot_pts:,.1f})",
            "停利目標點位": f"NT$ {target_pt:,.1f} (+{profit_per_lot_pts:,.1f})",
            "建議口數": f"{final_lots} 口",
            "應備資金/保證金": f"NT$ {total_cost_invested:,.0f}",
            "預估最大虧損": f"NT$ -{total_loss_at_stop:,.0f}",
            "預估達成獲利": f"NT$ +{total_profit_at_target:,.0f}",
            "盈虧報酬比 (R:R)": f"1 : {rr_ratio:.2f}"
        }])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        if is_stock_fut:
            # 計算相較現股節省的交易稅 (現股千分之三 vs 期貨十萬分之二)
            share_equivalent = final_lots * point_multiplier
            tax_stock = (entry_pt * share_equivalent) * 0.003
            tax_future = (entry_pt * share_equivalent) * 0.00002
            tax_saved = max(0, tax_stock - tax_future)
            st.success(f"💡 **個股期貨交易稅優勢**：交易相當於 `{share_equivalent/1000:.1f} 張現股` 規模，相較於直接買賣現股，本筆交易單趟**現省交易稅約 NT$ {tax_saved:,.0f} 元**（個股期貨稅率僅 0.002%，現股為 0.3%）！")


# ------------------------------------------------------------------------------
# 分頁 1: 💼 群益本機極速交易室 (Capital Trading Desk)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="sub-header-title">💼 群益金融 API (Capital API) 本機極速交易戰情室</div>', unsafe_allow_html=True)
    if HAS_CAPITAL and render_capital_trading_desk is not None:
        latest_p_for_desk = float(raw_df["Close"].dropna().iloc[-1]) if not raw_df.empty else 100.0
        render_capital_trading_desk(target_symbol=target_symbol, target_name=target_name, latest_price=latest_p_for_desk)
    else:
        st.info("""
        💡 **群益金融 API (Capital API) 運行環境說明**：
        * 群益 API（SKCOM）專為 **Windows 本機極速交易與海期下單** 設計，需於微軟 Windows 作業系統環境呼叫 COM 元件。
        * 目前此網頁運行於 **雲端環境 (Streamlit Cloud Linux)**，因作業系統架構不包含 Windows COM 運行庫，本機交易室已自動切換為雲端守護模式。
        * **若需使用群益極速下單功能**：請在您的本機 Windows 電腦上下載專案並執行 `run_capital_trading.bat`，即可體驗完整的群益極速下單戰情室！
        """)


# ------------------------------------------------------------------------------
# 分頁 2: ⚡ 永豐即時五檔與台指期行情 (Shioaji Live Quotes & Futures)
# ------------------------------------------------------------------------------
with tabs[2]:


    st.markdown('<div class="sub-header-title">⚡ 永豐金證券 Shioaji 即時盤口與台指期行情</div>', unsafe_allow_html=True)
    
    if sj_mgr.is_connected():
        col_live_stock, col_live_fut = st.columns(2)
        
        # 1. 個股即時快照與五檔
        with col_live_stock:
            st.subheader(f"📊 {target_name} ({target_symbol}) 即時五檔")
            stock_code_clean = target_symbol.split(".")[0]
            snap = sj_mgr.get_stock_snapshot(stock_code_clean)
            
            if "error" in snap:
                st.warning(snap["error"])
            else:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("最新成交價", f"NT${snap['close']:.2f}", f"{snap['change']:+.2f} ({snap['pct_change']:+.2f}%)")
                with m2:
                    st.metric("昨收參考價", f"NT${snap['ref_price']:.2f}")
                with m3:
                    st.metric("累積成交量", f"{snap['volume']:,} 張/股")

                # 五檔圖表
                fig_book = plot_best5_orderbook(snap, tw_style=tw_color_style)
                st.plotly_chart(fig_book, use_container_width=True)

        # 2. 台指期即時行情
        with col_live_fut:
            st.subheader("🎯 台指期貨 (TXF / 近月與夜盤) 即時報價")
            fut_snap = sj_mgr.get_futures_snapshot("TXFR1")
            
            if "error" in fut_snap:
                st.info(f"期貨行情: {fut_snap['error']}")
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.metric("台指期點位", f"{fut_snap['close']:,.0f}", f"{fut_snap['change']:+.0f} ({fut_snap['pct_change']:+.2f}%)")
                with f2:
                    st.metric("昨收點位", f"{fut_snap['ref_price']:,.0f}")
                with f3:
                    st.metric("期貨成交量", f"{fut_snap['volume']:,} 口")

                fig_fut_book = plot_best5_orderbook(fut_snap, tw_style=tw_color_style)
                st.plotly_chart(fig_fut_book, use_container_width=True)
                
                # 計算期現貨價差 (Basis)
                taiex_price = macro_data.get("台股加權指數 (TAIEX)", {}).get("price", np.nan)
                if not np.isnan(taiex_price):
                    basis = fut_snap['close'] - taiex_price
                    basis_type = "正價差 (期貨高於現貨)" if basis > 0 else "逆價差 (期貨低於現貨)"
                    st.info(f"💡 **期現貨基差 (Basis)**: `{basis:+.2f} 點` ({basis_type})")

    else:
        st.info("💡 **尚未連線至永豐金證券 Shioaji API**")
        st.markdown("""
        串接永豐金證券 Shioaji API 後，您可以在此分頁享受：
        - 🚀 **毫秒級即時 Tick / 五檔委託買賣價量盤口 (Best 5 Bids & Asks)**
        - 🎯 **台指期 (TXF / MXF) 日盤與夜盤即時點位與基差計算**
        - 📈 **分時 1 分鐘 / 5 分鐘高頻即時 K 棒分析**
        
        #### 📌 如何設定並連線：
        1. 在專案目錄中建立 `.env` 檔案（或直接在左側側邊欄輸入金鑰）：
        ```bash
        SHIOAJI_API_KEY=您的永豐API_KEY
        SHIOAJI_SECRET_KEY=您的永豐SECRET_KEY
        SHIOAJI_SIMULATION=True
        ```
        2. 點擊左側側邊欄的 **「🚀 連線至永豐 Shioaji」** 按鈕即可即時開通！
        """)


# ------------------------------------------------------------------------------
# 分頁 3: 🦅 多維策略選股獵鷹 (Stock Screener)
# ------------------------------------------------------------------------------
with tabs[3]:

    st.markdown('<div class="sub-header-title">🦅 台股多維度策略選股獵鷹 (Quant Strategy Screener)</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns([0.28, 0.32, 0.25, 0.15])
    with col_s1:
        screener_strat = st.selectbox(
            "1. 選擇選股掃描策略",
            [
                "🦅 多維度強勢訊號精選 (all_signals)",
                "📈 均線多頭排列強勢股 (ma_bull: Close > MA5 > MA20 > MA60)",
                "🚀 帶量突破月線發動股 (vol_breakout: 突破MA20 且 量比>1.3x)",
                "⚡ KD 低檔黃金交叉 (kd_golden: K向上突破D 且 K<70)",
                "🔥 爆量長紅攻擊股 (vol_surge: 成交量>2倍 且 大漲)",
                "💰 高殖利率與價值精選 (high_dividend: 殖利率>4.0%)",
                "🌊 RSI 低檔超賣反彈 (rsi_oversold: RSI<35 低檔反彈)"
            ],
            index=0
        )
    with col_s2:
        fleet_keys = list(THEMATIC_STOCK_GROUPS.keys())
        universe_options = [
            f"👑 全市場核心 260 檔旗艦股池 (全部 {len(POPULAR_TW_STOCKS)} 檔：涵蓋高息ETF/AI/軍工/半導體/中小型)",
        ] + [
            f"{fk} ({len(THEMATIC_STOCK_GROUPS[fk])} 檔)" for fk in fleet_keys
        ] + [
            "🎯 自訂股票代碼清單 (自行輸入任意台股代碼)"
        ]
        universe_choice = st.selectbox(
            "2. 選擇掃描標的池",
            universe_options,
            index=0
        )
    with col_s3:
        grade_filter = st.selectbox(
            "3. 🎯 精準等級過濾 (依勝率分級)",
            [
                "🥇 S + A 級精銳 (強勢發動 >= 60分【推薦】)",
                "🌟 僅顯示 S 級 (頂級共振 >= 75分)",
                "🥈 S + A + B 級 (多頭蓄勢 >= 45分)",
                "全部等級 (S / A / B / C 級全顯示)"
            ],
            index=0
        )
    with col_s4:
        st.write("")
        st.write("")
        run_screener_btn = st.button("🔍 執行精準掃描", use_container_width=True)

    # 決定 active_universe
    if "全市場" in universe_choice or "全系列" in universe_choice:
        active_universe = POPULAR_TW_STOCKS
    elif "自訂" in universe_choice:
        custom_symbols_input = st.text_input(
            "輸入欲掃描的台股代碼 (以逗號分隔)",
            value="2330, 2454, 2449, 2317, 3008, 2603, 0050, 6669, 3017, 3450, 8033, 6781"
        )
        tokens = [x.strip() for x in custom_symbols_input.split(",") if x.strip()]
        active_universe = []
        for tk in tokens:
            _, sym_name = normalize_symbol(tk)
            active_universe.append({"code": tk, "name": sym_name, "sector": "自選標的"})
    else:
        chosen_group_name = universe_choice.split(" (")[0].strip()
        active_universe = THEMATIC_STOCK_GROUPS.get(chosen_group_name, POPULAR_TW_STOCKS)

    raw_key = screener_strat.split("(")[1].split(")")[0].split(":")[0].strip()

    # 內建通用多核心平行選股掃描函式 (含等級排列與量化評分)
    def execute_screening(strategy_key, universe_list, grade_setting):
        import concurrent.futures

        def scan_single_item(item):
            code = item["code"]
            name = item["name"]
            sector = item.get("sector", "台股標的")
            norm_sym, norm_nm = normalize_symbol(code)
            try:
                df = fetch_stock_history(norm_sym, period="3mo")
                if df.empty or len(df) < 20:
                    return None
                df_ind = compute_all_indicators(df)
                latest = df_ind.iloc[-1]
                prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest
                close = float(latest["Close"])
                prev_close = float(prev["Close"])
                pct = ((close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
                vol = int(latest["Volume"])
                vol_ma5 = float(latest.get("Vol_MA5", vol))
                vol_ratio = (vol / vol_ma5) if vol_ma5 > 0 else 1.0
                ma5 = float(latest.get("MA5", 0))
                ma20 = float(latest.get("MA20", 0))
                ma60 = float(latest.get("MA60", 0))
                k_val = float(latest.get("K", 50))
                d_val = float(latest.get("D", 50))
                k_prev = float(prev.get("K", 50))
                d_prev = float(prev.get("D", 50))
                rsi6 = float(latest.get("RSI_6", 50))

                # 多因子評分與標籤
                score = 0
                tags = []

                # 1. 均線與趨勢結構 (最高 30 分)
                if close > ma5 > ma20 > ma60:
                    score += 30
                    tags.append("均線完美多頭")
                elif close > ma5 > ma20:
                    score += 20
                    tags.append("短中期均線多頭")
                elif close > ma20:
                    score += 10
                    tags.append("站上月線MA20")

                # 2. 量價動能 (最高 30 分)
                vol_score = 0
                if vol_ratio >= 2.0 and pct >= 2.5:
                    vol_score += 20
                    tags.append("爆量長紅")
                elif vol_ratio >= 1.3 and pct > 0.5:
                    vol_score += 15
                    tags.append("帶量突破")
                elif vol_ratio >= 1.1:
                    vol_score += 5

                if pct >= 5.0:
                    vol_score += 10
                elif pct >= 2.0:
                    vol_score += 7
                elif pct > 0.0:
                    vol_score += 4
                score += min(30, vol_score)

                # 3. 技術指標共振 KD & RSI (最高 25 分)
                ind_score = 0
                if k_val > d_val and k_prev <= d_prev and k_val < 75:
                    ind_score += 15
                    tags.append("KD黃金交叉")
                elif k_val > d_val and k_val >= 75:
                    ind_score += 10
                    tags.append("KD高檔強勢鈍化")
                elif k_val > d_val:
                    ind_score += 8

                if 55.0 <= rsi6 <= 80.0:
                    ind_score += 10
                    tags.append("RSI黃金攻擊區")
                elif rsi6 < 35.0:
                    ind_score += 8
                    tags.append("RSI超賣築底")
                score += min(25, ind_score)

                # 4. 基本面與殖利率 (最高 15 分)
                norm_div = 0.0
                fund = fetch_stock_fundamentals(norm_sym)
                dy = fund.get("dividend_yield")
                if dy:
                    norm_div = dy / 100.0 if dy > 50 else dy
                    if norm_div >= 5.0:
                        score += 15
                        tags.append(f"高殖利率({norm_div:.1f}%)")
                    elif norm_div >= 4.0:
                        score += 10
                        tags.append(f"穩健殖利率({norm_div:.1f}%)")
                    elif norm_div >= 3.0:
                        score += 5

                # 等級評定
                if score >= 75:
                    grade = "🌟 S級 (頂級共振)"
                    grade_code = "S"
                    action_advice = "🔥 強勢主升段，沿5日線偏多積極佈局"
                elif score >= 60:
                    grade = "🥇 A級 (強勢攻擊)"
                    grade_code = "A"
                    action_advice = "🚀 帶量轉強發動，突破關鍵頸線順勢進場"
                elif score >= 45:
                    grade = "🥈 B級 (轉強蓄勢)"
                    grade_code = "B"
                    action_advice = "👀 整理打底初見轉強，逢低分批佈局"
                else:
                    grade = "🥉 C級 (潛伏觀察)"
                    grade_code = "C"
                    action_advice = "⚠️ 單一訊號符合，量能尚未放大，保守觀望"

                # 策略符合性篩選
                match = False
                if strategy_key == "all_signals" and len(tags) > 0:
                    match = True
                elif strategy_key == "ma_bull" and "均線" in str(tags):
                    match = True
                elif strategy_key == "vol_breakout" and "帶量突破" in tags:
                    match = True
                elif strategy_key == "kd_golden" and "KD" in str(tags):
                    match = True
                elif strategy_key == "vol_surge" and "爆量長紅" in tags:
                    match = True
                elif strategy_key == "rsi_oversold" and "超賣" in str(tags):
                    match = True
                elif strategy_key == "high_dividend" and norm_div >= 4.0:
                    match = True

                # 等級過濾
                if match:
                    if "僅顯示 S 級" in grade_setting and grade_code != "S":
                        return None
                    if "S + A 級" in grade_setting and grade_code not in ["S", "A"]:
                        return None
                    if "S + A + B 級" in grade_setting and grade_code not in ["S", "A", "B"]:
                        return None

                    return {
                        "評級": grade,
                        "綜合評分": f"{score} 分",
                        "股票代碼": code,
                        "股票名稱": name,
                        "所屬產業": sector,
                        "最新收盤價": f"NT${close:,.2f}",
                        "今日漲跌幅 (%)": f"{pct:+.2f}%",
                        "今日量比 (倍)": f"{vol_ratio:.2f}x",
                        "今日成交量": f"{vol:,}",
                        "KD(9,3)": f"K:{k_val:.1f} D:{d_val:.1f}",
                        "RSI(6)": f"{rsi6:.1f}",
                        "觸發訊號標籤": " | ".join(tags) if tags else "多頭符合",
                        "操盤建議": action_advice,
                        "_score": score,
                        "_pct": pct
                    }
            except Exception:
                pass
            return None

        results = []
        max_workers = min(12, len(universe_list)) if universe_list else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(scan_single_item, it) for it in universe_list]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res:
                    results.append(res)

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df.sort_values(by=["_score", "_pct"], ascending=[False, False], inplace=True)
        return res_df

    if run_screener_btn or "last_screener_results" not in st.session_state:
        target_name_clean = universe_choice.split(' (')[0]
        strat_name_clean = screener_strat.split(' (')[0]
        with st.spinner(f"正在平行多核心掃描【{target_name_clean}】符合【{strat_name_clean}】之標的 (共 {len(active_universe)} 檔)..."):
            screen_df = execute_screening(raw_key, active_universe, grade_filter)
            st.session_state["last_screener_results"] = screen_df
            st.session_state["last_screener_strat"] = screener_strat
            st.session_state["last_screener_universe"] = universe_choice
            st.session_state["last_screener_grade"] = grade_filter

    current_screen_df = st.session_state.get("last_screener_results", pd.DataFrame())
    current_strat_name = st.session_state.get("last_screener_strat", screener_strat)
    current_universe_name = st.session_state.get("last_screener_universe", universe_choice)
    current_grade_filter = st.session_state.get("last_screener_grade", grade_filter)

    if not current_screen_df.empty:
        display_screen_df = current_screen_df.drop(columns=[c for c in current_screen_df.columns if c.startswith("_")])
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("策略命中檔數", f"{len(display_screen_df)} 檔")
        with m_col2:
            st.metric("篩選策略", current_strat_name.split(" (")[0])
        with m_col3:
            st.metric("精準等級過濾", current_grade_filter.split(" (")[0])
        with m_col4:
            st.metric("掃描標的池", current_universe_name.split(" (")[0])

        st.dataframe(display_screen_df, use_container_width=True, hide_index=True)

        # 命中標的今日漲跌長條圖
        st.markdown('<div class="sub-header-title">📊 策略命中標的今日漲跌幅排行</div>', unsafe_allow_html=True)
        fig_screen = px.bar(
            current_screen_df,
            x="股票名稱",
            y="_pct_raw",
            color="_pct_raw",
            color_continuous_scale=["#20bf6b", "#747d8c", "#eb3b5a"] if tw_color_style else ["#eb3b5a", "#747d8c", "#20bf6b"],
            text="今日漲跌幅 (%)",
            title=f"【{current_strat_name.split(' (')[0]}】命中標的漲跌動能排行"
        )
        fig_screen.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="標的", yaxis_title="漲跌幅 (%)")
        st.plotly_chart(fig_screen, use_container_width=True)

        # ----------------------------------------------------------------------
        # 💰 交易計畫與進出場價位 / 購買張數試算機 (Stock Position Sizer)
        # ----------------------------------------------------------------------
        st.markdown("---")
        with st.expander("💰 股票交易進出場點位、投入資金與購買張數試算機 (Trade Execution & Position Sizer)", expanded=True):
            st.markdown("針對篩選出的強勢標的，自動計算**建議進場價、停損價、停利價、建議買進張數與預期盈虧報酬比**，嚴格落實風控。")

            stock_candidates = [f"{row['股票代碼']} {row['股票名稱']}" for _, row in current_screen_df.iterrows()]
            calc_c1, calc_c2, calc_c3 = st.columns(3)
            
            with calc_c1:
                selected_candidate = st.selectbox("1. 選擇擬交易之標的", stock_candidates, index=0)
                sel_code = selected_candidate.split(" ")[0]
                sel_row = current_screen_df[current_screen_df["股票代碼"] == sel_code].iloc[0]
                curr_price = float(sel_row["_close_raw"])

                total_stock_capital = st.number_input(
                    "總投資可用資金 (NTD)",
                    min_value=10000,
                    max_value=100000000,
                    value=1000000,
                    step=50000
                )

            with calc_c2:
                risk_tolerance_pct = st.slider(
                    "2. 單筆交易最大風險容忍比例 (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="每次虧損限制在總資金的 2%，確保即使連續多次停損依然保全元氣。"
                )
                entry_stock_price = st.number_input(
                    "預定進場價位 (NTD)",
                    min_value=1.0,
                    max_value=20000.0,
                    value=curr_price,
                    step=1.0
                )

            with calc_c3:
                sl_stock_mode = st.selectbox(
                    "3. 停損價位設定方式",
                    [
                        "固定 -5% 停損 (短線動能)",
                        "固定 -7% 停損 (波段防守)",
                        "跌破 5 日均線 (MA5)",
                        "跌破月線 (MA20)",
                        "自訂停損價格"
                    ],
                    index=0
                )
                if "-5%" in sl_stock_mode:
                    stock_sl_price = round(entry_stock_price * 0.95, 2)
                elif "-7%" in sl_stock_mode:
                    stock_sl_price = round(entry_stock_price * 0.93, 2)
                elif "5 日均線" in sl_stock_mode:
                    stock_sl_price = round(float(sel_row.get("_ma5_raw", entry_stock_price * 0.95)), 2)
                elif "月線" in sl_stock_mode:
                    stock_sl_price = round(float(sel_row.get("_ma20_raw", entry_stock_price * 0.92)), 2)
                else:
                    stock_sl_price = st.number_input("自訂停損價 (NTD)", min_value=1.0, max_value=entry_stock_price, value=round(entry_stock_price * 0.95, 2))

                tp_stock_mode = st.selectbox(
                    "4. 停利目標設定方式",
                    [
                        "盈虧比 2.0 : 1 (獲利為停損 2 倍)",
                        "盈虧比 2.5 : 1 (推薦標準)",
                        "盈虧比 3.0 : 1 (波段大賺)",
                        "波段目標 +15%",
                        "波段目標 +20%",
                        "自訂停利價格"
                    ],
                    index=1
                )
                stock_loss_per_share = max(0.1, entry_stock_price - stock_sl_price)
                if "2.0" in tp_stock_mode:
                    stock_tp_price = round(entry_stock_price + stock_loss_per_share * 2.0, 2)
                elif "2.5" in tp_stock_mode:
                    stock_tp_price = round(entry_stock_price + stock_loss_per_share * 2.5, 2)
                elif "3.0" in tp_stock_mode:
                    stock_tp_price = round(entry_stock_price + stock_loss_per_share * 3.0, 2)
                elif "+15%" in tp_stock_mode:
                    stock_tp_price = round(entry_stock_price * 1.15, 2)
                elif "+20%" in tp_stock_mode:
                    stock_tp_price = round(entry_stock_price * 1.20, 2)
                else:
                    stock_tp_price = st.number_input("自訂停利價 (NTD)", min_value=entry_stock_price, max_value=50000.0, value=round(entry_stock_price * 1.15, 2))

            # 股票部位運算
            stock_profit_per_share = max(0.1, stock_tp_price - entry_stock_price)
            max_risk_dollar = total_stock_capital * (risk_tolerance_pct / 100.0)

            # 依風險計算建議股數
            allowed_shares_by_risk = int(max_risk_dollar // stock_loss_per_share) if stock_loss_per_share > 0 else 100
            # 不能超過資金上限
            max_shares_by_capital = int(total_stock_capital // entry_stock_price) if entry_stock_price > 0 else 100
            final_shares = max(1, min(allowed_shares_by_risk, max_shares_by_capital))

            lots_full = final_shares // 1000
            odd_shares = final_shares % 1000

            capital_required = final_shares * entry_stock_price
            total_loss_at_sl = final_shares * stock_loss_per_share
            total_profit_at_tp = final_shares * stock_profit_per_share
            stock_rr_ratio = (stock_profit_per_share / stock_loss_per_share) if stock_loss_per_share > 0 else 0.0

            res_st1, res_st2, res_st3, res_st4 = st.columns(4)
            with res_st1:
                lots_display = f"{lots_full} 張" + (f" {odd_shares} 股" if odd_shares > 0 else "")
                st.metric("🎯 建議買進數量", lots_display, f"共 {final_shares:,} 股")
            with res_st2:
                st.metric("💵 實際需備資金", f"NT${capital_required:,.0f}", f"佔總資金 {(capital_required/total_stock_capital)*100:.1f}%")
            with res_st3:
                st.metric("🛑 停損最大虧損", f"NT$ -{total_loss_at_sl:,.0f}", f"佔總資金 {(total_loss_at_sl/total_stock_capital)*100:.1f}%", delta_color="inverse")
            with res_st4:
                st.metric("🚀 達成停利預期獲利", f"NT$ +{total_profit_at_tp:,.0f}", f"預估報酬率 +{(total_profit_at_tp/capital_required)*100:.1f}%")

            plan_st_df = pd.DataFrame([{
                "擬買進標的": f"{sel_code} {sel_row['股票名稱']}",
                "進場參考價": f"NT$ {entry_stock_price:,.2f}",
                "停損出場價": f"NT$ {stock_sl_price:,.2f} (-{((entry_stock_price - stock_sl_price)/entry_stock_price)*100:.1f}%)",
                "停利目標價": f"NT$ {stock_tp_price:,.2f} (+{((stock_tp_price - entry_stock_price)/entry_stock_price)*100:.1f}%)",
                "建議購買數量": lots_display,
                "投入本金": f"NT$ {capital_required:,.0f}",
                "最大虧損金額": f"NT$ -{total_loss_at_sl:,.0f}",
                "目標獲利金額": f"NT$ +{total_profit_at_tp:,.0f}",
                "盈虧報酬比 (R:R)": f"1 : {stock_rr_ratio:.2f}"
            }])
            st.dataframe(plan_st_df, use_container_width=True, hide_index=True)
    else:
        st.info("暫無符合該策略條件之標的，請嘗試切換標的池或更換不同選股策略。")


# ------------------------------------------------------------------------------
# 分頁 4: 🌐 國際市場與總經連動 (Global & Macro)
# ------------------------------------------------------------------------------
with tabs[4]:

    # 每日情報速遞生成區塊
    with st.expander("📰 點擊查看【今日台股盤前情報速報 / 盤後戰報】", expanded=True):
        col_btn1, col_btn2 = st.columns([0.2, 0.8])
        report_mode = col_btn1.radio("情報模式", ["🌅 盤前速報", "🌆 盤後戰報"], index=0)
        
        generate_pre_market_briefing = None
        generate_post_market_briefing = None
        try:
            from daily_briefing import generate_pre_market_briefing, generate_post_market_briefing
        except Exception:
            try:
                from utils.daily_briefing import generate_pre_market_briefing, generate_post_market_briefing
            except Exception:
                try:
                    from .agents.skills.taiwan_market_daily_briefing.scripts.daily_briefing import generate_pre_market_briefing, generate_post_market_briefing
                except Exception:
                    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".agents/skills/taiwan-market-daily-briefing/scripts")))
                    try:
                        from daily_briefing import generate_pre_market_briefing, generate_post_market_briefing
                    except Exception:
                        pass

        if not generate_pre_market_briefing:
            def builtin_briefing():
                sox_info = macro_data.get("費城半導體 (SOX)", {})
                tsm_info = macro_data.get("台積電 ADR (TSM)", {})
                fx_info = macro_data.get("美元兌台幣 (USD/TWD)", {})
                return f"""### 🌅 台股早盤即時情報速報 (內建版)
- **費城半導體**：`{sox_info.get('price', 0):,.2f}` ({sox_info.get('pct_change', 0):+.2f}%)
- **台積電 ADR**：`${tsm_info.get('price', 0):,.2f}` ({tsm_info.get('pct_change', 0):+.2f}%)
- **美元兌台幣**：`NT${fx_info.get('price', 0):.3f}` ({fx_info.get('pct_change', 0):+.2f}%)
- **觀盤重點**：國際半導體與 ADR 走勢強勢連動，注意早盤開高震盪與 5 日線防守。"""
            generate_pre_market_briefing = builtin_briefing
            generate_post_market_briefing = builtin_briefing

        briefing_text = generate_pre_market_briefing() if "盤前" in report_mode else generate_post_market_briefing()
        st.markdown(briefing_text)

    st.markdown('<div class="sub-header-title">🎯 台積電 ADR (TSM) vs 現股 (2330) 溢價率分析</div>', unsafe_allow_html=True)
    
    adr_df = get_cached_adr_premium(period=selected_period)
    if not adr_df.empty:
        col_adr1, col_adr2, col_adr3, col_adr4 = st.columns(4)
        latest_tsm_usd = adr_df["TSM_USD"].iloc[-1]
        latest_tw_twd = adr_df["TW_2330"].iloc[-1]
        latest_fx = adr_df["USD_TWD"].iloc[-1]
        latest_adr_twd = adr_df["ADR_TWD_Equiv"].iloc[-1]
        latest_premium = adr_df["Premium_Pct"].iloc[-1]
        latest_spread = adr_df["Spread_TWD"].iloc[-1]

        with col_adr1:
            st.metric("台積電 ADR (USD)", f"${latest_tsm_usd:.2f}")
        with col_adr2:
            st.metric("美元兌台幣匯率", f"NT${latest_fx:.3f}")
        with col_adr3:
            st.metric("ADR折合每股現值 (NTD)", f"NT${latest_adr_twd:.2f}", delta=f"台積現股 NT${latest_tw_twd:.1f}")
        with col_adr4:
            st.metric(
                "ADR 溢價率 (Premium %)",
                f"{latest_premium:+.2f}%",
                delta=f"價差 NT${latest_spread:+.1f}",
                delta_color="inverse" if not tw_color_style else "normal"
            )

        st.plotly_chart(plot_adr_premium(adr_df), use_container_width=True)
        st.info("💡 **ADR 溢價率說明**：1 單位台積電 ADR 等於 5 股台積電普通股。當 ADR 呈現正溢價且持續擴大時，通常代表美股外資買盤強勁，對隔日台股開盤具有顯著正向帶動效果。")
    else:
        st.warning("暫時無法取得台積電 ADR 溢價資料，請稍後再試。")

    st.markdown('<div class="sub-header-title">🌍 國際主要指數與台股連動走勢比較</div>', unsafe_allow_html=True)
    
    macro_series = {}
    for name, info in macro_data.items():
        if not info["history"].empty:
            macro_series[name] = info["history"]

    if macro_series:
        st.plotly_chart(plot_macro_comparison(macro_series), use_container_width=True)

    st.markdown('<div class="sub-header-title">📋 國際市場即時行情與詳細指標數據</div>', unsafe_allow_html=True)
    table_rows = []
    for name, info in macro_data.items():
        table_rows.append({
            "指標名稱": name,
            "代碼": info["symbol"],
            "類別": info["category"],
            "最新報價": f"{info['price']:,.2f}" if not np.isnan(info['price']) else "N/A",
            "當日漲跌": f"{info['change']:+.2f}" if not np.isnan(info['change']) else "N/A",
            "漲跌幅 (%)": f"{info['pct_change']:+.2f}%" if not np.isnan(info['pct_change']) else "N/A",
            "說明": info["desc"],
            "更新日期": info["latest_date"]
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# 分頁 5: 📈 個股技術分析 (Technical Analysis)
# ------------------------------------------------------------------------------
with tabs[5]:

    if df_with_ind.empty:
        st.error(f"無法載入 `{target_symbol}` ({target_name}) 的歷史交易資料，請檢查代碼是否正確。")
    else:
        latest = df_with_ind.iloc[-1]
        prev = df_with_ind.iloc[-2] if len(df_with_ind) > 1 else latest
        change = latest["Close"] - prev["Close"]
        pct_change = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0.0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("最新收盤價", f"{latest['Close']:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
        with c2:
            st.metric("開盤價 (Open)", f"{latest['Open']:,.2f}")
        with c3:
            st.metric("最高價 (High)", f"{latest['High']:,.2f}")
        with c4:
            st.metric("最低價 (Low)", f"{latest['Low']:,.2f}")
        with c5:
            st.metric("成交量 (Volume)", f"{int(latest['Volume']):,}")
        with c6:
            amplitude = ((latest['High'] - latest['Low']) / prev['Close']) * 100 if prev['Close'] != 0 else 0.0
            st.metric("今日振幅", f"{amplitude:.2f}%")

        fig_candle = plot_stock_candlestick(
            df=df_with_ind,
            symbol_name=target_name,
            selected_mas=show_ma,
            show_bbands=show_bbands,
            sub_indicator=sub_indicator,
            tw_style=tw_color_style,
            chart_type=chart_type
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        st.markdown('<div class="sub-header-title">🔍 技術面指標快篩診斷</div>', unsafe_allow_html=True)
        diag_cols = st.columns(4)

        ma5 = latest.get("MA5", np.nan)
        ma20 = latest.get("MA20", np.nan)
        ma60 = latest.get("MA60", np.nan)
        ma_status = "多頭排列 (MA5 > MA20 > MA60)" if (ma5 > ma20 > ma60) else ("空頭排列 (MA5 < MA20 < MA60)" if (ma5 < ma20 < ma60) else "均線糾結/震盪整理")
        with diag_cols[0]:
            st.markdown(f"**均線趨勢狀態**<br>`{ma_status}`", unsafe_allow_html=True)

        k_val = latest.get("K", np.nan)
        d_val = latest.get("D", np.nan)
        kd_status = "黃金交叉 (K向上突破D)" if (k_val > d_val and prev.get("K", 0) <= prev.get("D", 0)) else ("死亡交叉 (K向下跌破D)" if (k_val < d_val and prev.get("K", 0) >= prev.get("D", 0)) else ("超買區 (>80)" if k_val > 80 else ("超賣區 (<20)" if k_val < 20 else "中性區域")))
        with diag_cols[1]:
            st.markdown(f"**KD(9,3) 狀態**<br>K: `{k_val:.1f}` | D: `{d_val:.1f}`<br>`{kd_status}`", unsafe_allow_html=True)

        rsi6 = latest.get("RSI_6", np.nan)
        rsi_status = "超買警戒區 (>75)" if rsi6 > 75 else ("超賣反彈區 (<25)" if rsi6 < 25 else "正常波動區")
        with diag_cols[2]:
            st.markdown(f"**RSI(6) 強弱**<br>數值: `{rsi6:.1f}`<br>`{rsi_status}`", unsafe_allow_html=True)

        macd_dif = latest.get("MACD_DIF", np.nan)
        macd_status = "柱狀圖轉正 (紅柱擴大)" if latest.get("MACD_Hist", 0) > 0 else "柱狀圖為負 (綠柱擴大)"
        with diag_cols[3]:
            st.markdown(f"**MACD 狀態**<br>DIF: `{macd_dif:.2f}` | OSC: `{latest.get('MACD_Hist', 0):.2f}`<br>`{macd_status}`", unsafe_allow_html=True)

        with st.expander("📥 檢視與下載歷史原始報價資料 (CSV)"):
            st.dataframe(df_with_ind.sort_index(ascending=False), use_container_width=True)
            csv_data = df_with_ind.to_csv().encode('utf-8-sig')
            st.download_button(
                label="下載歷史數據 CSV",
                data=csv_data,
                file_name=f"{target_symbol}_history.csv",
                mime="text/csv"
            )


# ------------------------------------------------------------------------------
# 分頁 6: 🏢 基本面與估值河流圖 (Fundamentals & Valuation)
# ------------------------------------------------------------------------------
with tabs[6]:

    # 財報診斷報告生成
    with st.expander(f"📑 點擊產出【{target_name} 完整財報體質診斷與多模型目標價估算報告】", expanded=False):
        fin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agents/skills/taiwan-financial-report-analyzer/scripts"))
        generate_financial_report = None
        try:
            from financial_analyzer import generate_financial_report
        except Exception:
            try:
                from utils.financial_analyzer import generate_financial_report
            except Exception:
                try:
                    fin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agents/skills/taiwan-financial-report-analyzer/scripts"))
                    if fin_path not in sys.path:
                        sys.path.insert(0, fin_path)
                    from financial_analyzer import generate_financial_report
                except Exception:
                    pass

        if generate_financial_report:
            st.markdown(generate_financial_report(target_symbol))
        else:
            funds = get_cached_fundamentals(target_symbol)
            st.markdown(f"""### 📑 {target_name} ({target_symbol}) 快速估值診斷
- **本益比 (P/E)**：`{funds.get('pe_ratio', 'N/A')}` 倍
- **股價淨值比 (P/B)**：`{funds.get('pb_ratio', 'N/A')}` 倍
- **每股盈餘 (EPS)**：`NT${funds.get('eps', 'N/A')}`
- **現金殖利率**：`{funds.get('dividend_yield', 'N/A')}%`
- **股東權益報酬率 (ROE)**：`{funds.get('roe', 'N/A')}%`""")

    st.markdown(f'<div class="sub-header-title">🏢 {target_name} ({target_symbol}) 財務體質與估值指標</div>', unsafe_allow_html=True)
    
    fundamentals = get_cached_fundamentals(target_symbol)
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        pe = fundamentals.get("pe_ratio")
        st.metric("本益比 (Trailing P/E)", f"{pe:.2f} 倍" if pe else "N/A")
    with f2:
        pb = fundamentals.get("pb_ratio")
        st.metric("股價淨值比 (P/B)", f"{pb:.2f} 倍" if pb else "N/A")
    with f3:
        div_y = fundamentals.get("dividend_yield")
        st.metric("現金殖利率 (Dividend Yield)", f"{div_y:.2f}%" if div_y else "N/A")
    with f4:
        eps = fundamentals.get("eps")
        st.metric("每股盈餘 (EPS)", f"NT${eps:.2f}" if eps else "N/A")

    f5, f6, f7, f8 = st.columns(4)
    with f5:
        mcap = fundamentals.get("market_cap")
        st.metric("總市值 (Market Cap)", f"NT${mcap/1e8:,.1f} 億" if mcap else "N/A")
    with f6:
        roe = fundamentals.get("roe")
        st.metric("股東權益報酬率 (ROE)", f"{roe:.2f}%" if roe else "N/A")
    with f7:
        margin = fundamentals.get("profit_margin")
        st.metric("淨利率 (Profit Margin)", f"{margin:.2f}%" if margin else "N/A")
    with f8:
        rev_growth = fundamentals.get("revenue_growth")
        st.metric("營收年增率 (YoY)", f"{rev_growth:+.2f}%" if rev_growth else "N/A")

    st.markdown('<div class="sub-header-title">🌊 本益比估值河流圖 (P/E River Valuation)</div>', unsafe_allow_html=True)
    if not raw_df.empty:
        pe_river_df = calculate_pe_bands(raw_df, eps=eps)
        fig_pe = plot_pe_river(pe_river_df, target_name)
        st.plotly_chart(fig_pe, use_container_width=True)

    st.markdown('<div class="sub-header-title">💰 歷年股利分派歷史紀錄</div>', unsafe_allow_html=True)
    div_series = fundamentals.get("dividends")
    if div_series is not None and not div_series.empty:
        div_df = pd.DataFrame({
            "除息日期": div_series.index.strftime("%Y-%m-%d"),
            "每股現金股利 (TWD)": div_series.values
        }).sort_values(by="除息日期", ascending=False)
        
        col_div_chart, col_div_tbl = st.columns([0.65, 0.35])
        with col_div_chart:
            fig_div = px.bar(
                div_df.sort_values(by="除息日期"),
                x="除息日期",
                y="每股現金股利 (TWD)",
                title=f"{target_name} 歷年配息趨勢 (NTD)",
                color_discrete_sequence=["#3867d6"]
            )
            fig_div.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_div, use_container_width=True)
        with col_div_tbl:
            st.dataframe(div_df, height=350, use_container_width=True, hide_index=True)
    else:
        st.info("暫無此標的之歷史股利分派明細。")


# ------------------------------------------------------------------------------
# 分頁 7: 👥 籌碼與法人動態 (Institutional & Chips)
# ------------------------------------------------------------------------------
with tabs[7]:

    st.markdown(f'<div class="sub-header-title">👥 {target_name} ({target_symbol}) 籌碼面與成交結構分析</div>', unsafe_allow_html=True)
    
    if not df_with_ind.empty:
        vol_df = df_with_ind.tail(60).copy()
        
        col_c1, col_c2, col_c3 = st.columns(3)
        avg_vol_5 = vol_df["Volume"].tail(5).mean()
        avg_vol_20 = vol_df["Volume"].tail(20).mean()
        vol_ratio = (latest["Volume"] / avg_vol_5) if avg_vol_5 > 0 else 1.0

        with col_c1:
            st.metric("5日均量 (張數/股)", f"{int(avg_vol_5):,}")
        with col_c2:
            st.metric("20日均量 (張數/股)", f"{int(avg_vol_20):,}")
        with col_c3:
            st.metric("今日量比 (今日量 / 5日均量)", f"{vol_ratio:.2f} 倍")

        fig_chip = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=["收盤價與移動平均線", "每日成交量 (Volume) 與 5/20日均量線"])
        fig_chip.add_trace(go.Scatter(x=vol_df.index, y=vol_df["Close"], mode="lines", name="收盤價", line=dict(color="#2f3542", width=2)), row=1, col=1)
        if "MA5" in vol_df.columns:
            fig_chip.add_trace(go.Scatter(x=vol_df.index, y=vol_df["MA5"], mode="lines", name="MA5", line=dict(color="#ff9f43", width=1)), row=1, col=1)
        if "MA20" in vol_df.columns:
            fig_chip.add_trace(go.Scatter(x=vol_df.index, y=vol_df["MA20"], mode="lines", name="MA20", line=dict(color="#ee5253", width=1.2)), row=1, col=1)

        chip_bar_colors = np.where(vol_df["Close"] >= vol_df["Open"], "#ff4757", "#2ed573")
        fig_chip.add_trace(go.Bar(x=vol_df.index, y=vol_df["Volume"], name="成交量", marker_color=chip_bar_colors, opacity=0.7), row=2, col=1)
        if "Vol_MA5" in vol_df.columns:
            fig_chip.add_trace(go.Scatter(x=vol_df.index, y=vol_df["Vol_MA5"], mode="lines", name="5日量均", line=dict(color="#ffa502", width=1.5)), row=2, col=1)
        if "Vol_MA20" in vol_df.columns:
            fig_chip.add_trace(go.Scatter(x=vol_df.index, y=vol_df["Vol_MA20"], mode="lines", name="20日量均", line=dict(color="#70a1ff", width=1.5)), row=2, col=1)

        fig_chip.update_layout(height=520, hovermode="x unified", margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig_chip, use_container_width=True)

        st.info("📌 **籌碼觀察心法**：當股價帶量突破均線糾結區（量比 > 1.5倍 且 紅K棒），通常為主力或外資法人發動攻勢之訊號；高檔量縮價不跌則為強勢整理。")
    else:
        st.warning("無籌碼分析資料。")


# ------------------------------------------------------------------------------
# 分頁 8: ⚔️ 多股報酬比較 (Stock Comparison)
# ------------------------------------------------------------------------------
with tabs[8]:

    st.markdown('<div class="sub-header-title">⚔️ 台股多檔標的與國際指數累積報酬率對照</div>', unsafe_allow_html=True)
    
    default_compare_stocks = ["2330.TW", "2454.TW", "2317.TW", "0050.TW", "TSM", "^SOX"]
    compare_input = st.text_input(
        "輸入欲比較的股票或指數代碼 (用逗號隔開)",
        value=", ".join(default_compare_stocks)
    )

    compare_tickers = [s.strip() for s in compare_input.split(",") if s.strip()]
    if compare_tickers:
        comp_dfs = {}
        for ticker_raw in compare_tickers:
            norm_sym, norm_name = normalize_symbol(ticker_raw)
            h = fetch_stock_history(norm_sym, period=selected_period)
            if not h.empty:
                comp_dfs[norm_name] = h["Close"]

        if comp_dfs:
            st.plotly_chart(plot_macro_comparison(comp_dfs), use_container_width=True)

            st.markdown('<div class="sub-header-title">🔗 標的每日報酬相關係數矩陣 (Correlation Heatmap)</div>', unsafe_allow_html=True)
            aligned_df = pd.DataFrame(comp_dfs).dropna()
            if not aligned_df.empty and len(aligned_df.columns) > 1:
                ret_df = aligned_df.pct_change().dropna()
                corr_matrix = ret_df.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="資產每日報酬率相關係數 (1.0 = 完全正相關, -1.0 = 完全負相關)"
                )
                fig_corr.update_layout(height=450, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("查無比較標的資料。")


# ------------------------------------------------------------------------------
# 分頁 9: 🧪 量化策略回測實驗室 (Strategy Backtesting)
# ------------------------------------------------------------------------------
with tabs[9]:

    st.markdown(f'<div class="sub-header-title">🧪 {target_name} 量化交易策略回測實驗室</div>', unsafe_allow_html=True)

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        strategy_choice = st.selectbox(
            "選擇回測交易策略",
            [
                "雙均線黃金/死亡交叉 (Dual MA Cross)",
                "RSI 超賣反彈與超買停利 (RSI Mean Reversion)",
                "MACD 柱狀與快慢線交叉 (MACD Cross)",
                "布林通道破底翻逆勢策略 (Bollinger Breakout)"
            ]
        )
    with b_col2:
        init_capital = st.number_input("初始回測本金 (TWD)", min_value=10000, max_value=10000000, value=100000, step=10000)
    with b_col3:
        strat_key = "Dual_MA"
        strat_kwargs = {}
        if "Dual MA" in strategy_choice:
            strat_key = "Dual_MA"
            fast_ma = st.slider("快線週期 (Fast MA)", 3, 20, 5)
            slow_ma = st.slider("慢線週期 (Slow MA)", 10, 120, 20)
            strat_kwargs = {"fast_period": fast_ma, "slow_period": slow_ma}
        elif "RSI" in strategy_choice:
            strat_key = "RSI"
            rsi_len = st.slider("RSI 週期", 5, 30, 14)
            buy_th = st.slider("超賣買進閥值", 10, 40, 30)
            sell_th = st.slider("超買賣出閥值", 60, 90, 70)
            strat_kwargs = {"rsi_period": rsi_len, "buy_threshold": buy_th, "sell_threshold": sell_th}
        elif "MACD" in strategy_choice:
            strat_key = "MACD"
        elif "Bollinger" in strategy_choice:
            strat_key = "Bollinger"

    if not df_with_ind.empty:
        metrics, equity_df, trades = run_strategy_backtest(
            df=df_with_ind,
            strategy=strat_key,
            initial_capital=init_capital,
            **strat_kwargs
        )

        if metrics:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("策略總報酬率", f"{metrics['total_return_pct']:+.2f}%", delta=f"買入持有: {metrics['buy_hold_return_pct']:+.2f}%")
            with m2:
                st.metric("年化報酬率 (CAGR)", f"{metrics['cagr_pct']:.2f}%")
            with m3:
                st.metric("最大回撤 (MDD)", f"{metrics['max_drawdown_pct']:.2f}%", delta_color="inverse")
            with m4:
                st.metric("夏普比率 (Sharpe)", f"{metrics['sharpe_ratio']:.2f}")
            with m5:
                st.metric("交易勝率 (Win Rate)", f"{metrics['win_rate_pct']:.1f}%", f"{metrics['total_trades']} 筆交易")
            with m6:
                st.metric("獲利因子 (Profit Factor)", f"{metrics['profit_factor']:.2f}")

            fig_bt = plot_backtest_chart(equity_df, trades, df_with_ind, strategy_choice.split(" (")[0])
            st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown('<div class="sub-header-title">📝 歷史交易明細紀錄表 (Trade Log)</div>', unsafe_allow_html=True)
            if trades:
                trade_records = []
                for t in trades:
                    trade_records.append({
                        "進場日期": t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"]),
                        "進場價格": f"{t['entry_price']:.2f}",
                        "出場日期": t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"]),
                        "出場價格": f"{t['exit_price']:.2f}",
                        "單筆報酬 (%)": f"{t['return_pct']:+.2f}%",
                        "持股天數": f"{t['holding_days']} 天",
                        "勝負結果": "🟢 獲利" if t["is_win"] else "🔴 虧損"
                    })
                st.dataframe(pd.DataFrame(trade_records), use_container_width=True, hide_index=True)
            else:
                st.info("在此期間內策略未觸發任何完整進出場交易。")
    else:
        st.warning("資料不足，無法執行回測。")

# 頁尾
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.85rem;'>"
    "台灣股市與國際連動分析儀表板 | 資料來源: 永豐金證券 Shioaji / yfinance / TWSE | 本工具僅供量化分析與學術研究參考，不構成任何投資買賣建議。"
    "</div>",
    unsafe_allow_html=True
)
