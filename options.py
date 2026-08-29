"""
Options Calculation & Strategy Module (選擇權量化定價、Greeks與策略模組)
Includes Black-Scholes Formula, Greeks, Strategy Payoff, TXO Weekly Status, and Decision Matrix
"""

import math
import numpy as np
import pandas as pd
import datetime
from scipy.stats import norm


def black_scholes_pricing_and_greeks(
    S: float,
    K: float,
    T_days: float,
    r: float = 0.015,
    sigma: float = 0.20,
    option_type: str = "call"
) -> dict:
    """
    計算 Black-Scholes 選擇權理論價格與希臘字母 (Greeks)
    :param S: 標的現價
    :param K: 履約價
    :param T_days: 到期天數 (Days to Expiration, DTE)
    :param r: 無風險利率 (例如 0.015 代表 1.5%)
    :param sigma: 隱含波動率 (例如 0.20 代表 20%)
    :param option_type: "call" 或 "put"
    :return: dict 包含 price, delta, gamma, theta_day, vega_1pct, rho_1pct
    """
    if S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta_day": 0.0, "vega_1pct": 0.0, "rho_1pct": 0.0}

    T = max(T_days / 365.0, 1e-5)
    sigma = max(sigma, 1e-4)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega_1pct = S * math.sqrt(T) * pdf_d1 * 0.01

    if option_type.lower() == "call":
        price = S * cdf_d1 - K * math.exp(-r * T) * cdf_d2
        delta = cdf_d1
        theta_year = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cdf_d2
        theta_day = theta_year / 365.0
        rho_1pct = K * T * math.exp(-r * T) * cdf_d2 * 0.01
    else:
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        price = K * math.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1
        delta = cdf_d1 - 1.0
        theta_year = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cdf_neg_d2
        theta_day = theta_year / 365.0
        rho_1pct = -K * T * math.exp(-r * T) * cdf_neg_d2 * 0.01

    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": gamma,
        "theta_day": theta_day,
        "vega_1pct": vega_1pct,
        "rho_1pct": rho_1pct,
        "d1": d1,
        "d2": d2
    }


def get_weekly_txo_status(current_dt: datetime.datetime = None) -> dict:
    """
    分析台指週選擇權 (TXO) 當前所處的生命週期與實戰建議
    """
    if current_dt is None:
        current_dt = datetime.datetime.now()

    weekday = current_dt.weekday()  # 0: Mon, 1: Tue, 2: Wed, ..., 6: Sun
    hour = current_dt.hour
    
    weekday_names = ["週一 (Mon)", "週二 (Tue)", "週三 (Wed)", "週四 (Thu)", "週五 (Fri)", "週六 (Sat)", "週日 (Sun)"]
    day_name = weekday_names[weekday]

    # 計算距離下一個週三結算的天數
    if weekday < 2:  # Mon, Tue
        days_to_wed = 2 - weekday
    elif weekday == 2:  # Wed
        if hour < 13:
            days_to_wed = 0
        else:
            days_to_wed = 7
    else:  # Thu, Fri, Sat, Sun
        days_to_wed = 7 - (weekday - 2)

    # 判定生命週期與特色
    if weekday == 2 and hour < 14:
        stage = "🔥 結算日當天 (0DTE 決戰日)"
        badge = "badge-danger"
        theta_desc = "極端快速崩跌（價外合約迅速歸零）"
        gamma_desc = "極端放大（破位即噴發數倍至數十倍）"
        strategy_tip = "重點關注早盤 09:00~10:30 開盤突破買方、或 0DTE 雙買樂透策略。11:30 後價外迅速收乾，嚴禁重倉盲目追價外。"
    elif weekday == 2 and hour >= 14:
        stage = "🌱 新合約開盤期 (次週 TXO 夜盤首發)"
        badge = "badge-info"
        theta_desc = "時間價值最厚"
        gamma_desc = "低"
        strategy_tip = "夜盤 15:00 開盤，時間價值飽滿，賣方可建立寬幅價外信用價差 (Credit Spread) 佈局一整週收租。"
    elif weekday in [3, 4]:  # Thu, Fri
        stage = "🚀 趨勢佈局與發酵期 (前期)"
        badge = "badge-primary"
        theta_desc = "穩健衰減"
        gamma_desc = "適中"
        strategy_tip = "時間價值尚厚，方向順勢者可佈局波段 Long Call / Long Put 或牛市/熊市價差，享受趨勢延續。"
    elif weekday in [0, 1]:  # Mon, Tue
        stage = "⏳ Theta 時間加速期 (中後期)"
        badge = "badge-warning"
        theta_desc = "加速流逝 (橫盤即大幅折損)"
        gamma_desc = "快速升高"
        strategy_tip = "若做買方需嚴格設立時間停損（橫盤超過 2 天未發動立刻撤退）；賣方獲利若達 60%~75% 建議提早停利入袋。"
    else:  # Weekend
        stage = "🏖️ 週末休市沉澱期"
        badge = "badge-secondary"
        theta_desc = "週末時間價值持續消耗"
        gamma_desc = "無"
        strategy_tip = "檢視國際市場（美股費半、TSM ADR）最新走勢，為週一開盤做好方向定位與風險預備。"

    return {
        "weekday": weekday,
        "day_name": day_name,
        "days_to_wed": days_to_wed,
        "stage": stage,
        "theta_desc": theta_desc,
        "gamma_desc": gamma_desc,
        "strategy_tip": strategy_tip,
        "current_time_str": current_dt.strftime("%Y-%m-%d %H:%M")
    }


def generate_txo_strike_ladder(
    current_price: float,
    step: int = 100,
    num_steps: int = 5,
    iv: float = 0.18,
    dte: int = 5
) -> pd.DataFrame:
    """
    產生台指選擇權 (TXO) 履約價鏈與 Greeks 試算階梯表
    """
    atm_strike = round(current_price / step) * step
    strikes = [atm_strike + i * step for i in range(-num_steps, num_steps + 1)]

    rows = []
    for k in strikes:
        moneyness = "價平 (ATM)" if k == atm_strike else ("價外 (OTM)" if k > current_price else "價內 (ITM)")
        put_moneyness = "價平 (ATM)" if k == atm_strike else ("價外 (OTM)" if k < current_price else "價內 (ITM)")
        
        call_res = black_scholes_pricing_and_greeks(current_price, k, dte, sigma=iv, option_type="call")
        put_res = black_scholes_pricing_and_greeks(current_price, k, dte, sigma=iv, option_type="put")

        rows.append({
            "Call Delta": round(call_res["delta"], 2),
            "Call 估計權利金 (點)": round(call_res["price"], 1),
            "Call Theta/日": round(call_res["theta_day"], 1),
            "履約價 (Strike)": k,
            "狀態位置": f"Call {moneyness} / Put {put_moneyness}",
            "Put 估計權利金 (點)": round(put_res["price"], 1),
            "Put Delta": round(put_res["delta"], 2),
            "Put Theta/日": round(put_res["theta_day"], 1)
        })

    return pd.DataFrame(rows)


def calculate_strategy_payoff(
    strategy: str,
    spot_price: float,
    params: dict
) -> tuple:
    """
    計算選擇權各主流策略在到期時的損益曲線 (Payoff Curve)
    :return: (prices_array, payoffs_array, max_profit, max_loss, breakevens, summary_text)
    """
    s_min = spot_price * 0.85
    s_max = spot_price * 1.15
    prices = np.linspace(s_min, s_max, 300)
    payoffs = np.zeros_like(prices)

    max_profit_str = "無限"
    max_loss_str = "全部權利金"
    breakevens = []
    summary_text = ""

    if strategy == "Long Call (買進買權)":
        k = params.get("k1", spot_price)
        prem = params.get("prem1", 50.0)
        payoffs = np.maximum(prices - k, 0) - prem
        bep = k + prem
        breakevens = [round(bep, 2)]
        max_profit_str = "無限 (股價漲越高賺越多)"
        max_loss_str = f"-{prem:.1f} 點 (已付之全部權利金)"
        summary_text = f"【看大漲】損益兩平點為 **{bep:,.1f}**。只要到期高於此點即開始獲利，最大風險鎖定在權利金 **{prem:.1f} 點**。"

    elif strategy == "Long Put (買進賣權)":
        k = params.get("k1", spot_price)
        prem = params.get("prem1", 50.0)
        payoffs = np.maximum(k - prices, 0) - prem
        bep = k - prem
        breakevens = [round(bep, 2)]
        max_profit_str = f"{bep:.1f} 點 (標的跌至 0)"
        max_loss_str = f"-{prem:.1f} 點 (已付之全部權利金)"
        summary_text = f"【看大跌】損益兩平點為 **{bep:,.1f}**。只要到期低於此點即開始獲利，最大風險鎖定在權利金 **{prem:.1f} 點**。"

    elif strategy == "Bull Call Spread (買權牛市價差)":
        k1 = params.get("k1", spot_price)       # Buy lower strike Call
        k2 = params.get("k2", spot_price + 200) # Sell higher strike Call
        prem1 = params.get("prem1", 80.0)
        prem2 = params.get("prem2", 30.0)
        net_cost = prem1 - prem2

        payoff_long = np.maximum(prices - k1, 0) - prem1
        payoff_short = -(np.maximum(prices - k2, 0) - prem2)
        payoffs = payoff_long + payoff_short

        bep = k1 + net_cost
        breakevens = [round(bep, 2)]
        max_profit = (k2 - k1) - net_cost
        max_profit_str = f"+{max_profit:.1f} 點"
        max_loss_str = f"-{net_cost:.1f} 點 (淨付出權利金)"
        summary_text = f"【溫和看多】買進低履約價 Call + 賣出高履約價 Call。損益兩平點 **{bep:,.1f}**，最大獲利 **+{max_profit:.1f} 點**，最大虧損 **-{net_cost:.1f} 點**。"

    elif strategy == "Bear Put Spread (賣權熊市價差)":
        k1 = params.get("k1", spot_price)       # Buy higher strike Put
        k2 = params.get("k2", spot_price - 200) # Sell lower strike Put
        prem1 = params.get("prem1", 80.0)
        prem2 = params.get("prem2", 30.0)
        net_cost = prem1 - prem2

        payoff_long = np.maximum(k1 - prices, 0) - prem1
        payoff_short = -(np.maximum(k2 - prices, 0) - prem2)
        payoffs = payoff_long + payoff_short

        bep = k1 - net_cost
        breakevens = [round(bep, 2)]
        max_profit = (k1 - k2) - net_cost
        max_profit_str = f"+{max_profit:.1f} 點"
        max_loss_str = f"-{net_cost:.1f} 點 (淨付出權利金)"
        summary_text = f"【溫和看空】買進高履約價 Put + 賣出低履約價 Put。損益兩平點 **{bep:,.1f}**，最大獲利 **+{max_profit:.1f} 點**，最大虧損 **-{net_cost:.1f} 點**。"

    elif strategy == "Bull Put Spread (賣權牛市信用價差)":
        # Sell higher strike Put, Buy lower strike Put
        k_sell = params.get("k1", spot_price - 100)
        k_buy = params.get("k2", spot_price - 300)
        prem_sell = params.get("prem1", 60.0)
        prem_buy = params.get("prem2", 20.0)
        net_credit = prem_sell - prem_buy

        payoff_short = -(np.maximum(k_sell - prices, 0) - prem_sell)
        payoff_long = np.maximum(k_buy - prices, 0) - prem_buy
        payoffs = payoff_short + payoff_long

        bep = k_sell - net_credit
        breakevens = [round(bep, 2)]
        max_loss = (k_sell - k_buy) - net_credit
        max_profit_str = f"+{net_credit:.1f} 點 (淨收權利金)"
        max_loss_str = f"-{max_loss:.1f} 點"
        summary_text = f"【賣方看多/不跌】賣出 OTM Put + 買進更深 OTM Put 保護。最大獲利為淨收之 **+{net_credit:.1f} 點**，最大虧損鎖定在 **-{max_loss:.1f} 點**，損益兩平點為 **{bep:,.1f}**。"

    elif strategy == "Iron Condor (鐵鷹價差/雙向收租)":
        put_sell = params.get("put_sell", spot_price - 200)
        put_buy = params.get("put_buy", spot_price - 400)
        call_sell = params.get("call_sell", spot_price + 200)
        call_buy = params.get("call_buy", spot_price + 400)
        
        prem_put_s = params.get("prem_put_s", 40.0)
        prem_put_b = params.get("prem_put_b", 15.0)
        prem_call_s = params.get("prem_call_s", 40.0)
        prem_call_b = params.get("prem_call_b", 15.0)

        net_credit = (prem_put_s - prem_put_b) + (prem_call_s - prem_call_b)
        
        payoff_puts = -(np.maximum(put_sell - prices, 0) - prem_put_s) + (np.maximum(put_buy - prices, 0) - prem_put_b)
        payoff_calls = -(np.maximum(prices - call_sell, 0) - prem_call_s) + (np.maximum(prices - call_buy, 0) - prem_call_b)
        payoffs = payoff_puts + payoff_calls

        bep_lower = put_sell - net_credit
        bep_upper = call_sell + net_credit
        breakevens = [round(bep_lower, 2), round(bep_upper, 2)]

        wing_width = (put_sell - put_buy)
        max_loss = wing_width - net_credit
        max_profit_str = f"+{net_credit:.1f} 點 (全部權利金)"
        max_loss_str = f"-{max_loss:.1f} 點"
        summary_text = f"【盤整雙向收租】只要到期標的落在 **{put_sell:,.0f} ~ {call_sell:,.0f}** 區間內，即可全額賺取 **+{net_credit:.1f} 點**；兩端均有買方保護，最大風險僅 **-{max_loss:.1f} 點**。"

    elif strategy == "Long Straddle (買進跨式/結算日雙買)":
        k = params.get("k1", spot_price)
        prem_c = params.get("prem1", 45.0)
        prem_p = params.get("prem2", 45.0)
        total_cost = prem_c + prem_p

        payoff_c = np.maximum(prices - k, 0) - prem_c
        payoff_p = np.maximum(k - prices, 0) - prem_p
        payoffs = payoff_c + payoff_p

        bep_lower = k - total_cost
        bep_upper = k + total_cost
        breakevens = [round(bep_lower, 2), round(bep_upper, 2)]
        max_profit_str = "無限 (單邊大暴衝即翻倍)"
        max_loss_str = f"-{total_cost:.1f} 點 (兩邊合約總成本)"
        summary_text = f"【押注重大事件/單邊大暴走】同時買進價平 Call 與 Put。只要大盤波幅大於 **±{total_cost:.1f} 點**（低於 {bep_lower:,.1f} 或高於 {bep_upper:,.1f}）即開始獲利！"

    return prices, payoffs, max_profit_str, max_loss_str, breakevens, summary_text


def recommend_options_play(
    direction_view: str,
    iv_environment: str,
    horizon: str
) -> dict:
    """
    智能選擇權策略推薦引導
    """
    if "大漲" in direction_view:
        if "低" in iv_environment:
            return {
                "strategy": "Long Call (單邊買進買權)",
                "type": "買方 (Buyer)",
                "strike_advice": "挑選 價平 (ATM Delta 0.50) 或 價外 1 檔 (Delta 0.40~0.45)",
                "reason": "低 IV 環境下權利金便宜，適合透過買方享受 Gamma 爆炸性槓桿與 Vega 膨脹雙重獲利。",
                "entry_trigger": "帶量突破關鍵壓力、布林通道開口發散、均線多頭發散。",
                "stop_loss": "權利金虧損 35%~50% 果斷停損；或跌破 5MA / 突破紅 K 低點。",
                "take_profit": "獲利達 +100% (翻倍) 先平倉 50% 本金，剩餘移動停利。"
            }
        else:
            return {
                "strategy": "Bull Call Spread (買權牛市價差) 或 Bull Put Spread (賣權牛市信用價差)",
                "type": "價差 / 混合",
                "strike_advice": "買進 ATM Call + 賣出 OTM Call (Delta 0.25)；或賣出 OTM Put (Delta 0.20) + 買更深保護",
                "reason": "高 IV 環境下單邊買方權利金過貴，透過垂直價差賣出高估的遠端合約，抵消高 IV 與 Theta 耗損。",
                "entry_trigger": "回測短期均線有守、沿上升趨勢線溫和走高。",
                "stop_loss": "單邊虧損達淨權利金 1.5 倍停損。",
                "take_profit": "達到最大利潤之 75%~85% 提前平倉。"
            }

    elif "大跌" in direction_view:
        if "低" in iv_environment:
            return {
                "strategy": "Long Put (單邊買進賣權)",
                "type": "買方 (Buyer)",
                "strike_advice": "挑選 價平 (ATM Delta -0.50) 或 價外 1 檔 (Delta -0.40~-0.45)",
                "reason": "低 IV 下直接買進 Put 享有高盈虧比，一旦行情跳空重挫，IV 飆升與 Delta 暴增將帶來倍數獲利。",
                "entry_trigger": "帶量跌破頸線/月線、布林通道向下開口、空頭吞噬。",
                "stop_loss": "權利金虧損 40% 停損；站回跌破黑 K 高點立即平倉。",
                "take_profit": "翻倍出本金，剩餘部位沿 5MA 順勢移動停利。"
            }
        else:
            return {
                "strategy": "Bear Put Spread (賣權熊市價差) / Bear Call Spread",
                "type": "價差 / 混合",
                "strike_advice": "買進 ATM Put + 賣出 OTM Put (Delta -0.25)",
                "reason": "高 IV 下可藉由賣出更深價外 Put 來補貼買方成本，鎖定風險並防禦 IV Crush 波動率崩跌。",
                "entry_trigger": "反彈遇重大壓力無力、指標高檔死亡交叉。",
                "stop_loss": "突破上方壓力防守點停損。",
                "take_profit": "賺取最大利潤 70%~80% 提前平倉出場。"
            }

    elif "盤整" in direction_view or "震盪" in direction_view:
        return {
            "strategy": "Iron Condor (鐵鷹價差) / 賣方雙賣價差 (Strangle Spread)",
            "type": "賣方收租 (Seller Credit)",
            "strike_advice": "上方賣出 Delta 0.15 OTM Call + 外側買保護；下方賣出 Delta 0.15 OTM Put + 外側買保護",
            "reason": "在高勝率區間內雙向賺取時間價值 (Theta) 與波動率回歸 (Vega Crush)，利用兩端買方鎖定最大風險。",
            "entry_trigger": "大盤於均線糾結量縮、指標位於 40~60 中性區、大額未平倉莊家重兵防守。",
            "stop_loss": "任一邊虧損達所收淨權利金之 1.5 倍即止損，絕不凹單。",
            "take_profit": "獲利達 50% ~ 70% 最大利潤即主動提前平倉，避開結算前 Gamma 風險。"
        }

    else:  # 雙向大暴走 (突破)
        return {
            "strategy": "Long Straddle / Strangle (買進跨式 / 結算日雙買)",
            "type": "雙買樂透 (Buyer Volatility)",
            "strike_advice": "同時買進價平 Call (Delta 0.50) + 價平 Put (Delta -0.50)",
            "reason": "即將迎來重大事件（非農/FOMC/財報/結算日），預期單邊有劇烈噴發行情但不預設多空方向。",
            "entry_trigger": "事件公佈前夕低 IV 布局，或週三結算日早盤 09:00~09:30 布局。",
            "stop_loss": "總權利金虧損 40% 停損；或事件落地後若無行情 30 分鐘內迅速撤退。",
            "take_profit": "單邊暴衝獲利覆蓋總成本後，翻倍平倉一半，另一邊歸零。"
        }
