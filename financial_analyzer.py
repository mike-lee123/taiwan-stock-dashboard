"""
Financial Analyzer (Root copy for easy upload to GitHub)
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import normalize_symbol

def analyze_financial_health(ticker_input: str) -> dict:
    norm_sym, norm_name = normalize_symbol(ticker_input)
    t = yf.Ticker(norm_sym)
    info = t.info or {}

    hist = t.history(period="1y")
    if hist.empty:
        return {"error": f"無法取得 {norm_sym} 之行情資料"}

    clean_close = hist["Close"].dropna()
    current_price = float(clean_close.iloc[-1])
    high_52w = float(hist["High"].dropna().max())
    low_52w = float(hist["Low"].dropna().min())

    eps = info.get("trailingEps")
    forward_eps = info.get("forwardEps")
    pe_ratio = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    pb_ratio = info.get("priceToBook")
    book_value = info.get("bookValue")
    market_cap = info.get("marketCap")
    
    gross_margin = (info.get("grossMargins") * 100) if info.get("grossMargins") is not None else None
    operating_margin = (info.get("operatingMargins") * 100) if info.get("operatingMargins") is not None else None
    profit_margin = (info.get("profitMargins") * 100) if info.get("profitMargins") is not None else None
    roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") is not None else None
    roa = (info.get("returnOnAssets") * 100) if info.get("returnOnAssets") is not None else None
    rev_growth = (info.get("revenueGrowth") * 100) if info.get("revenueGrowth") is not None else None

    debt_to_equity = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    quick_ratio = info.get("quickRatio")
    free_cashflow = info.get("freeCashflow")

    divs = t.dividends
    div_yield = (info.get("dividendYield") * 100) if info.get("dividendYield") is not None else None
    payout_ratio = (info.get("payoutRatio") * 100) if info.get("payoutRatio") is not None else None

    div_history = []
    if divs is not None and not divs.empty:
        if divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        recent_divs = divs.tail(5)
        for date, val in recent_divs.items():
            div_history.append({"date": date.strftime("%Y-%m-%d"), "amount": float(val)})

    ref_eps = eps if eps and eps > 0 else (current_price / 18.0)
    pe_cheap = ref_eps * 12.0
    pe_fair = ref_eps * 18.0
    pe_expensive = ref_eps * 24.0

    ref_bv = book_value if book_value and book_value > 0 else (current_price / 2.0)
    pb_cheap = ref_bv * 1.5
    pb_fair = ref_bv * 2.5
    pb_expensive = ref_bv * 3.5

    if pe_ratio:
        if pe_ratio < 13:
            valuation_status = "🟢 價值低估區（具備較高安全邊際）"
        elif pe_ratio <= 20:
            valuation_status = "🟡 合理評價區（價格與獲利相符）"
        elif pe_ratio <= 28:
            valuation_status = "🟠 偏高成長定價區（需持續高速成長支撐）"
        else:
            valuation_status = "🔴 高估警戒區（本益比偏高，注意回檔風險）"
    else:
        valuation_status = "⚪ 評價待評估"

    score = 60
    if roe and roe > 15:
        score += 10
    if profit_margin and profit_margin > 15:
        score += 10
    if current_ratio and current_ratio > 1.5:
        score += 10
    if rev_growth and rev_growth > 10:
        score += 10
    if debt_to_equity and debt_to_equity > 150:
        score -= 10

    return {
        "symbol": norm_sym,
        "name": norm_name,
        "current_price": current_price,
        "52w_high": high_52w,
        "52w_low": low_52w,
        "market_cap": market_cap,
        "eps": eps,
        "forward_eps": forward_eps,
        "pe_ratio": pe_ratio,
        "forward_pe": forward_pe,
        "pb_ratio": pb_ratio,
        "book_value": book_value,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "profit_margin": profit_margin,
        "roe": roe,
        "roa": roa,
        "revenue_growth": rev_growth,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "free_cashflow": free_cashflow,
        "dividend_yield": div_yield,
        "payout_ratio": payout_ratio,
        "div_history": div_history,
        "pe_cheap": pe_cheap,
        "pe_fair": pe_fair,
        "pe_expensive": pe_expensive,
        "pb_cheap": pb_cheap,
        "pb_fair": pb_fair,
        "pb_expensive": pb_expensive,
        "valuation_status": valuation_status,
        "health_score": min(score, 100)
    }


def generate_financial_report(ticker_input: str) -> str:
    data = analyze_financial_health(ticker_input)
    if "error" in data:
        return f"❌ 錯誤: {data['error']}"

    now = datetime.datetime.now()
    mcap_str = f"NT${data['market_cap']/1e8:,.1f} 億" if data['market_cap'] else "N/A"

    report = f"""# 📑 {data['name']} ({data['symbol']}) 財報體質診斷與價值估算報告
**分析日期**：{now.strftime('%Y年%m月%d日')} | **當前股價**：**NT${data['current_price']:,.2f}** | **總市值**：{mcap_str}
**綜合體質評分**：**`{data['health_score']} / 100`** | **當前估值位階**：**{data['valuation_status']}**

---

## 💎 獲利能力與三率分析 (Profitability)

| 財務獲利指標 | 數值 | 評估標準 | 狀態評估 |
| :--- | :---: | :---: | :--- |
| **每股盈餘 (EPS - Trailing)** | `{f"NT${data['eps']:.2f}" if data['eps'] else "N/A"}` | 獲利能力核心 | {'🟢 獲利穩健' if data['eps'] and data['eps']>0 else '⚠️ 需留意獲利'} |
| **預估每股盈餘 (Forward EPS)**| `{f"NT${data['forward_eps']:.2f}" if data['forward_eps'] else "N/A"}` | 未來一年展望 | 成長預期指標 |
| **毛利率 (Gross Margin)** | `{f"{data['gross_margin']:.2f}%" if data['gross_margin'] else "N/A"}` | 產品競爭優勢 | {'🟢 高附加價值 (>30%)' if data['gross_margin'] and data['gross_margin']>30 else '🟡 正常水準'} |
| **營業利益率 (Operating Margin)** | `{f"{data['operating_margin']:.2f}%" if data['operating_margin'] else "N/A"}` | 本業獲利能力 | {'🟢 本業獲利強 (>15%)' if data['operating_margin'] and data['operating_margin']>15 else '🟡 一般'} |
| **稅後淨利率 (Net Margin)** | `{f"{data['profit_margin']:.2f}%" if data['profit_margin'] else "N/A"}` | 最終獲利純度 | {'🟢 淨利維持良好' if data['profit_margin'] and data['profit_margin']>10 else '🟡 正常'} |
| **股東權益報酬率 (ROE)** | `{f"{data['roe']:.2f}%" if data['roe'] else "N/A"}` | 資金運用效率 | {'🌟 巴菲特優選 (>15%)' if data['roe'] and data['roe']>15 else '🟡 一般'} |
| **營收年增率 (YoY Growth)** | `{f"{data['revenue_growth']:+.2f}%" if data['revenue_growth'] else "N/A"}` | 業績成長動能 | {'🚀 業績爆發成長 (>20%)' if data['revenue_growth'] and data['revenue_growth']>20 else ('📈 穩健成長' if data['revenue_growth'] and data['revenue_growth']>0 else '📉 衰退整理')} |

---

## 🛡️ 財務結構與償債健全度 (Financial Health)

* **負債比率 (Debt / Equity)**：`{f"{data['debt_to_equity']:.1f}%" if data['debt_to_equity'] else "N/A"}` {'(🟢 負債比健康 < 100%)' if data['debt_to_equity'] and data['debt_to_equity']<100 else '(🟡 負債稍高但可控)'}
* **流動比率 (Current Ratio)**：`{f"{data['current_ratio']:.2f}" if data['current_ratio'] else "N/A"}` {'(🟢 短期償債無虞 > 1.5)' if data['current_ratio'] and data['current_ratio']>=1.5 else '(🟡 正常)'}
* **自由現金流 (Free Cash Flow)**：`{f"NT${data['free_cashflow']/1e8:,.1f} 億" if data['free_cashflow'] else "N/A"}` {'(🟢 自由現金流充沛正向)' if data['free_cashflow'] and data['free_cashflow']>0 else '(⚠️ 留意現金流支出)'}

---

## 💰 股利政策與收益評估 (Dividend Sustainability)

* **現金殖利率 (Dividend Yield)**：**`{f"{data['dividend_yield']:.2f}%" if data['dividend_yield'] else "N/A"}`**
* **盈餘發放率 (Payout Ratio)**：`{f"{data['payout_ratio']:.1f}%" if data['payout_ratio'] else "N/A"}` {'(🟢 配息政策大方且健康 50-80%)' if data['payout_ratio'] and 40<=data['payout_ratio']<=85 else '(政策維持)'}
* **近期配息紀錄**：
"""
    if data['div_history']:
        for d in data['div_history']:
            report += f"  - `{d['date']}`：每股現金股利 `NT${d['amount']:.2f}`\n"
    else:
        report += "  - 暫無近期除息明細\n"

    report += f"""
---

## 🎯 多模型目標價與合理價值推估 (Valuation Models)

### 1. 本益比估值法 (P/E Multiple Model)
* 當前本益比：`{f"{data['pe_ratio']:.2f} 倍" if data['pe_ratio'] else "N/A"}` (預估遠期 P/E: `{f"{data['forward_pe']:.2f} 倍" if data['forward_pe'] else "N/A"}`)
* **便宜價 (12x P/E)**：`NT${data['pe_cheap']:.1f}`
* **合理價 (18x P/E)**：`NT${data['pe_fair']:.1f}`
* **昂貴價 (24x P/E)**：`NT${data['pe_expensive']:.1f}`

### 2. 股價淨值比估值法 (P/B Model)
* 當前股價淨值比：`{f"{data['pb_ratio']:.2f} 倍" if data['pb_ratio'] else "N/A"}` (每股淨值: `{f"NT${data['book_value']:.1f}" if data['book_value'] else "N/A"}`)
* **淨值便宜區 (1.5x)**：`NT${data['pb_cheap']:.1f}`
* **淨值合理區 (2.5x)**：`NT${data['pb_fair']:.1f}`
* **淨值昂貴區 (3.5x)**：`NT${data['pb_expensive']:.1f}`

---

## 🧭 投資決策綜合總結

1. **基本面體質**：{data['name']} 在獲利與產業地位上{'展現極佳的競爭優勢與定價權' if data['health_score']>=80 else '維持穩健的營運表現'}。
2. **進場策略建議**：
   * 目前股價落在 **{data['valuation_status'].split('（')[0]}**。
   * 若欲長線存股或分批佈局，可參考 **合理價 (NT${data['pe_fair']:.1f})** 以下分批逢低承接，並以 **便宜價 (NT${data['pe_cheap']:.1f})** 作為強力價值防守區。
"""
    return report
