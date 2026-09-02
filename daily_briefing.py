"""
Daily Briefing (Root copy for easy upload to GitHub)
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def get_latest_quote(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            return {"price": np.nan, "change": 0.0, "pct_change": 0.0, "date": "N/A"}
        
        clean_close = hist["Close"].dropna()
        if len(clean_close) >= 1:
            latest = float(clean_close.iloc[-1])
            prev = float(clean_close.iloc[-2]) if len(clean_close) > 1 else latest
            chg = latest - prev
            pct = (chg / prev * 100.0) if prev > 0 else 0.0
            return {
                "price": latest,
                "change": chg,
                "pct_change": pct,
                "high": float(hist["High"].dropna().iloc[-1]),
                "low": float(hist["Low"].dropna().iloc[-1]),
                "date": clean_close.index[-1].strftime("%Y-%m-%d")
            }
    except Exception:
        pass
    return {"price": np.nan, "change": 0.0, "pct_change": 0.0, "date": "N/A"}


def generate_pre_market_briefing() -> str:
    now = datetime.datetime.now()
    date_str = now.strftime("%Y年%m月%d日")

    sox = get_latest_quote("^SOX")
    dji = get_latest_quote("^DJI")
    ixic = get_latest_quote("^IXIC")
    spx = get_latest_quote("^GSPC")
    fx = get_latest_quote("USDTWD=X")
    ewt = get_latest_quote("EWT")
    kospi = get_latest_quote("^KS11")
    tsm = get_latest_quote("TSM")
    tw_2330 = get_latest_quote("2330.TW")

    adr_prem_pct = 0.0
    adr_equiv_twd = 0.0
    if not np.isnan(tsm["price"]) and not np.isnan(fx["price"]) and not np.isnan(tw_2330["price"]):
        adr_equiv_twd = (tsm["price"] * fx["price"]) / 5.0
        adr_prem_pct = ((adr_equiv_twd - tw_2330["price"]) / tw_2330["price"]) * 100.0

    sentiment_score = 0
    if sox["pct_change"] > 1.0:
        sentiment_score += 2
    elif sox["pct_change"] > 0:
        sentiment_score += 1
    elif sox["pct_change"] < -1.0:
        sentiment_score -= 2
    else:
        sentiment_score -= 1

    if tsm["pct_change"] > 1.0:
        sentiment_score += 2
    elif tsm["pct_change"] < -1.0:
        sentiment_score -= 2

    if fx["pct_change"] < 0:
        sentiment_score += 1
    else:
        sentiment_score -= 1

    if sentiment_score >= 3:
        market_bias = "🟢 偏多開高（科技半導體強勢領漲）"
    elif sentiment_score >= 1:
        market_bias = "🟡 溫和偏多（震盪盤堅，逢低有撐）"
    elif sentiment_score <= -3:
        market_bias = "🔴 偏空開低（承壓震盪，注意防守支撐）"
    else:
        market_bias = "⚪ 中性整理（多空拉鋸，區間震盪）"

    report = f"""# 🌅 台股早盤情報速遞 (Pre-Market Briefing)
**報告產出時間**：{date_str} {now.strftime('%H:%M')}
**今日盤前展望**：**{market_bias}**

---

## 🌍 隔夜美股與國際市場掃描

| 關鍵指標 | 最新收盤 | 漲跌點數 | 漲跌幅 (%) | 趨勢解讀 |
| :--- | :---: | :---: | :---: | :--- |
| **費城半導體 (^SOX)** | `{sox['price']:,.2f}` | `{sox['change']:+.2f}` | **`{sox['pct_change']:+.2f}%`** | {'🚀 強勢大漲' if sox['pct_change']>=1 else ('📈 穩健上揚' if sox['pct_change']>0 else '📉 回檔整理')} |
| **那斯達克 (^IXIC)** | `{ixic['price']:,.2f}` | `{ixic['change']:+.2f}` | **`{ixic['pct_change']:+.2f}%`** | {'🔥 科技股買盤強' if ixic['pct_change']>0 else '⚠️ 科技股承壓'} |
| **道瓊工業指數 (^DJI)** | `{dji['price']:,.2f}` | `{dji['change']:+.2f}` | **`{dji['pct_change']:+.2f}%`** | 傳統藍籌權值代表 |
| **標普 500 (^GSPC)** | `{spx['price']:,.2f}` | `{spx['change']:+.2f}` | **`{spx['pct_change']:+.2f}%`** | 美股大盤綜合指標 |
| **韓國綜合指數 (KOSPI)** | `{kospi['price']:,.2f}` | `{kospi['change']:+.2f}` | **`{kospi['pct_change']:+.2f}%`** | 亞股半導體連動市場 |

---

## 🎯 台積電 ADR 溢價率與夜盤連動

* **台積電 ADR (TSM)**：`${tsm['price']:.2f}` ({tsm['pct_change']:+.2f}%)
* **美元兌新台幣 (USD/TWD)**：`NT${fx['price']:.3f}` ({fx['pct_change']:+.2f}%) → {'🔥 台幣升值，有利外資回流' if fx['pct_change']<=0 else '⚠️ 台幣微貶，留意資金流出'}
* **ADR 折算現股市價**：`NT${adr_equiv_twd:.1f}`
* **台積電現股昨日收盤**：`NT${tw_2330['price']:.1f}`
* **💡 最新 ADR 溢價率**：**`{adr_prem_pct:+.2f}%`**
  * *溢價率解讀*：{'正溢價維持高檔，台積電開盤具備向上比價動能' if adr_prem_pct>10 else ('正溢價平穩，有助支撐大盤開高' if adr_prem_pct>0 else '呈現折價，注意開盤震盪壓回')}
* **MSCI 台灣指數 ETF (EWT)**：`${ewt['price']:.2f}` ({ewt['pct_change']:+.2f}%)

---

## 🧭 今日操盤重點與族群觀察

1. **半導體與 AI 供應鏈**：
   * 費半與台積電 ADR 表現{'強勁，預期帶動台積電 (2330)、聯發科 (2454)、廣達 (2382)、鴻海 (2317) 早盤強勢表態' if sox['pct_change']>0 else '回檔，留意高檔獲利了結賣壓，觀察月線支撐是否穩固'}。
2. **匯率與外資動向**：
   * 當前匯率為 `{fx['price']:.3f}`，觀察開盤後外資在期現貨是否同步作多。
3. **操作建議**：
   * {'開高後不宜盲目追高，留意預估成交量若未放大可能出現拉回震盪，可於均線附近低接。' if sox['pct_change']>1 else '維持逢低分批佈局強勢股，嚴設 5 日均線停損防守點。'}
"""
    return report


def generate_post_market_briefing() -> str:
    now = datetime.datetime.now()
    date_str = now.strftime("%Y年%m月%d日")

    twii = get_latest_quote("^TWII")
    tw_2330 = get_latest_quote("2330.TW")
    tw_2317 = get_latest_quote("2317.TW")
    tw_2454 = get_latest_quote("2454.TW")
    tw_0050 = get_latest_quote("0050.TW")
    fx = get_latest_quote("USDTWD=X")

    report = f"""# 🌆 台股盤後總結與籌碼戰報 (Post-Market Review)
**報告產出時間**：{date_str} {now.strftime('%H:%M')}

---

## 📊 今日大盤行情總結

* **加權指數 (TAIEX)**：`{twii['price']:,.2f}` 點
* **今日漲跌**：`{twii['change']:+.2f}` 點 (**`{twii['pct_change']:+.2f}%`**)
* **今日高低震盪**：最高 `{twii['high']:,.2f}` / 最低 `{twii['low']:,.2f}`
* **匯率收盤**：美元兌新台幣 `{fx['price']:.3f}` ({fx['pct_change']:+.2f}%)

---

## 👑 核心指標股今日收盤統計

| 標的名稱 | 代碼 | 收盤價 (TWD) | 今日漲跌 | 漲跌幅 (%) | 狀態評估 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **台積電** | `2330` | `{tw_2330['price']:.1f}` | `{tw_2330['change']:+.1f}` | **`{tw_2330['pct_change']:+.2f}%`** | 護國神山/大盤定海神針 |
| **鴻海** | `2317` | `{tw_2317['price']:.1f}` | `{tw_2317['change']:+.1f}` | **`{tw_2317['pct_change']:+.2f}%`** | AI 伺服器代工指標 |
| **聯發科** | `2454` | `{tw_2454['price']:.1f}` | `{tw_2454['change']:+.1f}` | **`{tw_2454['pct_change']:+.2f}%`** | IC 設計族群領頭羊 |
| **元大台灣50**| `0050` | `{tw_0050['price']:.2f}` | `{tw_0050['change']:+.2f}` | **`{tw_0050['pct_change']:+.2f}%`** | 市值型大盤連動 ETF |

---

## 🔍 明日觀盤重點

1. **籌碼沉澱狀況**：觀察三大法人外資與投信是否維持買超態勢，融資餘額若維持健康未過度暴增，則多頭格局不變。
2. **夜盤美股關注重點**：今晚關注美股開盤後費半指數能否續強，以及即將公布之美國經濟數據。
"""
    return report
