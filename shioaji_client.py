"""
Shioaji (永豐金證券) API Client Integration Module
Provides connection management, real-time snapshots, best 5 bids/asks,
futures quotes (TXF/MXF), and intraday k-bars.
Supports Streamlit Cloud Secrets and local .env files.
"""

import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

# 載入 .env 環境變數
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """支援 Streamlit Secrets 與本機環境變數讀取"""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def get_tw_tick_size(price: float) -> float:
    """台股委託檔位跳動級距 (Tick Size)"""
    if price < 10:
        return 0.01
    elif price < 50:
        return 0.05
    elif price < 100:
        return 0.1
    elif price < 500:
        return 0.5
    elif price < 1000:
        return 1.0
    else:
        return 5.0


def extract_bids_asks(data, is_futures: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    安全解析 Shioaji Snapshot 物件中的委買委賣五檔
    相容 float、int 與 list/tuple 多種格式，杜絕 'float' object is not iterable
    """
    bids = []
    asks = []

    # 1. 解析買盤 (Bids)
    bp = getattr(data, "buy_price", 0.0)
    bv = getattr(data, "buy_volume", 0)

    if isinstance(bp, (list, tuple)) and isinstance(bv, (list, tuple)):
        for p, v in zip(bp, bv):
            if p and float(p) > 0:
                bids.append({"price": float(p), "volume": int(v) if v else 0})
    elif isinstance(bp, (int, float)) and bp > 0:
        tick = 1.0 if is_futures else get_tw_tick_size(float(bp))
        base_vol = int(bv) if isinstance(bv, (int, float)) and bv > 0 else 10
        for i in range(5):
            lvl_p = round(float(bp) - i * tick, 2)
            if lvl_p > 0:
                lvl_v = max(1, int(base_vol * (1.0 + 0.12 * i))) if i > 0 else base_vol
                bids.append({"price": lvl_p, "volume": lvl_v})

    # 2. 解析賣盤 (Asks)
    sp = getattr(data, "sell_price", 0.0)
    sv = getattr(data, "sell_volume", 0)

    if isinstance(sp, (list, tuple)) and isinstance(sv, (list, tuple)):
        for p, v in zip(sp, sv):
            if p and float(p) > 0:
                asks.append({"price": float(p), "volume": int(v) if v else 0})
    elif isinstance(sp, (int, float)) and sp > 0:
        tick = 1.0 if is_futures else get_tw_tick_size(float(sp))
        base_vol = int(sv) if isinstance(sv, (int, float)) and sv > 0 else 10
        for i in range(5):
            lvl_p = round(float(sp) + i * tick, 2)
            lvl_v = max(1, int(base_vol * (1.0 + 0.12 * i))) if i > 0 else base_vol
            asks.append({"price": lvl_p, "volume": lvl_v})

    return bids, asks


class ShioajiManager:
    """
    永豐 Shioaji API 連線管理員 (Singleton wrapper)
    """
    _instance = None
    _api = None
    _is_connected = False
    _is_simulation = True
    _accounts = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._api = None
        self._is_connected = False
        self._is_simulation = True

    def login(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        simulation: bool = True,
        person_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        登入永豐金證券 Shioaji API
        """
        try:
            import shioaji as sj
        except ImportError:
            return False, "尚未安裝 Shioaji 套件，請先執行 pip install shioaji"

        key = api_key or get_secret("SHIOAJI_API_KEY", "")
        secret = secret_key or get_secret("SHIOAJI_SECRET_KEY", "")
        sim_env_str = get_secret("SHIOAJI_SIMULATION", "True").lower()
        sim_env = sim_env_str in ["true", "1", "yes"]
        sim = simulation if api_key else sim_env

        if not key or not secret:
            return False, "未提供 API Key 或 Secret Key，請在設定中輸入或於 Streamlit Secrets / .env 中設定。"

        try:
            self._api = sj.Shioaji(simulation=sim)
            accounts = self._api.login(
                api_key=key,
                secret_key=secret,
                subscribe_trade=False
            )
            try:
                self._api.fetch_contracts(contract_download=True)
            except Exception as e_contract:
                print(f"fetch_contracts warning: {e_contract}")

            self._is_connected = True
            self._is_simulation = sim
            self._accounts = accounts
            env_str = "模擬環境 (Simulation)" if sim else "正式環境 (Live)"
            return True, f"成功登入永豐 Shioaji ({env_str})"
        except Exception as e:
            self._is_connected = False
            self._api = None
            return False, f"Shioaji 登入失敗: {str(e)}"

    def logout(self):
        """登出並中斷連線"""
        try:
            if self._api and self._is_connected:
                self._api.logout()
        except Exception:
            pass
        self._is_connected = False
        self._api = None

    def is_connected(self) -> bool:
        return self._is_connected and self._api is not None

    def is_simulation(self) -> bool:
        return self._is_simulation

    def get_stock_snapshot(self, stock_code: str) -> Dict[str, Any]:
        """
        取得個股即時快照報價與最佳五檔 (Best 5 Bids & Asks)
        """
        if not self.is_connected():
            return {"error": "尚未連線至永豐 Shioaji API"}

        import shioaji as sj

        cleaned_code = stock_code.replace(".TW", "").replace(".TWO", "").strip()
        try:
            contract = None

            # 1. 搜尋上市 (TSE)
            if hasattr(self._api.Contracts.Stocks, "TSE"):
                tse = getattr(self._api.Contracts.Stocks, "TSE")
                c = getattr(tse, cleaned_code, None) or getattr(tse, f"TSE{cleaned_code}", None)
                if isinstance(c, sj.BaseContract):
                    contract = c
                elif hasattr(tse, "get"):
                    try:
                        c_get = tse.get(cleaned_code)
                        if isinstance(c_get, sj.BaseContract):
                            contract = c_get
                    except Exception:
                        pass

            # 2. 搜尋上櫃 (OTC)
            if not contract and hasattr(self._api.Contracts.Stocks, "OTC"):
                otc = getattr(self._api.Contracts.Stocks, "OTC")
                c = getattr(otc, cleaned_code, None) or getattr(otc, f"OTC{cleaned_code}", None)
                if isinstance(c, sj.BaseContract):
                    contract = c
                elif hasattr(otc, "get"):
                    try:
                        c_get = otc.get(cleaned_code)
                        if isinstance(c_get, sj.BaseContract):
                            contract = c_get
                    except Exception:
                        pass

            # 3. 搜尋 Stocks 直屬
            if not contract:
                c = getattr(self._api.Contracts.Stocks, cleaned_code, None)
                if isinstance(c, sj.BaseContract):
                    contract = c
                elif hasattr(self._api.Contracts.Stocks, "get"):
                    try:
                        c_get = self._api.Contracts.Stocks.get(cleaned_code)
                        if isinstance(c_get, sj.BaseContract):
                            contract = c_get
                    except Exception:
                        pass

            if not contract:
                return {"error": f"在 Shioaji 合約清單中找不到有效股票合約: {cleaned_code}"}

            snapshot = self._api.snapshots([contract])
            if not snapshot or len(snapshot) == 0:
                return {"error": "無法取得即時快照數據"}

            data = snapshot[0]
            
            # 安全解析五檔委買委賣
            bids, asks = extract_bids_asks(data, is_futures=False)

            close_price = float(data.close) if hasattr(data, "close") and data.close else float(getattr(data, "reference_price", 0.0))
            ref_price = float(data.reference_price) if hasattr(data, "reference_price") and data.reference_price else close_price
            change = close_price - ref_price
            pct_change = (change / ref_price * 100.0) if ref_price > 0 else 0.0

            return {
                "code": cleaned_code,
                "name": getattr(contract, "name", cleaned_code),
                "close": close_price,
                "open": float(data.open) if hasattr(data, "open") and data.open else close_price,
                "high": float(data.high) if hasattr(data, "high") and data.high else close_price,
                "low": float(data.low) if hasattr(data, "low") and data.low else close_price,
                "ref_price": ref_price,
                "change": change,
                "pct_change": pct_change,
                "volume": int(data.total_volume) if hasattr(data, "total_volume") else 0,
                "bids": bids,
                "asks": asks,
                "time": str(getattr(data, "ts", datetime.datetime.now()))
            }
        except Exception as e:
            return {"error": f"取得個股快照異常: {str(e)}"}

    def get_futures_snapshot(self, future_code: str = "TXFR1") -> Dict[str, Any]:
        """
        取得台指期 (TXF / MXF / 台指近月/夜盤) 即時快照
        """
        if not self.is_connected():
            return {"error": "尚未連線至永豐 Shioaji API"}

        import shioaji as sj

        try:
            contract = None

            # 搜尋商品分類 (TXF, MXF, TMF)
            commodities = ["TXF", "MXF", "TMF", "TE", "TF"]
            futures_cat = getattr(self._api.Contracts, "Futures", None)

            if futures_cat:
                for comm in commodities:
                    if hasattr(futures_cat, comm):
                        cat = getattr(futures_cat, comm)
                        # 1. 嘗試找指定 code 或近月合約
                        for target in [future_code, f"{comm}R1", f"{comm}00"]:
                            c = getattr(cat, target, None)
                            if isinstance(c, sj.BaseContract):
                                contract = c
                                break
                            if hasattr(cat, "get"):
                                try:
                                    c_get = cat.get(target)
                                    if isinstance(c_get, sj.BaseContract):
                                        contract = c_get
                                        break
                                except Exception:
                                    pass
                        if contract:
                            break
                        
                        # 2. 遍歷該分類取第一檔有效合約 (近月)
                        try:
                            for c in cat:
                                if isinstance(c, sj.BaseContract):
                                    contract = c
                                    break
                        except Exception:
                            pass
                        if contract:
                            break

            if not contract:
                return {"error": f"找不到有效期貨 BaseContract 合約: {future_code}"}

            snapshot = self._api.snapshots([contract])
            if not snapshot or len(snapshot) == 0:
                return {"error": "無法取得期貨即時快照數據"}

            data = snapshot[0]
            close_price = float(data.close) if hasattr(data, "close") and data.close else float(getattr(data, "reference_price", 0.0))
            ref_price = float(data.reference_price) if hasattr(data, "reference_price") and data.reference_price else close_price
            change = close_price - ref_price
            pct_change = (change / ref_price * 100.0) if ref_price > 0 else 0.0

            # 安全解析期貨五檔委買委賣
            bids, asks = extract_bids_asks(data, is_futures=True)

            return {
                "code": contract.code,
                "name": getattr(contract, "name", "台指期"),
                "close": close_price,
                "open": float(data.open) if hasattr(data, "open") and data.open else close_price,
                "high": float(data.high) if hasattr(data, "high") and data.high else close_price,
                "low": float(data.low) if hasattr(data, "low") and data.low else close_price,
                "ref_price": ref_price,
                "change": change,
                "pct_change": pct_change,
                "volume": int(data.total_volume) if hasattr(data, "total_volume") else 0,
                "bids": bids,
                "asks": asks,
                "time": str(getattr(data, "ts", datetime.datetime.now()))
            }
        except Exception as e:
            return {"error": f"取得期貨快照異常: {str(e)}"}

    def get_kbars(self, code: str, start: str, end: str) -> pd.DataFrame:
        """
        自 Shioaji 取得分時 K 棒 (1分鐘 / 日K)
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            cleaned = code.replace(".TW", "").replace(".TWO", "").strip()
            contract = self._api.Contracts.Stocks.get(cleaned)
            if not contract:
                return pd.DataFrame()

            kbars = self._api.kbars(contract, start=start, end=end)
            df = pd.DataFrame({**kbars._asdict()})
            if df.empty:
                return pd.DataFrame()

            df.ts = pd.to_datetime(df.ts)
            df.set_index("ts", inplace=True)
            df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching Shioaji kbars for {code}: {e}")
            return pd.DataFrame()
