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

        cleaned_code = stock_code.replace(".TW", "").replace(".TWO", "").strip()
        try:
            contract = self._api.Contracts.Stocks.get(cleaned_code)
            if not contract:
                contract = getattr(self._api.Contracts.Stocks, f"TSE{cleaned_code}", None) or \
                           getattr(self._api.Contracts.Stocks, f"OTC{cleaned_code}", None)

            if not contract:
                return {"error": f"在 Shioaji 合約清單中找不到代碼: {cleaned_code}"}

            snapshot = self._api.snapshots([contract])
            if not snapshot or len(snapshot) == 0:
                return {"error": "無法取得即時快照數據"}

            data = snapshot[0]
            
            bids = []
            asks = []
            if hasattr(data, "buy_price") and hasattr(data, "buy_volume"):
                for p, v in zip(data.buy_price, data.buy_volume):
                    if p > 0:
                        bids.append({"price": float(p), "volume": int(v)})

            if hasattr(data, "sell_price") and hasattr(data, "sell_volume"):
                for p, v in zip(data.sell_price, data.sell_volume):
                    if p > 0:
                        asks.append({"price": float(p), "volume": int(v)})

            close_price = float(data.close) if hasattr(data, "close") and data.close else float(data.reference_price)
            ref_price = float(data.reference_price) if hasattr(data, "reference_price") and data.reference_price else close_price
            change = close_price - ref_price
            pct_change = (change / ref_price * 100.0) if ref_price > 0 else 0.0

            return {
                "code": cleaned_code,
                "name": getattr(contract, "name", cleaned_code),
                "close": close_price,
                "open": float(data.open) if hasattr(data, "open") else close_price,
                "high": float(data.high) if hasattr(data, "high") else close_price,
                "low": float(data.low) if hasattr(data, "low") else close_price,
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

        try:
            contract = None
            if hasattr(self._api.Contracts.Futures, future_code):
                contract = getattr(self._api.Contracts.Futures, future_code)
            else:
                for target in ["TXFR1", "TX00", "MXFR1", "MX00", "TMF"]:
                    if hasattr(self._api.Contracts.Futures, target):
                        contract = getattr(self._api.Contracts.Futures, target)
                        break

            if not contract:
                return {"error": f"找不到期貨合約代碼: {future_code}"}

            snapshot = self._api.snapshots([contract])
            if not snapshot or len(snapshot) == 0:
                return {"error": "無法取得期貨即時快照數據"}

            data = snapshot[0]
            close_price = float(data.close) if hasattr(data, "close") and data.close else float(data.reference_price)
            ref_price = float(data.reference_price) if hasattr(data, "reference_price") and data.reference_price else close_price
            change = close_price - ref_price
            pct_change = (change / ref_price * 100.0) if ref_price > 0 else 0.0

            bids = []
            asks = []
            if hasattr(data, "buy_price") and hasattr(data, "buy_volume"):
                for p, v in zip(data.buy_price, data.buy_volume):
                    if p > 0:
                        bids.append({"price": float(p), "volume": int(v)})

            if hasattr(data, "sell_price") and hasattr(data, "sell_volume"):
                for p, v in zip(data.sell_price, data.sell_volume):
                    if p > 0:
                        asks.append({"price": float(p), "volume": int(v)})

            return {
                "code": contract.code,
                "name": getattr(contract, "name", "台指期"),
                "close": close_price,
                "open": float(data.open) if hasattr(data, "open") else close_price,
                "high": float(data.high) if hasattr(data, "high") else close_price,
                "low": float(data.low) if hasattr(data, "low") else close_price,
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
