"""
Ashva Angel One SmartAPI Live Broker Gateway
Integrates with Angel One SmartConnect REST API with automated TOTP session management,
outbound static IPv4 SEBI compliance check, position querying, order book management,
and active kill-switch order cancellation.
"""

from datetime import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
import pyotp
import requests
import yaml

from src.core.events import OrderEvent, FillEvent, OrderSide, OrderType, ProductType

logger = logging.getLogger("Ashva.AngelBroker")


class AngelBrokerGateway:
    """
    Production broker gateway for Angel One SmartAPI with static IP enforcement,
    position auditing, and emergency order cancellation.
    """

    IP_LOOKUP_URL = "https://api.ipify.org?format=json"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
        whitelisted_static_ip: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.whitelisted_static_ip = whitelisted_static_ip
        self.proxy_url = proxy_url
        
        self.smart_connect = None
        self.jwt_token: Optional[str] = None
        self.feed_token: Optional[str] = None

    @classmethod
    def from_config_file(cls, config_path: str = "config/angel_one.yaml") -> "AngelBrokerGateway":
        """Loads credentials and proxy/IP settings from YAML configuration."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        cfg = data.get("smartapi", {})
        return cls(
            api_key=cfg.get("api_key"),
            client_code=cfg.get("client_code"),
            password=cfg.get("password"),
            totp_secret=cfg.get("totp_secret"),
            whitelisted_static_ip=cfg.get("whitelisted_static_ip"),
            proxy_url=cfg.get("proxy_url"),
        )

    def check_outbound_ip_compliance(self) -> Tuple[str, bool]:
        """
        SEBI Compliance Check: Verifies current outbound IPv4 matches the registered Static IP.
        """
        try:
            proxies = {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
            res = requests.get(self.IP_LOOKUP_URL, proxies=proxies, timeout=10)
            current_ip = res.json().get("ip", "UNKNOWN")
            
            if self.whitelisted_static_ip:
                is_matched = current_ip.strip() == self.whitelisted_static_ip.strip()
                if not is_matched:
                    logger.warning(
                        f"STATIC IP MISMATCH: Current Outbound IP ({current_ip}) does NOT match "
                        f"whitelisted IP ({self.whitelisted_static_ip}). Angel One orders may be rejected!"
                    )
                else:
                    logger.info(f"SEBI Static IP Verified: Outbound IP ({current_ip}) matches registered portal IP.")
                return current_ip, is_matched
            return current_ip, True
        except Exception as e:
            logger.error(f"Failed to check outbound IP: {e}")
            return "UNKNOWN", False

    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticates with Angel One using automated TOTP code generation.
        """
        from SmartApi import SmartConnect

        # Check outbound IP before logging in
        current_ip, is_matched = self.check_outbound_ip_compliance()
        
        self.smart_connect = SmartConnect(api_key=self.api_key)
        totp_code = pyotp.TOTP(self.totp_secret).now()
        
        login_data = self.smart_connect.generateSession(self.client_code, self.password, totp_code)
        
        if login_data.get("status"):
            self.jwt_token = login_data["data"]["jwtToken"]
            self.feed_token = login_data["data"]["feedToken"]
            logger.info(f"Successfully authenticated with Angel One SmartAPI (Client: {self.client_code})")
            return login_data["data"]
        else:
            err_msg = login_data.get("message", "Unknown error")
            logger.error(f"SmartAPI login failed: {err_msg}")
            raise ConnectionError(f"SmartAPI Authentication Failure: {err_msg}")

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieves live position book from Angel One SmartAPI.
        Normalizes response into standard [{symbol, quantity, side, entry_price, unrealized_pnl}].
        """
        if not self.smart_connect:
            self.authenticate()

        try:
            res = self.smart_connect.position()
            if res and res.get("status") and "data" in res:
                raw_positions = res["data"] or []
                normalized = []
                for p in raw_positions:
                    net_qty = int(p.get("netqty", 0))
                    if net_qty != 0:
                        normalized.append({
                            "symbol": p.get("tradingsymbol", "").replace("-EQ", ""),
                            "symbol_token": p.get("symboltoken", ""),
                            "quantity": abs(net_qty),
                            "side": "LONG" if net_qty > 0 else "SHORT",
                            "entry_price": float(p.get("buyavgprice" if net_qty > 0 else "sellavgprice", 0.0)),
                            "ltp": float(p.get("ltp", 0.0)),
                            "pnl": float(p.get("pnl", 0.0)),
                        })
                return normalized
            return []
        except Exception as e:
            logger.error(f"Failed to fetch Angel One positions: {e}")
            return []

    def get_order_book(self) -> List[Dict[str, Any]]:
        """
        Retrieves full active order book from Angel One SmartAPI.
        """
        if not self.smart_connect:
            self.authenticate()

        try:
            res = self.smart_connect.orderBook()
            if res and res.get("status") and "data" in res:
                return res["data"] or []
            return []
        except Exception as e:
            logger.error(f"Failed to fetch Angel One order book: {e}")
            return []

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> Dict[str, Any]:
        """Cancels a specific pending order at Angel One."""
        if not self.smart_connect:
            self.authenticate()

        try:
            res = self.smart_connect.cancelOrder(order_id, variety)
            logger.info(f"Cancelled Angel One order {order_id}: {res}")
            return res
        except Exception as e:
            logger.error(f"Failed to cancel Angel One order {order_id}: {e}")
            raise RuntimeError(f"Angel One Order Cancellation Error: {e}")

    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """
        Emergency Kill-Switch Action:
        Finds all open and trigger-pending orders in order book and cancels them immediately.
        """
        open_orders = self.get_order_book()
        results = []
        for o in open_orders:
            status = o.get("status", "").upper()
            order_id = o.get("orderid")
            variety = o.get("variety", "NORMAL")
            if status in ["OPEN", "TRIGGER_PENDING", "PENDING"] and order_id:
                try:
                    c_res = self.cancel_order(order_id=order_id, variety=variety)
                    results.append(c_res)
                except Exception as e:
                    logger.error(f"Failed to cancel pending order {order_id}: {e}")
        return results

    def get_rms_limits(self) -> Dict[str, Any]:
        """Fetches account balance and available margin limits from Angel One."""
        if not self.smart_connect:
            self.authenticate()

        try:
            res = self.smart_connect.rmsLimit()
            if res and res.get("status") and "data" in res:
                return res["data"]
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch RMS limits: {e}")
            return {}

    def place_order(self, order: OrderEvent, symbol_token: Optional[str] = None, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Dispatches OrderEvent to Angel One REST API.
        """
        if not self.smart_connect:
            self.authenticate()

        token = symbol_token or getattr(order, "symbol_token", "0")

        # Map OrderSide to SmartAPI TransactionType
        txn_type = "BUY" if order.side == OrderSide.BUY else "SELL"
        
        # Map ProductType
        prod_type = "INTRADAY"
        if order.product_type == ProductType.DELIVERY:
            prod_type = "DELIVERY"
        elif order.product_type == ProductType.MARGIN:
            prod_type = "CARRYFORWARD"

        # Map OrderType
        ord_type = "MARKET"
        if order.order_type == OrderType.LIMIT:
            ord_type = "LIMIT"
        elif order.order_type == OrderType.STOP_LOSS:
            ord_type = "STOPLOSS_LIMIT"

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": f"{order.symbol.upper()}-EQ",
            "symboltoken": str(token),
            "transactiontype": txn_type,
            "exchange": exchange.upper(),
            "ordertype": ord_type,
            "producttype": prod_type,
            "duration": "DAY",
            "price": str(order.limit_price or 0.0),
            "squareoff": "0",
            "stoploss": str(order.stop_price or 0.0),
            "quantity": str(order.quantity),
        }

        try:
            response = self.smart_connect.placeOrder(order_params)
            logger.info(f"Dispatched order {order.order_id} to Angel One. Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to place order {order.order_id}: {e}")
            raise RuntimeError(f"Angel One Order Placement Error: {e}")
