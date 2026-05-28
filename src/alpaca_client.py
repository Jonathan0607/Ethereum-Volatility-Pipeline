import os
import logging
from alpaca_trade_api.rest import REST, TimeFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlpacaClient")

class AlpacaExecutionClient:
    def __init__(self):
        self.api = REST(
            key_id=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
            base_url=os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        )
        self.symbol = "ETHUSD"

    def execute_trade(self, action: str, target_weight: float, current_price: float):
        """
        Translates gateway actions into live Alpaca orders.
        action: "CASH" or "SELL_SHORT" (Breakout only)
        target_weight: Float between 0.0 and 1.0 (from the LSTM sizing)
        """
        try:
            # 1. Handle CASH action
            if action == "CASH":
                logger.info("Executing CASH action: Liquidating all ETH positions.")
                self.api.close_position(self.symbol)
                return True

            # 2. Get Account Equity for Sizing
            account = self.api.get_account()
            equity = float(account.equity)
            
            # 3. Calculate Target Fractional Quantity
            target_notional = equity * target_weight
            target_qty = round(target_notional / current_price, 4) # ETH allows up to 4 decimals

            if target_qty <= 0.001:
                logger.warning("Target quantity too small to execute. Defaulting to CASH.")
                return False

            # 4. Handle SELL_SHORT action (Crisis Alpha)
            if action == "SELL_SHORT":
                logger.info(f"Executing SHORT Breakout: Selling {target_qty} ETH.")
                # Clear any existing long positions first
                try:
                    pos = self.api.get_position(self.symbol)
                    if pos.side == 'long':
                        self.api.close_position(self.symbol)
                except Exception:
                    pass # No position exists
                
                # Submit the short order (Market order to guarantee execution)
                self.api.submit_order(
                    symbol=self.symbol,
                    qty=target_qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                return True

        except Exception as e:
            logger.error(f"Alpaca Execution Error: {e}")
            return False
