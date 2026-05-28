import os
import logging
from alpaca_trade_api.rest import REST, TimeFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlpacaClient")

class AlpacaExecutionClient:
    def __init__(self):
        self.key_id = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        self.symbol = "ETHUSD"
        
        if not self.key_id or not self.secret_key:
            logger.error("CRITICAL CONFIGURATION ERROR: Alpaca API credentials (APCA_API_KEY_ID, APCA_API_SECRET_KEY) are missing from the environment variables.")
            self.api = None
        else:
            try:
                self.api = REST(
                    key_id=self.key_id,
                    secret_key=self.secret_key,
                    base_url=self.base_url
                )
                logger.info(f"Alpaca REST client successfully initialized. Target base URL: {self.base_url}")
            except Exception as e:
                logger.error(f"CRITICAL INITIALIZATION ERROR: Failed to instantiate Alpaca REST API: {e}")
                self.api = None

    def execute_trade(self, action: str, target_weight: float, current_price: float):
        """
        Translates gateway actions into live Alpaca orders.
        action: "CASH" or "SELL_SHORT" (Breakout only)
        target_weight: Float between 0.0 and 1.0 (from the LSTM sizing)
        """
        if not self.api:
            logger.error("Trade abort: Alpaca API Client is not initialized due to missing credentials or configuration errors.")
            return False

        try:
            # 1. Handle CASH action (emergency liquidations / stop-outs)
            if action == "CASH":
                logger.info("Executing CASH action: Liquidating all ETH positions.")
                try:
                    self.api.close_position(self.symbol)
                    return True
                except Exception as e:
                    # If position is already closed or does not exist, Alpaca REST returns a 404. Handle gracefully.
                    if "position does not exist" in str(e).lower() or "404" in str(e):
                        logger.info("No active ETH position found to close. Flat state verified.")
                        return True
                    raise e

            # 2. Get Account Equity for Sizing
            account = self.api.get_account()
            equity = float(account.equity)
            
            # 3. Calculate Target Fractional Quantity
            target_notional = equity * target_weight
            target_qty = round(target_notional / current_price, 4) # ETH allows up to 4 decimals

            if target_qty <= 0.001:
                logger.warning(f"Aborting execution: Calculated target quantity ({target_qty} ETH) is below minimum execution threshold (0.001 ETH).")
                return False

            # 4. Handle SELL_SHORT action (Crisis Alpha)
            if action == "SELL_SHORT":
                logger.info(f"Executing SHORT Breakout: Selling {target_qty} ETH at approx ${current_price:.2f}.")
                
                # Clear any existing long positions first to avoid counter-position offset warnings
                try:
                    pos = self.api.get_position(self.symbol)
                    if pos.side == 'long':
                        logger.info("Offset long position detected. Closing long position before entering short.")
                        self.api.close_position(self.symbol)
                except Exception:
                    pass # No position exists
                
                # Submit the short order (Market order to guarantee execution)
                self.api.submit_order(
                    symbol=self.symbol,
                    qty=str(target_qty), # Use string serialization to avoid decimal truncation issues
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"SHORT order successfully submitted for {target_qty} ETH.")
                return True

        except Exception as e:
            logger.error(f"Alpaca Execution Error encountered during action {action}: {e}")
            return False

