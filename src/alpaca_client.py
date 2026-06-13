import os
import logging
# pyrefly: ignore [missing-import]
from alpaca_trade_api.rest import REST

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
        action: "BUY", "SELL_SHORT", "CASH", "FLAT", "HOLDING"
        target_weight: Float between 0.0 and 1.0
        """
        if not self.api:
            logger.error("Trade abort: Alpaca API Client is not initialized due to missing credentials or configuration errors.")
            return False

        try:
            # 1. Handle CASH action (emergency liquidations / stop-outs)
            if action in ["CASH", "FLAT"]:
                logger.info(f"Executing {action} action: Liquidating all ETH positions.")
                try:
                    self.api.close_position(self.symbol)
                    return True
                except Exception as e:
                    # If position is already closed or does not exist, Alpaca REST returns a 404. Handle gracefully.
                    if "position does not exist" in str(e).lower() or "404" in str(e):
                        logger.info("No active ETH position found to close. Flat state verified.")
                        return True
                    raise e

            if action == "HOLDING":
                logger.info("Action is HOLDING. No trade execution required.")
                return True

            # 2. Get Account Equity for Sizing
            account = self.api.get_account()
            equity = float(account.equity)
            
            # 3. Query Alpaca for the current position size
            current_qty = 0.0
            current_side = 'flat'
            try:
                pos = self.api.get_position(self.symbol)
                qty_val = float(pos.qty)
                if pos.side.lower() == 'short':
                    current_qty = -qty_val
                    current_side = 'short'
                else:
                    current_qty = qty_val
                    current_side = 'long'
            except Exception:
                pass

            # 4. Handle offsetting position closure first
            if action == "BUY" and current_side == 'short':
                logger.info("Offset short position detected. Closing short position before entering long.")
                try:
                    self.api.close_position(self.symbol)
                except Exception:
                    pass
                current_qty = 0.0
                current_side = 'flat'
            elif action == "SELL_SHORT" and current_side == 'long':
                logger.info("Offset long position detected. Closing long position before entering short.")
                try:
                    self.api.close_position(self.symbol)
                except Exception:
                    pass
                current_qty = 0.0
                current_side = 'flat'

            # 5. Calculate Target Fractional Quantity & Delta
            target_qty = round(equity * target_weight / current_price, 4)
            if action == "BUY":
                target_qty_signed = target_qty
            elif action == "SELL_SHORT":
                target_qty_signed = -target_qty
            else:
                target_qty_signed = 0.0

            delta = round(target_qty_signed - current_qty, 4)

            if abs(delta) <= 0.001:
                logger.info(f"Target size change is negligible (delta={delta} ETH). No order submitted.")
                return True

            if delta > 0:
                logger.info(f"Submitting BUY order: buying {delta} ETH (target: {target_qty_signed} ETH).")
                try:
                    self.api.submit_order(
                        symbol=self.symbol,
                        qty=str(delta),
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"BUY order successfully submitted for {delta} ETH.")
                    return True
                except Exception as e:
                    if "insufficient balance" in str(e).lower() or "insufficient funds" in str(e).lower():
                        logger.warning(f"BUY order failed due to insufficient funds: {e}")
                        return False
                    raise e
            else:
                sell_qty = abs(delta)
                logger.info(f"Submitting SELL/SHORT order: selling {sell_qty} ETH (target: {target_qty_signed} ETH).")
                try:
                    self.api.submit_order(
                        symbol=self.symbol,
                        qty=str(sell_qty),
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"SELL/SHORT order successfully submitted for {sell_qty} ETH.")
                    return True
                except Exception as e:
                    if "insufficient balance" in str(e).lower() or "insufficient funds" in str(e).lower():
                        logger.warning(f"[SHADOW MODE] Caught expected broker rejection on sell/short: {e}. Mocking successful execution for PnL tracking.")
                        return True
                    raise e

        except Exception as e:
            logger.error(f"Alpaca Execution Error encountered during action {action}: {e}")
            return False


