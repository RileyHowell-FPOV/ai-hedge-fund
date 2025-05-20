import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.analysts import ANALYST_ORDER
from src.main import run_hedge_fund
from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from src.utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable
from src.utils.ollama import ensure_ollama_and_model

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0,
        interval: str = "day",
        interval_multiplier: int = 1,
    ):
        """
        :param agent: The trading agent (Callable).
        :param tickers: List of tickers to backtest.
        :param start_date: Start date string (YYYY-MM-DD).
        :param end_date: End date string (YYYY-MM-DD).
        :param initial_capital: Starting portfolio cash.
        :param model_name: Which LLM model name to use (gpt-4, etc).
        :param model_provider: Which LLM provider (OpenAI, etc).
        :param selected_analysts: List of analyst names or IDs to incorporate.
        :param initial_margin_requirement: The margin ratio (e.g. 0.5 = 50%).
        :param interval: Time interval for price data.
        :param interval_multiplier: Multiplier for the interval.
        """
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.interval = interval
        self.interval_multiplier = interval_multiplier

        # Initialize portfolio with support for long/short positions
        self.portfolio_values = []
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,  # total margin usage across all short positions
            "margin_requirement": initial_margin_requirement,  # The margin ratio required for shorts
            "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for ticker in tickers},  # Number of shares held long  # Number of shares held short  # Average cost basis per share (long)  # Average cost basis per share (short)  # Dollars of margin used for this ticker's short
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                }
                for ticker in tickers
            },
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        Execute trades with support for both long and short positions.
        `quantity` is the number of shares the agent wants to buy/sell/short/cover.
        We will only trade integer shares to keep it simple.
        """
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # force integer shares
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] / current_price)
                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            # You can only sell as many as you own
            quantity = min(quantity, position["long"])
            if quantity > 0:
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = current_price * quantity
              2) Post margin_required = proceeds * margin_ratio
              3) Net effect on cash = +proceeds - margin_required
            """
            proceeds = current_price * quantity
            margin_required = proceeds * self.portfolio["margin_requirement"]
            if margin_required <= self.portfolio["cash"]:
                # Weighted average short cost basis
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity

                # Update margin usage
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required

                # Increase cash by proceeds, then subtract the required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity
                margin_ratio = self.portfolio["margin_requirement"]
                if margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (current_price * margin_ratio))
                else:
                    max_quantity = 0

                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required

                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = current_price * quantity
              2) Release a proportional share of the margin
              3) Net effect on cash = -cover_cost + released_margin
            """
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - current_price) * quantity

                if position["short"] > 0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0

                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0

    def calculate_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value, including:
          - cash
          - market value of long positions
          - unrealized gains/losses for short positions
        """
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]

            # Long position value
            long_value = position["long"] * price
            total_value += long_value

            # Short position unrealized PnL = short_shares * (short_cost_basis - current_price)
            if position["short"] > 0:
                total_value -= position["short"] * price

        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        print("\nPre-fetching data for the entire backtest period...")

        # Convert end_date string to datetime, fetch up to 1 year before
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self.tickers:
            # Fetch price data for the entire period, plus 1 year
            get_prices(
                ticker,
                start_date_str, 
                self.end_date,
                interval=self.interval,
                interval_multiplier=self.interval_multiplier
            )

            # Fetch financial metrics
            get_financial_metrics(ticker, self.end_date, limit=10)

            # Fetch insider trades
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)

            # Fetch company news
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)

        print("Data pre-fetch complete.")

    def run_backtest(self):
        # Pre-fetch all data at the start
        self.prefetch_data()

        all_price_data = {}
        min_date = None
        max_date = None

        # If using a sub-daily interval, fetch all data for the entire date range
        # and iterate over the unique timestamps within that range.
        # Otherwise (if interval is "day"), use the existing daily iteration logic.
        if self.interval != "day":
            print(f"\nRunning {self.interval} backtest from {self.start_date} to {self.end_date} with multiplier {self.interval_multiplier}x")
            for ticker in self.tickers:
                try:
                    # Fetch all data for the entire date range with the specified interval
                    df = get_price_data(ticker, self.start_date, self.end_date, self.interval, self.interval_multiplier)
                    if not df.empty:
                        all_price_data[ticker] = df.sort_index() 
                        current_min_date = df.index.min()
                        current_max_date = df.index.max()
                        if min_date is None or current_min_date < min_date:
                            min_date = current_min_date
                        if max_date is None or current_max_date > max_date:
                            max_date = current_max_date
                    else:
                        print(f"Warning: No {self.interval} price data for {ticker} between {self.start_date} and {self.end_date}. Skipping ticker for sub-daily processing.")
                except Exception as e:
                    print(f"Error fetching initial {self.interval} price data for {ticker} between {self.start_date} and {self.end_date}: {e}. Skipping ticker.")
                    continue 
            
            if not all_price_data:
                print(f"Error: No sub-daily price data found for any ticker for the range {self.start_date} - {self.end_date} at {self.interval} interval. Aborting backtest.")
                return {}

            unique_timestamps = pd.Index([])
            for ticker_df in all_price_data.values():
                unique_timestamps = unique_timestamps.union(ticker_df.index)
            
            unique_timestamps = unique_timestamps.sort_values()
            iteration_points = unique_timestamps
            if not iteration_points.empty:
                print(f"Found {len(iteration_points)} unique timestamps for backtesting between {min_date} and {max_date}.")
            else:
                print(f"Warning: No unique timestamps found for {self.interval} backtest between {self.start_date} and {self.end_date} despite some ticker data being present. Check data alignment.")
                return {}

        else: # interval is "day"
            iteration_points = pd.date_range(self.start_date, self.end_date, freq="B")
            print(f"\nRunning daily backtest from {self.start_date} to {self.end_date}...")


        table_rows = []
        performance_metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None, "long_short_ratio": None, "gross_exposure": None, "net_exposure": None}

        print("\nStarting backtest...")

        # Initialize portfolio values list with initial capital
        if len(iteration_points) > 0:
            self.portfolio_values = [{"Date": iteration_points[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        for current_timestamp_or_date in iteration_points:
            if self.interval != "day": # Covers both single-day and multi-day sub-daily
                current_datetime_obj = current_timestamp_or_date 
                current_date_str = current_datetime_obj.strftime("%Y-%m-%d %H:%M:%S") 
                agent_lookback_start_date = (current_datetime_obj.normalize() - timedelta(days=30)).strftime("%Y-%m-%d")
                agent_current_date = current_datetime_obj.strftime("%Y-%m-%d") 
            else: # Daily backtesting
                current_datetime_obj = current_timestamp_or_date # This is a pandas Timestamp (representing a business day)
                current_date_str = current_datetime_obj.strftime("%Y-%m-%d")
                agent_lookback_start_date = (current_datetime_obj - timedelta(days=30)).strftime("%Y-%m-%d")
                agent_current_date = current_date_str

            # Skip if there's no prior day/timestamp to look back (i.e., first point in the range)
            # This check might need refinement for intraday, but for now, a simple check for first item
            if current_timestamp_or_date == iteration_points[0] and len(iteration_points) > 1:
                 # For intraday, we still want to process the first timestamp unless it's the only one
                 # For daily, we skip the first day if lookback_start == current_date_str, which is handled by original logic for daily.
                 if not (self.interval != "day"):
                    # This was the original condition for daily: if lookback_start == current_date_str: continue
                    # We need to ensure agent_lookback_start_date is defined for this check if we reinstate it.
                    # For now, the first timestamp for sub-daily will be processed.
                    pass 

            # Get current prices for all tickers
            try:
                current_prices = {}
                missing_data_for_timestamp = False

                for ticker in self.tickers:
                    if ticker not in all_price_data and self.interval != "day" and self.start_date == self.end_date:
                        # This ticker was skipped during initial load for single-day, sub-daily interval
                        # We need a strategy here - e.g., assume it can't be traded, or use last known price (which we don't have easily here)
                        # For now, let's mark as missing data for this timestamp to skip trading this ticker at this point.
                        # This implicitly means the agent won't be able to trade it if its data isn't loaded.
                        # print(f"Skipping {ticker} at {current_date_str}, data not preloaded.")
                        missing_data_for_timestamp = True 
                        break
                    
                    if self.interval != "day": # Covers both single-day and multi-day sub-daily
                        # For sub-daily, get the price from the pre-fetched DataFrame
                        # Use asof to get the latest price at or before the current_datetime_obj
                        if ticker in all_price_data and not all_price_data[ticker].empty:
                            # Ensure current_datetime_obj is timezone-aware if DataFrame index is, or vice-versa
                            # Timestamps from price data might be tz-aware (e.g., UTC).
                            # We need to make them compatible for comparison/lookup.
                            df_ticker = all_price_data[ticker]
                            
                            # Timezone handling:
                            # Ensure current_datetime_obj is correctly aligned with df_ticker.index.tz
                            iteration_timestamp = pd.Timestamp(current_timestamp_or_date) # Ensure it's a pandas Timestamp

                            if df_ticker.index.tz is not None: # DataFrame index is timezone-aware
                                if iteration_timestamp.tz is None: # Iteration timestamp is naive
                                    current_datetime_obj_localized = iteration_timestamp.tz_localize(df_ticker.index.tz)
                                elif iteration_timestamp.tz != df_ticker.index.tz: # Both aware, but different timezones
                                    current_datetime_obj_localized = iteration_timestamp.tz_convert(df_ticker.index.tz)
                                else: # Both aware and same timezone
                                    current_datetime_obj_localized = iteration_timestamp
                            else: # DataFrame index is timezone-naive
                                if iteration_timestamp.tz is not None: # Iteration timestamp is aware
                                    current_datetime_obj_localized = iteration_timestamp.tz_convert(None) # Make it naive
                                else: # Both naive
                                    current_datetime_obj_localized = iteration_timestamp
                            
                            # Find the row for the current exact timestamp
                            if current_datetime_obj_localized in df_ticker.index:
                                price_at_timestamp = df_ticker.loc[current_datetime_obj_localized]
                                if not pd.isna(price_at_timestamp["close"]):
                                    current_prices[ticker] = price_at_timestamp["close"]
                                else:
                                    # print(f"Warning: No explicit close price for {ticker} at {current_date_str}. Using asof.")
                                    # Fallback to asof if exact match has NaN close, or if needed
                                    price_series = df_ticker["close"].asof(current_datetime_obj_localized)
                                    if not pd.isna(price_series):
                                        current_prices[ticker] = price_series
                                    else:
                                        # print(f"Warning: No price data (asof) for {ticker} at {current_date_str}")
                                        missing_data_for_timestamp = True
                                        break
                            else:
                                # If exact timestamp not found, try asof - this implies data might be sparse for some tickers
                                price_series = df_ticker["close"].asof(current_datetime_obj_localized)
                                if not pd.isna(price_series):
                                    current_prices[ticker] = price_series
                                else:
                                    # print(f"Warning: No price data (asof) for {ticker} at {current_date_str} (exact timestamp not found)")
                                    missing_data_for_timestamp = True
                                    break
                        else:
                            # print(f"Warning: No preloaded data for {ticker} at {current_date_str}")
                            missing_data_for_timestamp = True
                            break
                    else: # Daily backtesting price fetching
                        # For daily, get_price_data fetches for a [previous_date, current_date] range.
                        # The agent decision is for current_date, based on data up to previous_date.
                        # Trades execute at current_date's close.
                        # We need the closing price of current_date_str for execution and valuation.
                        # For simplicity and consistency, let's assume get_price_data called with current_date_str for both start and end
                        # will give us the data for that specific day, and we can take the close.
                        # The original logic used previous_date to current_date which might be for agent lookback within get_price_data.
                        # Let's fetch for current_date_str to ensure we get today's close for execution.
                        price_data_df = get_price_data(ticker, current_date_str, current_date_str) # Fetch for the specific day
                        if price_data_df.empty or pd.isna(price_data_df.iloc[-1]["close"]):
                            # print(f"Warning: No closing price data for {ticker} on {current_date_str}")
                            missing_data_for_timestamp = True
                            break
                        current_prices[ticker] = price_data_df.iloc[-1]["close"]

                if missing_data_for_timestamp or len(current_prices) != len(self.tickers):
                    # print(f"Skipping trading point {current_date_str} due to missing price data for one or more tickers.")
                    # If data is missing for a timestamp, we should ideally record the portfolio value with last known prices
                    # but not execute new trades or run the agent. For now, we skip the agent call.
                    # Let's ensure all tickers have a price, even if it's carried forward, before calculating portfolio value.
                    # This part needs careful handling of how to proceed when some tickers have data and others don't at a specific timestamp.
                    # For now, if any data is missing, we might just skip this timestamp for agent decisions.
                    # However, portfolio valuation should ideally still happen.
                    # A simple skip for now:
                    if missing_data_for_timestamp:
                        print(f"Skipping agent decision at {current_date_str} due to missing price data for one or more tickers.")
                        # Still attempt to log portfolio value if possible
                        # This needs robust handling of potentially stale prices if we carry them forward.
                        # For now, if crucial data is missing, we might not be ableable to accurately value.
                        # Let's assume if current_prices is not fully populated, we can't reliably value or trade.
                        if len(current_prices) == len(self.tickers):
                             total_value_at_skip = self.calculate_portfolio_value(current_prices)
                             self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": total_value_at_skip})
                        else:
                            print(f"Cannot reliably value portfolio at {current_date_str} due to insufficient price data. Carrying forward last value.")
                            if self.portfolio_values:
                                self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": self.portfolio_values[-1]["Portfolio Value"]})
                            else:
                                self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": self.initial_capital})
                        continue # Skip agent and trading if critical price data is missing

            except Exception as e:
                # If there's a general API error or other error fetching prices for this timestamp
                print(f"Error fetching prices for {current_date_str}: {e}. Skipping this point.")
                if self.portfolio_values: # Carry forward last known portfolio value
                    self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": self.portfolio_values[-1]["Portfolio Value"]})
                else:
                    self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": self.initial_capital})
                continue

            # 1) Execute the agent's trades
            # ---------------------------------------------------------------
            # Agent's view of dates: agent_lookback_start_date and agent_current_date
            output = self.agent(
                tickers=self.tickers,
                start_date=agent_lookback_start_date, # Agent's historical lookback start (e.g., T-30 days)
                end_date=agent_current_date,      # Agent's current decision point (e.g., current day for daily, or current day for intraday)
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
            )
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            # Execute trades for each ticker
            executed_trades = {}
            for ticker in self.tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action, quantity = decision.get("action", "hold"), decision.get("quantity", 0)

                executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_quantity

            # ---------------------------------------------------------------
            # 2) Now that trades have executed trades, recalculate the final
            #    portfolio value for this day.
            # ---------------------------------------------------------------
            total_value = self.calculate_portfolio_value(current_prices)

            # Also compute long/short exposures for final post‐trade state
            long_exposure = sum(self.portfolio["positions"][t]["long"] * current_prices[t] for t in self.tickers)
            short_exposure = sum(self.portfolio["positions"][t]["short"] * current_prices[t] for t in self.tickers)

            # Calculate gross and net exposures
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = long_exposure / short_exposure if short_exposure > 1e-9 else float("inf")

            # Track each day's portfolio value in self.portfolio_values
            self.portfolio_values.append({"Date": current_datetime_obj, "Portfolio Value": total_value, "Long Exposure": long_exposure, "Short Exposure": short_exposure, "Gross Exposure": gross_exposure, "Net Exposure": net_exposure, "Long/Short Ratio": long_short_ratio})

            # ---------------------------------------------------------------
            # 3) Build the table rows to display
            # ---------------------------------------------------------------
            date_rows = []

            # For each ticker, record signals/trades
            for ticker in self.tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Calculate net position value
                pos = self.portfolio["positions"][ticker]
                # Ensure current_prices[ticker] exists, otherwise, this day/timestamp might have been skipped for this ticker
                price_for_valuation = current_prices.get(ticker)
                if price_for_valuation is None:
                    # This should ideally not happen if we skipped the agent call when data was missing.
                    # If it does, we need a fallback or error. For now, assume 0 if price is missing.
                    print(f"Critical Error: Price for {ticker} missing at {current_date_str} during table row generation.")
                    long_val = 0
                    short_val = 0
                else:
                    long_val = pos["long"] * price_for_valuation
                    short_val = pos["short"] * price_for_valuation
                net_position_value = long_val - short_val

                # Get the action and quantity from the decisions
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)

                # Append the agent action to the table rows
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices.get(ticker, 0), # Use .get for safety
                        shares_owned=pos["long"] - pos["short"],  # net shares
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
            # ---------------------------------------------------------------
            # 4) Calculate performance summary metrics
            # ---------------------------------------------------------------
            # Calculate portfolio return vs. initial capital
            # The realized gains are already reflected in cash balance, so we don't add them separately
            portfolio_return = (total_value / self.initial_capital - 1) * 100

            # Add summary row for this day
            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                ),
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            # Update performance metrics if we have enough data
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)

        # Store the final performance metrics for reference in analyze_performance
        self.performance_metrics = performance_metrics
        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Helper method to update performance metrics using daily returns."""
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        clean_returns = values_df["Daily Return"].dropna()

        if len(clean_returns) < 2:
            return  # not enough data points

        # Assumes 252 trading days/year
        daily_risk_free_rate = 0.0434 / 252
        excess_returns = clean_returns - daily_risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Sharpe ratio
        if std_excess_return > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
        else:
            performance_metrics["sharpe_ratio"] = 0.0

        # Sortino ratio
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float("inf") if mean_excess_return > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float("inf") if mean_excess_return > 0 else 0

        # Maximum drawdown (ensure it's stored as a negative percentage)
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max

        if len(drawdown) > 0:
            min_drawdown = drawdown.min()
            # Store as a negative percentage
            performance_metrics["max_drawdown"] = min_drawdown * 100

            # Store the date of max drawdown for reference
            if min_drawdown < 0:
                performance_metrics["max_drawdown_date"] = drawdown.idxmin().strftime("%Y-%m-%d")
            else:
                performance_metrics["max_drawdown_date"] = None
        else:
            performance_metrics["max_drawdown"] = 0.0
            performance_metrics["max_drawdown_date"] = None

    def analyze_performance(self):
        """Creates a performance DataFrame, prints summary stats, and plots equity curve."""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if performance_df.empty:
            print("No valid performance data to analyze.")
            return performance_df

        final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")

        # Print realized P&L for informational purposes only
        total_realized_gains = sum(self.portfolio["realized_gains"][ticker]["long"] + self.portfolio["realized_gains"][ticker]["short"] for ticker in self.tickers)
        print(f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}${total_realized_gains:,.2f}{Style.RESET_ALL}")

        # Plot the portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 252  # daily risk-free rate
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        # Annualized Sharpe Ratio
        if std_daily_return != 0:
            annualized_sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            annualized_sharpe = 0
        print(f"\nSharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

        # Use the max drawdown value calculated during the backtest if available
        max_drawdown = getattr(self, "performance_metrics", {}).get("max_drawdown")
        max_drawdown_date = getattr(self, "performance_metrics", {}).get("max_drawdown_date")

        # If no value exists yet, calculate it
        if max_drawdown is None:
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            max_drawdown_date = drawdown.idxmin().strftime("%Y-%m-%d") if pd.notnull(drawdown.idxmin()) else None

        if max_drawdown_date:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL} (on {max_drawdown_date})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}")

        # Win Rate
        winning_days = len(performance_df[performance_df["Daily Return"] > 0])
        total_days = max(len(performance_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

        # Average Win/Loss Ratio
        positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        if avg_loss != 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float("inf") if avg_win > 0 else 0
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

        # Maximum Consecutive Wins / Losses
        returns_binary = (performance_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0

        print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

        return performance_df


### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Margin ratio for short positions, e.g. 0.5 for 50% (default: 0.0)",
    )
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    parser.add_argument(
        "--interval",
        type=str,
        default="day",
        help="Time interval for price data (e.g., second, minute, hour, day, week, month, year)",
    )
    parser.add_argument(
        "--interval-multiplier",
        type=int,
        default=1,
        help="Multiplier for the interval (e.g., 5 for 5 minutes)",
    )

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else []

    # Choose analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: " f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}")

    # Select LLM model based on whether Ollama is being used
    model_name = ""
    model_provider = None

    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

        # Select from Ollama-specific models
        model_name = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        # Ensure Ollama is installed, running, and the model is available
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        # Use the standard cloud-based LLM selection
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        
        model_name, model_provider = model_choice

        model_info = get_model_info(model_name, model_provider)
        if model_info:
            if model_info.is_custom():
                model_name = questionary.text("Enter the custom model name:").ask()
                if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)

            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # Create and run the backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
        interval=args.interval,
        interval_multiplier=args.interval_multiplier,
    )

    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()
