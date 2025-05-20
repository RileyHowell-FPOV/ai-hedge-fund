import datetime
import os
import pandas as pd
import requests
import time
import re

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    # Convert start_date and end_date to datetime objects for comparison if they are not already
    # This is a simplified approach; robust date parsing might be needed depending on timestamp format from API for sub-daily data
    start_dt_str = start_date
    end_dt_str = end_date
    if "T" not in start_date: # implies it's just a date, not a datetime
        start_dt_str = start_date + "T00:00:00"
    if "T" not in end_date:
        end_dt_str = end_date + "T23:59:59"

    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        # Ensure 'time' in cached_data is comparable with start_dt_str and end_dt_str
        filtered_data = [
            Price(**price) for price in cached_data 
            if start_dt_str <= price["time"] <= end_dt_str
        ]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    
    max_retries = 5
    base_delay = 2  # seconds, Increased from 1 to 2

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            price_response = PriceResponse(**response.json())
            prices = price_response.prices
            if not prices:
                return []
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            return prices
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                if retry_after_match:
                    parsed_delay = int(retry_after_match.group(1))
                    delay = max(parsed_delay, base_delay) # Ensure minimum delay
                    print(f"Rate limited for {ticker} prices. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                else:
                    delay = base_delay * (2**attempt)
                    print(f"Rate limited for {ticker} prices. Retrying after {delay} seconds (exponential backoff)...")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise Exception(f"Error fetching price data for {ticker} after {max_retries} retries: {e.response.status_code} - {e.response.text}") from e
            else:
                raise Exception(f"Error fetching price data for {ticker}: {e.response.status_code} - {e.response.text}") from e
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"Price request for {ticker} failed: {e}. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                raise Exception(f"Error fetching price data for {ticker} after {max_retries} retries: {e}") from e
    return [] # Should not be reached


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"

    max_retries = 5
    base_delay = 2  # seconds, Increased from 1 to 2

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            metrics_response = FinancialMetricsResponse(**response.json())
            financial_metrics = metrics_response.financial_metrics
            if not financial_metrics:
                return []
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
            return financial_metrics
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                if retry_after_match:
                    parsed_delay = int(retry_after_match.group(1))
                    delay = max(parsed_delay, base_delay) # Ensure minimum delay
                    print(f"Rate limited for {ticker} financial metrics. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                else:
                    delay = base_delay * (2**attempt)
                    print(f"Rate limited for {ticker} financial metrics. Retrying after {delay} seconds (exponential backoff)...")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise Exception(f"Error fetching financial metrics for {ticker} after {max_retries} retries: {e.response.status_code} - {e.response.text}") from e
            else:
                raise Exception(f"Error fetching financial metrics for {ticker}: {e.response.status_code} - {e.response.text}") from e
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"Financial metrics request for {ticker} failed: {e}. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                raise Exception(f"Error fetching financial metrics for {ticker} after {max_retries} retries: {e}") from e
    return [] # Should not be reached


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }

    max_retries = 5
    base_delay = 2  # seconds, Increased from 1 to 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            response_model = LineItemResponse(**data)
            search_results = response_model.search_results
            if not search_results:
                return []
            # Cache the results
            return search_results[:limit]

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                if retry_after_match:
                    parsed_delay = int(retry_after_match.group(1))
                    delay = max(parsed_delay, base_delay) # Ensure minimum delay
                    print(f"Rate limited for {ticker} prices. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                else:
                    delay = base_delay * (2**attempt)
                    print(f"Rate limited for {ticker} prices. Retrying after {delay} seconds (exponential backoff)...")
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise Exception(f"Error fetching data after {max_retries} retries: {ticker} - {e.response.status_code} - {e.response.text}") from e
            else:
                # For other HTTP errors, raise immediately
                raise Exception(f"Error fetching data: {ticker} - {e.response.status_code} - {e.response.text}") from e
        except requests.exceptions.RequestException as e:
            # For other request exceptions (e.g., network issues)
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"Request failed: {e}. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                raise Exception(f"Error fetching data after {max_retries} retries: {ticker} - {e}") from e

    return [] # Should not be reached if retries are exhausted and exception is raised


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date) and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date

    max_retries = 5 # Added for retry
    base_delay = 2  # seconds, Increased from 1 to 2

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        for attempt in range(max_retries): # Added for retry
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                response_model = InsiderTradeResponse(**data)
                insider_trades = response_model.insider_trades
                break # Break from retry loop on success
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                    if retry_after_match:
                        parsed_delay = int(retry_after_match.group(1))
                        delay = max(parsed_delay, base_delay) # Ensure minimum delay
                        print(f"Rate limited for {ticker} insider trades. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                    else:
                        delay = base_delay * (2**attempt)
                        print(f"Rate limited for {ticker} insider trades. Retrying after {delay} seconds (exponential backoff)...")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        raise Exception(f"Error fetching insider trades for {ticker} after {max_retries} retries: {e.response.status_code} - {e.response.text}") from e
                else:
                    raise Exception(f"Error fetching insider trades for {ticker}: {e.response.status_code} - {e.response.text}") from e
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(f"Insider trades request for {ticker} failed: {e}. Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Error fetching insider trades for {ticker} after {max_retries} retries: {e}") from e
        else: # Corresponds to for-loop, executed if loop completes normally (without break due to error)
            pass # Successful attempt, continue with logic below

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data if (start_date is None or news["date"] >= start_date) and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date

    max_retries = 5 # Added for retry
    base_delay = 2  # seconds, Increased from 1 to 2

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        for attempt in range(max_retries): # Added for retry
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status() # Added for retry

                data = response.json()
                response_model = CompanyNewsResponse(**data)
                company_news = response_model.news

                # Break from retry loop if successful
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                    if retry_after_match:
                        parsed_delay = int(retry_after_match.group(1))
                        delay = max(parsed_delay, base_delay) # Ensure minimum delay
                        print(f"Rate limited for {ticker} news. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                    else:
                        delay = base_delay * (2**attempt)
                        print(f"Rate limited for {ticker} news. Retrying after {delay} seconds (exponential backoff)...")
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        # If this was the last attempt for this specific API call, raise an error for this call
                        raise Exception(f"Error fetching news data for {ticker} after {max_retries} retries: {e.response.status_code} - {e.response.text}") from e
                else:
                    # For other HTTP errors, raise immediately
                    raise Exception(f"Error fetching news data for {ticker}: {e.response.status_code} - {e.response.text}") from e
            except requests.exceptions.RequestException as e:
                # For other request exceptions (e.g., network issues)
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(f"News request for {ticker} failed: {e}. Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    # If this was the last attempt for this specific API call, raise an error for this call
                    raise Exception(f"Error fetching news data for {ticker} after {max_retries} retries: {e}") from e
        else: # This else corresponds to the for loop, executed if the loop completed without a break
            # If all retries failed for a paginated call, we might want to raise or handle differently.
            # For now, this will effectively stop pagination for this specific call if it exhausts retries.
            # Or, we can re-raise the last error if needed by storing it outside the loop.
            # Given the current structure, if retries are exhausted, the exception from the last attempt will have already been raised.
            # This path might not be hit if the exception is always raised.
            # Let's ensure an exception is always raised if all retries fail.
            # The raise in the except block should prevent this else from being reached on failure.
            pass # Loop finished (should be due to break on success)

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        
        max_retries = 5
        base_delay = 2  # seconds, Increased from 1 to 2
        market_cap_from_facts = None

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                response_model = CompanyFactsResponse(**data)
                market_cap_from_facts = response_model.company_facts.market_cap
                break # Success
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after_match = re.search(r"Expected available in (\d+) seconds", e.response.text)
                    if retry_after_match:
                        parsed_delay = int(retry_after_match.group(1))
                        delay = max(parsed_delay, base_delay) # Ensure minimum delay
                        print(f"Rate limited for {ticker} company facts. Retrying after {delay} seconds (API suggested {parsed_delay}s)...")
                    else:
                        delay = base_delay * (2**attempt)
                        print(f"Rate limited for {ticker} company facts. Retrying after {delay} seconds (exponential backoff)...")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        # Log error and continue, as market_cap can be sourced from financial_metrics as a fallback
                        print(f"Error fetching company facts for {ticker} after {max_retries} retries: {e.response.status_code} - {e.response.text}. Will try fallback.")
                        break # Break to allow fallback
                else:
                    print(f"Error fetching company facts for {ticker}: {e.response.status_code} - {e.response.text}. Will try fallback.")
                    break # Break to allow fallback
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(f"Company facts request for {ticker} failed: {e}. Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Error fetching company facts for {ticker} after {max_retries} retries: {e}. Will try fallback.")
                    break # Break to allow fallback
        # If market_cap_from_facts was successfully fetched, return it
        if market_cap_from_facts is not None:
             return market_cap_from_facts
        # Otherwise, an error occurred and was printed, proceed to fallback or return None if already tried fallback.

    # Fallback or primary method if not today's date
    # The get_financial_metrics call already has retry logic internally.
    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, interval=interval, interval_multiplier=interval_multiplier)
    return prices_to_df(prices)
