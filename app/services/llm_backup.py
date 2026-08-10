import os
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import chromadb
import fitz
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ============================================================
# GEMINI
# ============================================================

model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        generation_config={
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
        },
    )

# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="financevision_knowledge_base"
)

# ============================================================
# TEXT / DOCUMENT PROCESSING
# ============================================================

def chunk_text_with_overlap(text: str, chunk_size: int = 300, overlap: int = 50):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks


def process_uploaded_file(file_path: str, filename: str) -> str:
    try:
        text_chunks = []

        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
            row_texts = []
            for index, row in df.iterrows():
                row_texts.append(
                    f"Row {index + 1}: "
                    + ", ".join(f"{col}: {val}" for col, val in row.items())
                )
            for i in range(0, len(row_texts), 10):
                group = row_texts[i:i + 10]
                text_chunks.append(
                    f"[Source: {filename}, Section: Rows {i + 1}-{i + len(group)}]\n"
                    + "\n".join(group)
                )

        elif filename.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page_text = doc.load_page(page_num).get_text("text")
                if page_text.strip():
                    for chunk in chunk_text_with_overlap(page_text, 250, 40):
                        text_chunks.append(
                            f"[Source: {filename}, Page: {page_num + 1}]\n{chunk}"
                        )
            doc.close()
        else:
            return "Unsupported file type. Please upload CSV or PDF."

        if not text_chunks:
            return "No readable text found in document."

        for i in range(0, len(text_chunks), 500):
            batch = text_chunks[i:i + 500]
            timestamp = int(time.time())
            ids = [f"{filename}_chunk_{i + j}_{timestamp}" for j in range(len(batch))]
            collection.add(documents=batch, ids=ids)

        return "Success"
    except Exception as e:
        return str(e)

# ============================================================
# DETERMINISTIC MATH
# ============================================================

def calculate_cagr(start_value: float, end_value: float, periods: float) -> str:
    if start_value <= 0 or periods <= 0:
        return "Invalid parameters for CAGR calculation."
    cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100
    total_growth = ((end_value - start_value) / start_value) * 100
    return (
        "### Deterministic CAGR Valuation Result\n"
        f"- **Initial Value:** {start_value:,.2f}\n"
        f"- **Final Value:** {end_value:,.2f}\n"
        f"- **Time Horizon:** {periods} Years\n"
        f"- **Compound Annual Growth Rate (CAGR):** **{cagr:.2f}%**\n"
        f"- **Total Return:** **{total_growth:.2f}%**\n"
    )


def calculate_dcf(initial_fcf: float, growth_rate: float, discount_rate: float,
                  terminal_growth: float, years: int = 5) -> str:
    try:
        if discount_rate <= terminal_growth:
            return "DCF Calculation Error: Discount rate must be greater than terminal growth rate."
        pv_cf = []
        current_fcf = initial_fcf
        for yr in range(1, years + 1):
            current_fcf *= 1 + growth_rate / 100
            pv_cf.append(current_fcf / ((1 + discount_rate / 100) ** yr))
        terminal_value = (current_fcf * (1 + terminal_growth / 100)) / ((discount_rate - terminal_growth) / 100)
        pv_terminal_value = terminal_value / ((1 + discount_rate / 100) ** years)
        enterprise_value = sum(pv_cf) + pv_terminal_value
        return (
            "### Deterministic DCF Intrinsic Valuation\n"
            f"- **Starting Free Cash Flow (FCF):** ${initial_fcf:,.2f}M\n"
            f"- **Projected Growth Rate ({years} Yrs):** {growth_rate}%\n"
            f"- **Discount Rate (WACC):** {discount_rate}%\n"
            f"- **Terminal Growth Rate:** {terminal_growth}%\n"
            f"- **Sum of PV Cash Flows:** ${sum(pv_cf):,.2f}M\n"
            f"- **PV of Terminal Value:** ${pv_terminal_value:,.2f}M\n"
            f"- **Estimated Enterprise Intrinsic Value:** **${enterprise_value:,.2f} Million**\n"
        )
    except Exception as e:
        return f"DCF Calculation Error: {e}"


def calculate_emi(principal: float, annual_rate: float, tenure_years: float) -> str:
    monthly_rate = (annual_rate / 100) / 12
    tenure_months = int(tenure_years * 12)
    if tenure_months <= 0:
        return "Invalid loan tenure."
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
    total_payable = emi * tenure_months
    total_interest = total_payable - principal
    return (
        "### Deterministic Loan/EMI Amortization\n"
        f"- **Principal Loan Amount:** ${principal:,.2f}\n"
        f"- **Annual Interest Rate:** {annual_rate}%\n"
        f"- **Loan Tenure:** {tenure_years} Years ({tenure_months} Months)\n"
        f"- **Monthly EMI Payment:** **${emi:,.2f}**\n"
        f"- **Total Interest Payable:** ${total_interest:,.2f}\n"
        f"- **Total Payment Amount:** ${total_payable:,.2f}\n"
    )


def parse_agentic_math(user_query: str) -> str:
    q = user_query.lower()
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)

    if "cagr" in q or "compound annual" in q:
        if len(nums) >= 3:
            try:
                return calculate_cagr(float(nums[0]), float(nums[1]), float(nums[2]))
            except Exception:
                pass

    if "dcf" in q or "discounted cash flow" in q or "intrinsic value" in q:
        if len(nums) >= 3:
            try:
                return calculate_dcf(
                    float(nums[0]), float(nums[1]), float(nums[2]),
                    float(nums[3]) if len(nums) > 3 else 2.5,
                )
            except Exception:
                pass

    if "emi" in q or "loan" in q or "amortization" in q:
        if len(nums) >= 2:
            try:
                return calculate_emi(
                    float(nums[0]), float(nums[1]),
                    float(nums[2]) if len(nums) > 2 else 5.0,
                )
            except Exception:
                pass
    return ""

# ============================================================
# YAHOO HELPERS
# ============================================================

COMMODITIES = {
    "gold": {"ticker": "GC=F", "name": "Gold", "unit": "USD per troy ounce"},
    "silver": {"ticker": "SI=F", "name": "Silver", "unit": "USD per troy ounce"},
    "copper": {"ticker": "HG=F", "name": "Copper", "unit": "USD per pound"},
    "platinum": {"ticker": "PL=F", "name": "Platinum", "unit": "USD per troy ounce"},
    "palladium": {"ticker": "PA=F", "name": "Palladium", "unit": "USD per troy ounce"},
}

GLOBAL_OVERRIDES = {
    "sensex": "^BSESN", "nifty": "^NSEI", "itc": "ITC.NS", "hdfc": "HDFCBANK.NS",
    "sbi": "SBIN.NS", "tata motors": "TATAMOTORS.NS", "capgemini": "CAP.PA",
    "tcs": "TCS.NS", "tata power": "TATAPOWER.NS", "tat power": "TATAPOWER.NS",
    "tata-power": "TATAPOWER.NS", "reliance": "RELIANCE.NS", "tesla": "TSLA",
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amd": "AMD",
    "intel": "INTC", "tsmc": "TSM", "tata steel": "TATASTEEL.NS", "jsw steel": "JSWSTEEL.NS",
    "jindal steel": "JINDALSTEL.NS", "sail": "SAIL.NS", "posco": "005490.KS",
    "arcelormittal": "MT", "infosys": "INFY.NS", "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
}


def get_latest_yahoo_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        try:
            value = stock.fast_info.get("last_price")
            if value is not None and pd.notna(value):
                return float(value)
        except Exception:
            pass
        try:
            history = stock.history(period="5d", auto_adjust=False)
            if not history.empty:
                history = history.dropna(subset=["Close"])
                if not history.empty:
                    return float(history["Close"].iloc[-1])
        except Exception:
            pass
        try:
            history = stock.history(period="1d", interval="1m", auto_adjust=False)
            if not history.empty:
                history = history.dropna(subset=["Close"])
                if not history.empty:
                    return float(history["Close"].iloc[-1])
        except Exception:
            pass
    except Exception as e:
        print(f"Yahoo price error for {ticker}: {e}")
    return None


def get_yahoo_history(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if history is None or history.empty:
            return pd.DataFrame()
        return history.dropna(subset=["Close"])
    except Exception as e:
        print(f"Yahoo history error for {ticker}: {e}")
        return pd.DataFrame()


def _extract_explicit_tickers(query: str):
    result = []
    for raw in re.findall(r"(?<![A-Za-z0-9_=^.-])\^?[A-Z]{2,10}(?:\.(?:NS|BO|PA|KS|L|DE))?(?:=F)?(?![A-Za-z0-9_.-])", query):
        if raw.upper() not in result:
            result.append(raw.upper())
    return result


def resolve_tickers(query: str):
    q = query.lower()
    tickers = []

    for key, ticker in sorted(GLOBAL_OVERRIDES.items(), key=lambda x: len(x[0]), reverse=True):
        if key in q and ticker not in tickers:
            tickers.append(ticker)

    for key, meta in COMMODITIES.items():
        if re.search(rf"\b{re.escape(key)}\b", q) and meta["ticker"] not in tickers:
            tickers.append(meta["ticker"])

    for ticker in _extract_explicit_tickers(query):
        if ticker not in tickers:
            tickers.append(ticker)

    if not tickers:
        # Keep Yahoo symbol discovery as a fallback for unfamiliar companies.
        ignored = {
            "what", "is", "the", "price", "of", "show", "me", "chart", "for", "stock", "draw",
            "interactive", "maximum", "timeframe", "from", "launch", "and", "or", "to", "in", "on",
            "a", "an", "all", "time", "history", "trend", "give", "send", "detail", "details", "about",
            "company", "world", "market", "calculate", "total", "budget", "combined", "math", "all-time",
            "invest", "buy", "average", "month", "previous", "mean", "median", "difference", "comparison",
            "bar", "graph", "worth", "net", "till", "now", "today", "current", "live", "please", "tell",
            "detailed", "analysis", "compare", "versus", "vs", "year", "past", "last", "profit", "revenue",
        }
        terms = [w.strip(",.!?()[]{}") for w in query.split() if w.lower() not in ignored and len(w) > 2]
        if terms:
            try:
                url = "https://query2.finance.yahoo.com/v1/finance/search?q=" + urllib.parse.quote(" ".join(terms[:3]))
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    for quote in data.get("quotes", [])[:3]:
                        symbol = quote.get("symbol")
                        if symbol and symbol not in tickers:
                            tickers.append(symbol)
            except Exception as e:
                print(f"Yahoo symbol search failed: {e}")
    return tickers[:5]


def _get_usdinr():
    value = get_latest_yahoo_price("INR=X")
    return value


def _gold_inr_per_10g(usd_per_oz: float):
    usd_inr = _get_usdinr()
    if usd_inr is None:
        return None, None
    return usd_per_oz * usd_inr * 10 / 31.1034768, usd_inr


def _clean_number(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _nearest_one_year_ago(hist: pd.DataFrame):
    if hist.empty:
        return None, None
    target = pd.Timestamp.now(tz=hist.index.tz) - pd.Timedelta(days=365) if getattr(hist.index, "tz", None) else pd.Timestamp.now() - pd.Timedelta(days=365)
    idx = hist.index.get_indexer([target], method="nearest")[0]
    return hist.index[idx], float(hist["Close"].iloc[idx])


def _market_snapshot(ticker: str):
    stock = yf.Ticker(ticker)
    current = get_latest_yahoo_price(ticker)
    if current is None:
        return None

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    currency = info.get("currency") or ("INR" if ".NS" in ticker or ".BO" in ticker else "USD")
    name = info.get("shortName") or info.get("longName") or ticker
    hist = get_yahoo_history(ticker, "1y", "1d")
    one_year_date, one_year_price = _nearest_one_year_ago(hist)

    previous_close = None
    try:
        h5 = stock.history(period="5d", auto_adjust=False).dropna(subset=["Close"])
        if len(h5) >= 2:
            previous_close = float(h5["Close"].iloc[-2])
    except Exception:
        pass

    change = current - previous_close if previous_close is not None else None
    pct = (change / previous_close * 100) if previous_close not in (None, 0) else None

    return {
        "ticker": ticker,
        "name": name,
        "currency": currency,
        "current": current,
        "previous_close": previous_close,
        "change": change,
        "pct_change": pct,
        "history": hist,
        "one_year_date": one_year_date,
        "one_year_price": one_year_price,
        "market_cap": _clean_number(info.get("marketCap")),
        "revenue": _clean_number(info.get("totalRevenue")),
        "gross_profit": _clean_number(info.get("grossProfits")),
        "net_income": _clean_number(info.get("netIncomeToCommon")),
        "eps": _clean_number(info.get("trailingEps")),
        "pe": _clean_number(info.get("trailingPE")),
        "debt_to_equity": _clean_number(info.get("debtToEquity")),
        "high_52": _clean_number(info.get("fiftyTwoWeekHigh")),
        "low_52": _clean_number(info.get("fiftyTwoWeekLow")),
    }

# ============================================================
# INTENT / CHART CONTROL
# ============================================================

CHART_WORDS = (
    "draw chart", "show chart", "display chart", "plot", "graph", "line chart", "price trend",
    "price history", "historical price", "time series", "visualize", "visualise", "chart for",
)
COMPARE_WORDS = (
    "compare", "comparison", "vs", "versus", "difference between", "year over year", "year-over-year",
)


def detect_intent(query: str):
    q = query.lower().strip()
    chart_requested = any(p in q for p in CHART_WORDS)
    comparison_requested = any(re.search(rf"\b{re.escape(p)}\b", q) for p in COMPARE_WORDS)
    return {
        "chart_requested": chart_requested,
        "comparison_requested": comparison_requested,
        "tickers": resolve_tickers(query),
    }


def _format_live_context(snapshots):
    if not snapshots:
        return "No verified live market data was retrieved."

    parts = []
    for s in snapshots:
        parts.append(f"""
--- VERIFIED MARKET DATA: {s['name']} ({s['ticker']}) ---
Source: Yahoo Finance
Latest available price: {s['currency']} {s['current']:.6f}
Previous close: {s['currency']} {s['previous_close']:.6f} if available
Change: {s['currency']} {s['change']:.6f} if available
Change percent: {s['pct_change']:.4f}% if available
One-year comparison date: {s['one_year_date']}
One-year comparison price: {s['currency']} {s['one_year_price']:.6f} if available
Market cap: {s['market_cap']}
Revenue: {s['revenue']}
Gross profit: {s['gross_profit']}
Net income: {s['net_income']}
EPS: {s['eps']}
P/E: {s['pe']}
Debt-to-equity: {s['debt_to_equity']}
52-week high: {s['high_52']}
52-week low: {s['low_52']}
""")
    return "\n".join(parts)


def fetch_live_stock_data(query: str) -> str:
    tickers = resolve_tickers(query)
    snapshots = []
    for ticker in tickers:
        try:
            # Commodity futures are handled separately because their quoted units differ.
            commodity = next((v for v in COMMODITIES.values() if v["ticker"] == ticker), None)
            if commodity:
                price = get_latest_yahoo_price(ticker)
                if price is None:
                    continue
                text = (
                    f"\n--- VERIFIED COMMODITY DATA: {commodity['name']} ({ticker}) ---\n"
                    f"Source: Yahoo Finance\n"
                    f"Latest available international price: USD {price:.6f} ({commodity['unit']})\n"
                )
                if ticker == "GC=F":
                    inr_10g, fx = _gold_inr_per_10g(price)
                    if inr_10g is not None:
                        text += f"USD/INR: {fx:.6f}\nIndicative INR value per 10g: INR {inr_10g:.2f}\n"
                        text += "This is a market conversion, not a local jeweller quote.\n"
                hist = get_yahoo_history(ticker, "1y", "1d")
                if not hist.empty:
                    _, one_year_price = _nearest_one_year_ago(hist)
                    text += f"One-year comparison price: USD {one_year_price:.6f}\n"
                snapshots.append(text)
                continue

            snap = _market_snapshot(ticker)
            if snap:
                snapshots.append(_format_live_context([snap]))
        except Exception as e:
            print(f"Market processing failed for {ticker}: {e}")
    return "\n".join(snapshots) if snapshots else "No verified live market data was retrieved."


def _build_deterministic_chart(query: str, intent: dict):
    tickers = intent["tickers"]
    q = query.lower()

    # Explicit historical comparison: one asset, e.g. "compare gold from past year till now".
    if intent["comparison_requested"] and len(tickers) == 1:
        ticker = tickers[0]
        commodity = next((v for v in COMMODITIES.values() if v["ticker"] == ticker), None)
        if commodity:
            hist = get_yahoo_history(ticker, "1y", "1d")
            current = get_latest_yahoo_price(ticker)
            if current is not None and not hist.empty:
                _, old = _nearest_one_year_ago(hist)
                if ticker == "GC=F" and ("rupee" in q or "inr" in q or "india" in q):
                    current_inr, fx = _gold_inr_per_10g(current)
                    # Convert the historical USD gold value using today's FX only when a historical FX series is not requested.
                    old_inr = old * fx * 10 / 31.1034768 if fx else None
                    if current_inr is not None and old_inr is not None:
                        return f"[COMPARE_CHART|Gold Price Comparison (INR per 10g)|1 Year Ago:{old_inr:.2f}|Today:{current_inr:.2f}]"
                return f"[COMPARE_CHART|{commodity['name']} Price Comparison ({commodity['unit']})|1 Year Ago:{old:.2f}|Today:{current:.2f}]"

        snap = _market_snapshot(ticker)
        if snap and snap["one_year_price"] is not None:
            metric = "price"
            if "revenue" in q:
                metric = "revenue"
            elif "profit" in q or "net income" in q:
                metric = "net_income"
            elif "market cap" in q or "market capitalization" in q:
                metric = "market_cap"
            value_old = snap["one_year_price"] if metric == "price" else snap.get(metric)
            value_now = snap["current"] if metric == "price" else snap.get(metric)
            if value_old is not None and value_now is not None:
                return f"[COMPARE_CHART|{snap['name']} {metric.replace('_', ' ').title()} Comparison|1 Year Ago:{value_old:.2f}|Today:{value_now:.2f}]"

    # Explicit comparison between two or more assets.
    if intent["comparison_requested"] and len(tickers) >= 2:
        snapshots = [_market_snapshot(t) for t in tickers]
        snapshots = [s for s in snapshots if s]
        if len(snapshots) >= 2:
            metric = "current"
            unit = "Price"
            if "market cap" in q or "market capitalization" in q:
                metric, unit = "market_cap", "Market Capitalization"
            elif "revenue" in q or "sales" in q:
                metric, unit = "revenue", "Revenue"
            elif "gross profit" in q:
                metric, unit = "gross_profit", "Gross Profit"
            elif "profit" in q or "net income" in q:
                metric, unit = "net_income", "Net Income"
            elif "eps" in q:
                metric, unit = "eps", "EPS"
            elif "p/e" in q or "pe ratio" in q:
                metric, unit = "pe", "P/E Ratio"

            pairs = []
            for s in snapshots:
                value = s.get(metric)
                if value is not None:
                    pairs.append(f"{s['name']}:{value:.2f}")
            if len(pairs) >= 2:
                return f"[COMPARE_CHART|{unit} Comparison|{'|'.join(pairs)}]"

    return ""


def _strip_chart_tokens(text: str):
    # If a chart was not explicitly requested, never allow the LLM to create one.
    return re.sub(r"\[(?:INTERACTIVE_CHART|COMPARE_CHART)\|.*?\]", "", text, flags=re.I | re.S)

# ============================================================
# FINANCIAL RESPONSE
# ============================================================

def _retrieve_rag(query: str, intent: dict):
    q = query.lower()
    # Market-only requests should not pull random company documents into the prompt.
    rag_keywords = (
        "uploaded", "document", "report", "pdf", "knowledge base", "from the file", "from my file",
        "according to", "in the report", "our company", "company data", "what does the document",
    )
    should_rag = any(k in q for k in rag_keywords) or not intent["tickers"]
    if not should_rag or collection.count() <= 0:
        return "No specific document data found in the knowledge base."
    try:
        results = collection.query(query_texts=[query], n_results=5)
        docs = results.get("documents") or []
        if docs and docs[0]:
            return "\n...\n".join(docs[0])
    except Exception as e:
        print(f"ChromaDB query failed: {e}")
    return "No specific document data found in the knowledge base."


def _system_prompt(user_query, user_profile, live_data, document_context, math_context):
    personal_context = ""
    if user_profile and (user_profile.get("professional_role") or user_profile.get("focus_area")):
        personal_context = (
            "[USER PROFILE]\n"
            f"Name: {user_profile.get('full_name', 'User')}\n"
            f"Professional Role: {user_profile.get('professional_role', 'Unknown')}\n"
            f"Primary Focus: {user_profile.get('focus_area', 'Unknown')}\n"
        )

    return f"""
You are FinanceVision AI, a financial and corporate RAG assistant.

{personal_context}

[DETERMINISTIC MATH]
{math_context or 'None'}
[/DETERMINISTIC MATH]

[VERIFIED MARKET DATA]
{live_data}
[/VERIFIED MARKET DATA]

[UPLOADED COMPANY KNOWLEDGE BASE]
{document_context}
[/UPLOADED COMPANY KNOWLEDGE BASE]

MANDATORY RULES:
1. Stay within finance, markets, investing, companies, economics, financial documents and corporate analysis.
2. NEVER invent a current market price. Use verified market data when supplied.
3. If verified market data is unavailable, say so instead of guessing.
4. For uploaded documents, cite the supplied source/page/section when the source metadata is available.
5. A normal price request returns text only. DO NOT create a chart unless the user explicitly asks for a chart, graph, plot, trend, price history, or visualization.
6. A normal company/fundamental analysis returns analysis only. DO NOT add any chart unless explicitly requested.
7. A comparison request may use a comparison chart, but only use the deterministic numbers supplied by the application. Do not invent chart values.
8. Historical/time-series price requests use a LINE chart. Explicit comparisons use a BAR chart.
9. Do not output [INTERACTIVE_CHART] or [COMPARE_CHART] yourself. The application will append the correct verified chart token when required.
10. Keep the answer focused on exactly what the user asked. Do not add unrelated market data, extra charts, or unrelated company sections.
11. For investment analysis, clearly separate facts from interpretation and mention important risks. Do not present predictions as guaranteed outcomes.
12. For gold/silver/metal prices, state the quoted unit and distinguish international market price from an indicative INR conversion or local retail/jeweller price.

USER QUERY:
{user_query}
"""


def stream_financial_response(user_query: str, user_profile: dict = None):
    intent = detect_intent(user_query)
    math_result = parse_agentic_math(user_query)
    live_data = fetch_live_stock_data(user_query)
    document_context = _retrieve_rag(user_query, intent)
    system_prompt = _system_prompt(user_query, user_profile, live_data, document_context, math_result)

    # Deterministic chart token is generated outside the LLM.
    chart_token = ""
    if intent["chart_requested"] and intent["tickers"]:
        chart_token = f"[INTERACTIVE_CHART|{intent['tickers'][0]}|line]"
    elif intent["comparison_requested"]:
        chart_token = _build_deterministic_chart(user_query, intent)

    emitted_any = False

    if model is not None:
        try:
            response = model.generate_content(system_prompt, stream=True)
            for chunk in response:
                text = getattr(chunk, "text", "") or ""
                if text:
                    emitted_any = True
                    # Defense-in-depth: remove any chart instruction generated by the LLM.
                    yield _strip_chart_tokens(text)
            if chart_token:
                yield "\n\n" + chart_token
            return
        except Exception as e:
            print(f"Gemini error: {e}")

    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                "stream": True,
            }
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            data = json.loads(decoded[6:])
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                emitted_any = True
                                yield _strip_chart_tokens(delta)
                        except Exception:
                            pass
                if chart_token:
                    yield "\n\n" + chart_token
                return
            print(f"Groq error: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Groq fallback failed: {e}")

    try:
        import ollama
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            stream=True,
        )
        for chunk in response:
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield _strip_chart_tokens(text)
        if chart_token:
            yield "\n\n" + chart_token
    except Exception as e:
        yield f"\n\n**All AI Engines Failed.**\nDetails: {e}"

# ============================================================
# MARKET OVERVIEW
# ============================================================

def _overview_item(ticker, name):
    current = get_latest_yahoo_price(ticker)
    if current is None:
        return None
    previous_close = None
    try:
        history = yf.Ticker(ticker).history(period="5d", auto_adjust=False).dropna(subset=["Close"])
        if len(history) >= 2:
            previous_close = float(history["Close"].iloc[-2])
    except Exception:
        pass
    if previous_close not in (None, 0):
        change = current - previous_close
        pct = change / previous_close * 100
    else:
        change = 0.0
        pct = 0.0
    sentiment = "BULLISH" if pct > 1 else "BEARISH" if pct < -1 else "NEUTRAL"
    return {
        "name": name, "ticker": ticker, "price": f"{current:.2f}",
        "change": round(change, 2), "pct_change": round(pct, 2),
        "sentiment": sentiment,
        "news": "Latest available Yahoo Finance market data.",
    }


def get_market_overview():
    assets = [
        ("^BSESN", "SENSEX"),
        ("TATAPOWER.NS", "Tata Power"),
        ("TSLA", "Tesla"),
        ("AAPL", "Apple"),
    ]
    data = []
    for ticker, name in assets:
        try:
            item = _overview_item(ticker, name)
            if item:
                data.append(item)
        except Exception as e:
            print(f"Market overview error for {ticker}: {e}")
    return data

# ============================================================
# GLOBAL INDICES
# ============================================================

def get_global_indices():
    indices = [
        ("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ"),
        ("^N225", "Nikkei 225"), ("^FTSE", "FTSE 100"), ("^NSEI", "NIFTY 50"),
        ("^BSESN", "SENSEX"),
    ]
    data = []
    for ticker, name in indices:
        try:
            item = _overview_item(ticker, name)
            if item:
                data.append(item)
        except Exception as e:
            print(f"Global index error for {ticker}: {e}")
    return data
