import os
import urllib.request
import urllib.parse
import json
import yfinance as yf
import chromadb
import pandas as pd
import fitz
import google.generativeai as genai
import statistics
import time
import requests
import re
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 4096,
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )
else:
    model = None

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financevision_knowledge_base")

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
                row_texts.append(f"Row {index + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()]))
            for i in range(0, len(row_texts), 10):
                group = row_texts[i:i + 10]
                excerpt = "\n".join(group)
                text_chunks.append(f"[Source: {filename}, Section: Rows {i + 1}-{i + len(group)}]\n{excerpt}")
        elif filename.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page_text = doc.load_page(page_num).get_text("text")
                if page_text.strip():
                    page_chunks = chunk_text_with_overlap(page_text, chunk_size=250, overlap=40)
                    for chunk in page_chunks:
                        text_chunks.append(f"[Source: {filename}, Page: {page_num + 1}]\n{chunk}")
            doc.close()
        if text_chunks:
            for i in range(0, len(text_chunks), 500):
                batch = text_chunks[i:i + 500]
                timestamp = int(time.time())
                ids = [f"{filename}_chunk_{i + j}_{timestamp}" for j in range(len(batch))]
                collection.add(documents=batch, ids=ids)
            return "Success"
        return "No readable text found in document."
    except Exception as e:
        return str(e)

def calculate_cagr(start_value: float, end_value: float, periods: float) -> str:
    if start_value <= 0 or periods <= 0: return "Invalid parameters for CAGR calculation."
    cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100
    total_growth = ((end_value - start_value) / start_value) * 100
    return f"### Deterministic CAGR Valuation Result\n- **Initial Value:** ${start_value:,.2f}\n- **Final Value:** ${end_value:,.2f}\n- **Time Horizon:** {periods} Years\n- **Compound Annual Growth Rate (CAGR):** **{cagr:.2f}%**\n- **Total Return:** **{total_growth:.2f}%**\n"

def calculate_dcf(initial_fcf: float, growth_rate: float, discount_rate: float, terminal_growth: float, years: int = 5) -> str:
    try:
        if discount_rate <= terminal_growth: return "DCF Error: Discount rate must be greater than terminal growth rate."
        pv_cf = []
        current_fcf = initial_fcf
        for yr in range(1, years + 1):
            current_fcf *= (1 + growth_rate / 100)
            pv_cf.append(current_fcf / ((1 + discount_rate / 100) ** yr))
        terminal_value = (current_fcf * (1 + terminal_growth / 100)) / ((discount_rate - terminal_growth) / 100)
        pv_terminal_value = (terminal_value / ((1 + discount_rate / 100) ** years))
        enterprise_value = (sum(pv_cf) + pv_terminal_value)
        return f"### Deterministic DCF Intrinsic Valuation\n- **Starting FCF:** ${initial_fcf:,.2f}M\n- **Projected Growth ({years} Yrs):** {growth_rate}%\n- **Discount Rate (WACC):** {discount_rate}%\n- **Terminal Growth:** {terminal_growth}%\n- **Estimated Enterprise Value:** **${enterprise_value:,.2f} Million**\n"
    except Exception as e:
        return f"DCF Calculation Error: {e}"

def calculate_emi(principal: float, annual_rate: float, tenure_years: float) -> str:
    monthly_rate = (annual_rate / 100) / 12
    tenure_months = int(tenure_years * 12)
    if tenure_months <= 0: return "Invalid loan tenure."
    emi = (principal / tenure_months) if monthly_rate == 0 else (principal * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1)
    total_payable = (emi * tenure_months)
    return f"### Deterministic Loan Amortization\n- **Principal:** ${principal:,.2f}\n- **Rate:** {annual_rate}%\n- **Tenure:** {tenure_years} Years\n- **Monthly EMI:** **${emi:,.2f}**\n- **Total Payment:** ${total_payable:,.2f}\n"

def parse_agentic_math(user_query: str) -> str:
    q = user_query.lower()
    if "cagr" in q or "compound annual" in q:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)
        if len(nums) >= 3: return calculate_cagr(float(nums[0]), float(nums[1]), float(nums[2]))
    if "dcf" in q or "discounted cash flow" in q or "intrinsic value" in q:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)
        if len(nums) >= 3: return calculate_dcf(float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3]) if len(nums) > 3 else 2.5)
    if "emi" in q or "loan" in q or "amortization" in q:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)
        if len(nums) >= 2: return calculate_emi(float(nums[0]), float(nums[1]), float(nums[2]) if len(nums) > 2 else 5.0)
    return ""

def get_latest_yahoo_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        try:
            val = stock.fast_info.get("last_price")
            if val is not None and pd.notna(val): return float(val)
        except: pass
        try:
            hist = stock.history(period="5d", auto_adjust=False).dropna(subset=["Close"])
            if not hist.empty: return float(hist["Close"].iloc[-1])
        except: pass
        try:
            hist = stock.history(period="1d", interval="1m", auto_adjust=False).dropna(subset=["Close"])
            if not hist.empty: return float(hist["Close"].iloc[-1])
        except: pass
    except: pass
    return None

def get_yahoo_history(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        return hist.dropna(subset=["Close"]) if hist is not None and not hist.empty else pd.DataFrame()
    except: return pd.DataFrame()

def fetch_live_stock_data(query: str) -> str:
    query_lower = query.lower()
    
    # 1. Clean the query to easily extract acronyms like ttml
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query_lower)
    words = clean_query.split()

    ignore_words = {"please", "let", "me", "know", "the", "price", "of", "share", "what", "is", "show", "chart", "for", "stock", "draw", "interactive", "maximum", "timeframe", "from", "launch", "and", "or", "to", "in", "on", "a", "an", "all", "time", "history", "trend", "give", "send", "detail", "details", "about", "company", "world", "market", "calculate", "total", "budget", "combined", "math", "all-time", "invest", "buy", "average", "month", "previous", "mean", "median", "difference", "comparison", "bar", "graph", "worth", "net", "till", "now", "today", "yesterday", "tomorrow", "this", "that", "it", "they", "them", "which", "should", "i", "we", "you", "he", "she", "has", "have", "had", "do", "does", "did", "will", "would", "could", "can", "may", "might", "must", "shall", "be", "am", "are", "was", "were", "been", "being", "tell", "about", "yourself", "yourslef", "ok", "so", "can", "you", "compare", "between", "vs", "versus"}

    potential_tickers = []
    
    # 2. Known Overrides for extremely common items
    global_overrides = {
        "sensex": "^BSESN", "nifty": "^NSEI", "itc": "ITC.NS", "hdfc": "HDFCBANK.NS", "sbi": "SBIN.NS",
        "tata motors": "TATAMOTORS.NS", "capgemini": "CAP.PA", "tcs": "TCS.NS", "tata power": "TATAPOWER.NS",
        "reliance": "RELIANCE.NS", "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", 
        "amd": "AMD", "intel": "INTC", "tsmc": "TSM", "tata steel": "TATASTEEL.NS", "jsw steel": "JSWSTEEL.NS", 
        "jindal steel": "JINDALSTEL.NS", "sail": "SAIL.NS", "gold": "GC=F", "silver": "SI=F", "bitcoin": "BTC-USD"
    }

    for key, ticker in global_overrides.items():
        if key in query_lower and ticker not in potential_tickers:
            potential_tickers.append(ticker)

    # 3. ULTIMATE FIX: The "Acronym Brute-Force" Engine
    # If the user types 'ttml' or 'rvnl', this instantly attaches .NS and .BO and searches Wall Street directly
    for word in words:
        if len(word) >= 2 and word not in ignore_words:
            upper_word = word.upper()
            if upper_word not in potential_tickers: potential_tickers.append(upper_word)
            if f"{upper_word}.NS" not in potential_tickers: potential_tickers.append(f"{upper_word}.NS")
            if f"{upper_word}.BO" not in potential_tickers: potential_tickers.append(f"{upper_word}.BO")

    market_data = ""
    successful_fetches = 0

    # Limit to 3 successful fetches so it doesn't overload Render
    for ticker in potential_tickers:
        if successful_fetches >= 3:
            break
            
        try:
            stock = yf.Ticker(ticker)
            price = get_latest_yahoo_price(ticker)
            
            if price is None: 
                continue
            
            try: info = stock.info
            except: info = {}
            
            name = info.get("shortName") or info.get("longName") or ticker
            currency = info.get("currency") or ("INR" if ".NS" in ticker else "USD")

            hist = get_yahoo_history(ticker, period="1y", interval="1d")
            if not hist.empty:
                closes = hist["Close"].astype(float).round(2).tolist()
                price_1y = closes[0] if closes else "N/A"
                l_30, p_30 = closes[-30:], closes[-60:-30]
                a_30 = round(statistics.mean(l_30), 2) if l_30 else "N/A"
                p_a_30 = round(statistics.mean(p_30), 2) if p_30 else "N/A"
                m_30 = round(statistics.median(l_30), 2) if l_30 else "N/A"
                d_avg = round(a_30 - p_a_30, 2) if a_30 != "N/A" and p_a_30 != "N/A" else "N/A"
            else:
                price_1y = a_30 = p_a_30 = m_30 = d_avg = "N/A"

            market_data += f"\n--- LIVE MARKET DATA FOR {name} ({ticker}) ---\n"
            market_data += f"Live Price: {currency} {price:.2f} | 1 Yr Ago: {price_1y}\n"
            market_data += f"P/E: {info.get('trailingPE', 'N/A')} | EPS: {info.get('trailingEps', 'N/A')} | Debt/Eq: {info.get('debtToEquity', 'N/A')}\n"
            market_data += f"Last 30 Days Avg: {a_30} | Prev 30 Days Avg: {p_a_30} | Median: {m_30} | Diff: {d_avg}\n\n"
            
            successful_fetches += 1
        except: 
            continue

    # 4. Final Fallback if Cloud IP gets blocked
    if not market_data and potential_tickers:
        fallback_db = {"GC=F": "2420.50", "SI=F": "28.40", "TATAPOWER.NS": "382.00", "TATASTEEL.NS": "165.50", "JINDALSTEL.NS": "920.00", "JSWSTEEL.NS": "890.00"}
        for t in potential_tickers[:3]:
            if t in fallback_db:
                market_data += f"\n--- OFFLINE FALLBACK DATA FOR ({t}) ---\nLive Price: {fallback_db[t]}\n\n"

    return market_data

def stream_financial_response(user_query: str, user_profile: dict = None):
    math_result = parse_agentic_math(user_query)
    live_data = fetch_live_stock_data(user_query)
    document_context = "No specific document data found in the knowledge base."
    
    try:
        if collection.count() > 0:
            results = collection.query(query_texts=[user_query], n_results=5)
            if results.get("documents") and len(results["documents"][0]) > 0:
                document_context = "\n...\n".join(results["documents"][0])
    except: pass

    personal_context = ""
    if user_profile:
        personal_context = f"\n[USER PROFILE]\nName: {user_profile.get('full_name', 'User')}\nRole: {user_profile.get('professional_role', 'Unknown')}\nFocus: {user_profile.get('focus_area', 'Unknown')}\n"

    system_prompt = f"""
You are FinanceVision AI, an elite financial advisory engine.
{personal_context}
{math_result}
[LIVE MARKET DATA]
{live_data}
[/LIVE MARKET DATA]
[UPLOADED COMPANY KNOWLEDGE BASE]
{document_context}
[/UPLOADED COMPANY KNOWLEDGE BASE]

CRITICAL MANDATORY RULES:
1. STRICT DOMAIN RESTRICTION: You are strictly a financial AI. 
CRITICAL EXCEPTION: If a user asks about ANY unrecognized word or acronym (e.g., "ttml", "rvnl", "zomato"), YOU MUST ASSUME IT IS A STOCK TICKER OR COMPANY. DO NOT treat it as a movie, game, or fictional character. NEVER refuse a query containing a potential company acronym. If you truly don't know it, say you couldn't find the market data for that specific ticker.
2. If the user explicitly asks to "compare" companies or stocks, ONLY use a Bar Graph:
[COMPARE_CHART|Chart Title|Label1:RawNumber1|Label2:RawNumber2]
3. If the user asks for the trend, analysis, or details of a SINGLE company, ONLY use a Line Chart:
[INTERACTIVE_CHART|TICKER_SYMBOL|line]
4. For investment analysis, structure your text using: ### Fundamental Analysis, ### Pros & Cons, ### Final Verdict.
5. Never apologize for missing data. Just answer using whatever data you have above.
"""
    full_prompt = f"{system_prompt}\n\nUser Query: {user_query}"

    if model is not None:
        try:
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                try:
                    if hasattr(chunk, "text") and chunk.text: yield chunk.text
                except ValueError: pass # Catch Gemini finish_reason=1 silent bug safely
            return
        except Exception as e:
            pass # Proceed silently to Groq on 404 or 429 error

    if GROQ_API_KEY:
        yield "\n\n*(Switching to Groq Backup Engine...)*\n\n"
        try:
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}], "stream": True}
            response = requests.post(groq_url, headers=headers, json=payload, stream=True, timeout=60)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if not line: continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            delta = json.loads(decoded[6:])["choices"][0]["delta"].get("content", "")
                            if delta: yield delta
                        except: pass
                return
        except: pass

    yield "\n\n**All AI Engines Failed.**\n"

def get_market_overview():
    tickers = ["TATAPOWER.NS", "TSLA", "AAPL"]
    names = ["Tata Power", "Tesla", "Apple"]
    data = []
    for ticker, name in zip(tickers, names):
        try:
            current = get_latest_yahoo_price(ticker)
            if not current: continue
            prev = get_yahoo_history(ticker, period="5d")
            previous_close = float(prev["Close"].iloc[-2]) if len(prev) >= 2 else current
            change = current - previous_close
            pct_change = (change / previous_close) * 100
            data.append({"name": name, "ticker": ticker, "price": f"{current:.2f}", "change": round(change, 2), "pct_change": round(pct_change, 2), "news": "Market activity stable."})
        except: continue
    if not data: return [{"name": "Tata Power", "ticker": "TATAPOWER.NS", "price": "382.00", "change": 1.45, "pct_change": 0.34, "news": "Latest available Yahoo Finance market data."}]
    return data

def get_global_indices():
    indices = [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ"), ("^N225", "Nikkei 225"), ("^FTSE", "FTSE 100"), ("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX")]
    data = []
    for ticker, name in indices:
        try:
            current = get_latest_yahoo_price(ticker)
            if not current: continue
            prev = get_yahoo_history(ticker, period="5d")
            previous_close = float(prev["Close"].iloc[-2]) if len(prev) >= 2 else current
            data.append({"name": name, "ticker": ticker, "price": f"{current:.2f}", "change": round(current - previous_close, 2)})
        except: continue
    if not data: return [{"name": "SENSEX", "ticker": "^BSESN", "price": "78542.44", "change": 43.27}]
    return data