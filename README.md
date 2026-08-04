📈 FinanceVision | Enterprise Intelligence Engine

FinanceVision is a highly resilient, enterprise-grade financial AI terminal. Built to mirror institutional trading software, it provides real-time market data, deep fundamental analysis, and secure Retrieval-Augmented Generation (RAG) for massive corporate documents.

🚀 Core Enterprise Features

1. Triple-Engine AI Failover

To guarantee 100% uptime and bypass strict API rate limits, FinanceVision utilizes a custom cascading LLM architecture:

Primary Engine: Google Gemini 2.5 Flash (Optimized for speed and complex reasoning).

Secondary Fallback: Groq Llama 3.1 8B (Instantly catches HTTP 429 Rate Limit errors).

Tertiary Fallback: Local Ollama (Ensures the application never goes offline, even without internet).

2. Advanced RAG Pipeline (ChromaDB)

Capable of ingesting 200+ page corporate PDFs and dense CSV financial ledgers. The system utilizes overlapping chunking strategies to bypass context-window limits, allowing the AI to extract hidden metrics and cite exact document context seamlessly alongside live internet data.

3. Deterministic Agentic Math

LLMs are notoriously prone to math hallucinations. FinanceVision solves this by utilizing an Agentic Intercept approach. When quantitative calculations are required (e.g., share purchasing, budget differences), the backend bypasses the LLM, computes the exact statistics natively in Python using pandas and statistics, and injects the factual numbers directly into the AI's context.

4. Ironclad Domain Guardrails

The engine is strictly locked to financial and economic analysis. System-level guardrails automatically detect and courteously reject non-financial prompts (e.g., history, sports, coding), preventing prompt injection and off-topic hallucination even if the answers exist in the uploaded RAG documents.

5. Multi-Modal Interactive UI

Dynamic Charts: Auto-generates interactive Chart.js comparison bars and time-series line graphs from live Wall Street data.

Voice Engine: Native browser Web Speech API integration for hands-free dictation and Text-to-Speech (TTS) auditory report readouts.

Export Pipeline: One-click Markdown (.md) report compilation for sharing analyst sessions.

🛠️ Installation & Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/financial-chatboat.git
cd financial-chatboat


2. Install dependencies

pip install -r requirements.txt


3. Configure Environment Variables
Create a .env file in the root directory and add your API credentials:

GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
SMTP_EMAIL=your_gmail_here
SMTP_PASSWORD=your_app_password_here


4. Initialize the Database & Run the Server

uvicorn app.main:app --reload


Navigate to http://127.0.0.1:8000 to access the terminal.

🔒 Security & Authentication

Features a robust OAuth2 and standard JWT authentication pipeline.

Google SSO: One-click enterprise login.

MFA/OTP Verification: Standard registrations are secured via real-time SMTP 6-digit email verification.

Bcrypt Hashing: Passwords are never stored in plain text.