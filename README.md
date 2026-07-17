# BlindSpot AI

BlindSpot AI is a local full-stack AI Analyzer that finds the most critical missing piece in uploaded documents through a six-agent debate architecture.

## Stack

- Backend: FastAPI
- Database: local SQLite vector storage
- AI orchestration: OpenAI Python SDK with `gpt-5.6-sol` by default
- Frontend: vanilla HTML, Tailwind CDN, and standard JavaScript

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## AI Provider

The app uses Groq Cloud by default when `GROQ_API_KEY` is present.

```env
AI_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your-groq-api-key
```

OpenRouter can be added later:

```env
OPENROUTER_MODEL=openrouter/free
OPENROUTER_API_KEY=your-openrouter-api-key
```

Without an API key, the debate and fix endpoints use deterministic local fallback output.

SQLite data is stored in `blindspot_store.sqlite3` for local vector storage.

## Routes

- `GET /` renders `templates/index.html`.
- `POST /upload` accepts `.pdf`, `.docx`, and `.pptx` files, extracts text, chunks it, and indexes it in local SQLite.
- `GET /analyze` streams the multi-agent debate over Server-Sent Events.
- `POST /fix` drafts the missing content for the final critical issue.

If `/fix` receives custom instructions, the exact literal text is preserved and passed through unchanged.
