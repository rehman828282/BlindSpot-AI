# BlindSpot AI Implementation Plan

## Objective

BlindSpot AI is a FastAPI-based AI Analyzer that accepts uploaded `.pdf`, `.docx`, and `.pptx` documents, extracts their text, indexes that content into a local SQLite vector store, and streams a multi-agent debate that identifies the single most critical missing piece in the uploaded material.

## Backend Architecture

The backend is a Python FastAPI application rooted in `app/main.py`.

- `GET /` serves the vanilla frontend from `templates/index.html` using FastAPI `Jinja2Templates`.
- `POST /upload` accepts a single uploaded document, validates the extension, extracts text with the appropriate parser, chunks the extracted text, creates deterministic local embeddings, and stores chunks in SQLite.
- `GET /analyze` opens a Server-Sent Events stream. It queries SQLite for the most relevant document chunks, then streams status updates from a six-agent debate pipeline.
- `POST /fix` accepts the final issue and optional user instructions, then drafts the missing content. When user instructions are provided, the exact literal text is passed through unchanged and is never automatically paraphrased before use.

SQLite state is stored in `blindspot_store.sqlite3`, so the app can run locally without a separate database service. The app performs cosine ranking over deterministic local embeddings stored in SQLite.

## Debate Architecture

The debate workflow lives in `app/debate_logic.py`.

Six agents run concurrently:

- Skeptic
- Evidence
- Risk
- User Perspective
- Competitor
- Standards

Each agent receives the same retrieved document context but evaluates it from its assigned perspective. Their responses are then sent to a Criticality Ranking Engine, which returns one strict JSON object containing the required `"Critical Missing Piece"` field plus supporting ranking metadata.

The OpenAI Python SDK is used through an async client. The default orchestration model is `gpt-5.6-sol`, overridable with `OPENAI_MODEL`. If no API key is present during local development, the app emits deterministic local fallback analysis so the UI and SSE workflow remain testable.

## Frontend Architecture

The frontend is a single file at `templates/index.html`.

- Tailwind is loaded by CDN.
- No React, Vite, bundler, or frontend build step is required.
- Drag-and-drop upload starts processing immediately when a file is dropped.
- The layout is split into a left extracted-text preview panel and a right terminal-style SSE debate panel.
- The "Generate Missing Piece" form is hidden until the debate consensus event is received.
- A footer developer asset card is included without displaying the names Awais Ahmed or Saad Malik.

## Verification Plan

1. Install dependencies from `requirements.txt`.
2. Run the FastAPI app with `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
3. Open `http://localhost:8000` in Chromium.
4. Confirm the root route serves `templates/index.html`, styles load from Tailwind CDN, and the static UI renders.
5. Generate a final walkthrough artifact summarizing all files, routes, and usage.
