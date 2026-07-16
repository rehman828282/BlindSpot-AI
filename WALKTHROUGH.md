# BlindSpot AI Walkthrough

## What Was Built

BlindSpot AI is now a full-stack FastAPI repository for analyzing uploaded documents and discovering the most critical missing piece through a six-agent debate flow.

## Key Files

- `IMPLEMENTATION_PLAN.md`: Architecture plan drafted before implementation.
- `app/main.py`: FastAPI app, document upload parsing, chunking, embedding, vector storage, SSE analysis route, and fix route.
- `app/debate_logic.py`: Concurrent six-agent debate orchestration and Criticality Ranking Engine.
- `templates/index.html`: Single-file vanilla frontend using Tailwind CDN, standard JavaScript, `fetch()`, and `EventSource`.
- `requirements.txt`: Python dependencies for the FastAPI/OpenAI/document parsing stack.
- `README.md`: Setup and route reference.

## Backend Flow

1. `GET /` renders `templates/index.html` through `Jinja2Templates`.
2. `POST /upload` accepts `.pdf`, `.docx`, and `.pptx`, extracts text, chunks it, embeds it, and stores chunks locally.
3. SQLite stores document chunks and deterministic local embeddings at `blindspot_store.sqlite3`.
4. `GET /analyze` streams live SSE events from the six-agent debate:
   - Skeptic
   - Evidence
   - Risk
   - User Perspective
   - Competitor
   - Standards
5. The Criticality Ranking Engine emits a strict JSON object with `"Critical Missing Piece"`.
6. `POST /fix` drafts missing content for the final issue. Any user-provided custom fix instruction is passed through as literal text without paraphrasing.

## Frontend Flow

1. The user drops a document into the upload zone.
2. Upload and extraction begin immediately; there is no manual analysis start button.
3. The left panel displays extracted text preview and chunk count.
4. The right terminal listens to `/analyze` through `EventSource`.
5. The "Generate Missing Piece" form remains hidden until consensus is reached.
6. The footer includes a developer asset card and does not display the names Awais Ahmed or Saad Malik.

## Verification

- `python -m compileall app` passed.
- `from app.main import app, vector_store` passed with the SQLite vector store.
- `GET http://127.0.0.1:8000/` returned `200`.
- Chromium-family render verification was completed with Playwright controlling installed Google Chrome because the bundled Playwright Chromium binary was not present.
- Browser checks confirmed:
  - Page title is `BlindSpot AI`.
  - Upload zone renders.
  - Extracted text panel renders.
  - Live debate terminal renders.
  - Fix panel is hidden before consensus.
  - Developer asset card renders.
  - Awais Ahmed and Saad Malik are not present in the displayed page text.

## Notes

The app can run without `OPENAI_API_KEY` for local UI testing. In that mode, debate and fix generation use deterministic fallback output. With `OPENAI_API_KEY` set, the OpenAI Python SDK uses `gpt-5.6-sol` by default, or `OPENAI_MODEL` if provided.
