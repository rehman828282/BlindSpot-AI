import asyncio
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if load_dotenv is not None:
    load_dotenv()
else:
    _load_local_env()

AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": None,
}


def _active_provider() -> str:
    if AI_PROVIDER in {"groq", "openrouter", "openai", "local"}:
        return AI_PROVIDER
    if GROQ_API_KEY:
        return "groq"
    if OPENROUTER_API_KEY:
        return "openrouter"
    if OPENAI_API_KEY:
        return "openai"
    return "local"


def _active_model() -> str:
    provider = _active_provider()
    if provider == "groq":
        return GROQ_MODEL
    if provider == "openrouter":
        return OPENROUTER_MODEL
    if provider == "openai":
        return OPENAI_MODEL
    return "local-fallback"


MODEL = _active_model()

AGENTS: dict[str, dict[str, Any]] = {
    "Skeptic": {
        "mission": "Stress-test the document's core claims and expose the one assumption that would collapse the argument if challenged.",
        "lens": (
            "Think like a hostile judge, investor, regulator, or technical reviewer. Treat vague confidence, hidden premises, "
            "undefined terms, and unproven cause-effect claims as high-priority weaknesses."
        ),
        "red_flags": [
            "major claim without proof",
            "ambiguous success criteria",
            "unsupported causal leap",
            "missing counterargument",
            "overconfident conclusion",
        ],
        "scoring_dimensions": {
            "assumption_fragility": "How easily the document fails if this missing piece is questioned.",
            "argument_centrality": "How central the missing piece is to the document's main purpose.",
            "challenge_likelihood": "How likely a smart reviewer is to notice and attack the gap.",
        },
    },
    "Evidence": {
        "mission": "Find the highest-impact missing proof, citation, metric, example, benchmark, or source-backed substantiation.",
        "lens": (
            "Think like an evidence auditor. Separate claims from demonstrated facts and identify the missing evidence that "
            "would most improve trust, credibility, and decision-readiness."
        ),
        "red_flags": [
            "numbers without source",
            "claims without examples",
            "no baseline or benchmark",
            "no methodology",
            "missing before-and-after proof",
        ],
        "scoring_dimensions": {
            "proof_gap": "How much critical evidence is missing.",
            "credibility_gain": "How much trust would improve if this missing evidence were added.",
            "decision_value": "How much the missing evidence helps the reader choose what to do.",
        },
    },
    "Risk": {
        "mission": "Identify the missing piece that creates the greatest downside if the reader acts on the document as written.",
        "lens": (
            "Think like a risk officer. Look for missing constraints, failure modes, liabilities, misuse cases, implementation "
            "risks, security gaps, compliance exposure, and operational blind spots."
        ),
        "red_flags": [
            "no failure mode analysis",
            "no mitigation plan",
            "no ownership or accountability",
            "legal or privacy exposure",
            "hidden operational cost",
        ],
        "scoring_dimensions": {
            "severity": "How bad the consequence could be.",
            "probability": "How likely the consequence is under realistic conditions.",
            "detectability": "How hard the risk would be to catch before damage occurs.",
        },
    },
    "User Perspective": {
        "mission": "Discover the missing reader-facing content that prevents the target audience from understanding, trusting, or using the document.",
        "lens": (
            "Think like the intended user with limited patience and high stakes. Identify missing context, onboarding, next steps, "
            "definitions, examples, objections, or decision guidance."
        ),
        "red_flags": [
            "unclear target audience",
            "missing next step",
            "unexplained jargon",
            "reader benefit not explicit",
            "decision criteria absent",
        ],
        "scoring_dimensions": {
            "reader_blocker": "How strongly the gap prevents reader action.",
            "clarity_gain": "How much clearer the document becomes if the gap is fixed.",
            "empathy_gap": "How much the current document ignores user fears, needs, or context.",
        },
    },
    "Competitor": {
        "mission": "Find the missing strategic positioning that would make the document weaker than competing alternatives.",
        "lens": (
            "Think like a competitor, customer, judge, or market evaluator comparing this document against stronger submissions. "
            "Look for missing differentiation, alternatives, tradeoffs, market proof, and defensibility."
        ),
        "red_flags": [
            "no comparison to alternatives",
            "weak differentiation",
            "missing market context",
            "no defensible advantage",
            "unclear why now",
        ],
        "scoring_dimensions": {
            "differentiation_loss": "How much the gap weakens competitive uniqueness.",
            "substitution_risk": "How easily a reader could choose another option.",
            "market_relevance": "How much the missing piece affects market or judge confidence.",
        },
    },
    "Standards": {
        "mission": "Detect the missing professional, technical, ethical, accessibility, or compliance standard that the document must satisfy.",
        "lens": (
            "Think like a standards reviewer. Look for absent acceptance criteria, governance, quality bars, auditability, "
            "security, privacy, accessibility, safety, and domain-specific compliance expectations."
        ),
        "red_flags": [
            "no acceptance criteria",
            "missing quality threshold",
            "no privacy or security note",
            "no accessibility consideration",
            "no audit trail",
        ],
        "scoring_dimensions": {
            "standard_gap": "How far the document is from expected professional quality.",
            "compliance_exposure": "How much formal or informal compliance risk exists.",
            "review_readiness": "How likely the document is to pass expert review after the gap is fixed.",
        },
    },
}

AGENT_OUTPUT_CONTRACT = (
    "Return only one compact JSON object. No markdown. No headings outside JSON. "
    "Use these exact keys: missing_piece, why_it_matters, evidence, fix, score, confidence. "
    "missing_piece must be 12 words or fewer. why_it_matters and fix must be one short sentence each. "
    "evidence must be an array of 1-3 short facts found in the document, not generic assumptions. "
    "score must be an integer from 1 to 10. confidence must be Low, Medium, or High. "
    "If your lens suggests a gap that is irrelevant to this document type, choose a more relevant gap."
)

CONSENSUS_KEYS = [
    "Critical Missing Piece",
    "one_sentence_summary",
    "why_this_matters",
    "facts_from_document",
    "recommended_fix",
    "supporting_agents",
    "confidence",
]


def _has_api_key() -> bool:
    provider = _active_provider()
    if AsyncOpenAI is None:
        return False
    if provider == "groq":
        return bool(GROQ_API_KEY and GROQ_API_KEY.strip())
    if provider == "openrouter":
        return bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip())
    if provider == "openai":
        return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
    return False


def _client():
    if AsyncOpenAI is None:
        raise RuntimeError("The OpenAI Python SDK is not installed.")
    provider = _active_provider()
    if provider == "groq":
        return AsyncOpenAI(api_key=GROQ_API_KEY, base_url=PROVIDER_BASE_URLS["groq"])
    if provider == "openrouter":
        return AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=PROVIDER_BASE_URLS["openrouter"])
    if provider == "openai":
        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    raise RuntimeError("No LLM provider API key is configured.")


async def _call_llm(system_prompt: str, user_prompt: str, *, expect_json: bool = False) -> str:
    client = _client()
    response_format = {"type": "json_object"} if expect_json else None
    provider = _active_provider()

    if provider == "openai":
        try:
            response = await client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text
        except TypeError:
            pass

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format,
    )
    return completion.choices[0].message.content or ""


def _profile_prompt(name: str, profile: dict[str, Any]) -> str:
    red_flags = "\n".join(f"- {flag}" for flag in profile["red_flags"])
    scoring = "\n".join(
        f"- {dimension}: {description}"
        for dimension, description in profile["scoring_dimensions"].items()
    )
    return (
        f"Agent: {name}\n"
        f"Mission: {profile['mission']}\n\n"
        f"Review lens: {profile['lens']}\n\n"
        f"High-priority red flags:\n{red_flags}\n\n"
        f"Scoring dimensions:\n{scoring}\n\n"
        f"{AGENT_OUTPUT_CONTRACT}"
    )


def _shorten(text: Any, *, words: int = 26) -> str:
    clean = " ".join(str(text or "").replace("\n", " ").split())
    parts = clean.split()
    if len(parts) <= words:
        return clean
    return " ".join(parts[:words]).rstrip(".,;:") + "..."


def _as_list(value: Any, *, limit: int = 3) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []
    return [_shorten(item, words=22) for item in items if str(item).strip()][:limit]


def _coerce_score(value: Any, *, default: int = 8) -> int:
    try:
        score = int(float(str(value).split("/")[0]))
    except Exception:
        score = default
    return max(1, min(10, score))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_agent_payload(name: str, payload: dict[str, Any] | str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {"missing_piece": _shorten(payload, words=12)}

    finding = _shorten(payload.get("missing_piece") or payload.get("finding"), words=12)
    if not finding:
        finding = "Missing decision-ready proof"

    evidence = _as_list(payload.get("evidence"), limit=3)
    why = _shorten(payload.get("why_it_matters"), words=24)
    fix = _shorten(payload.get("fix"), words=26)

    if not evidence:
        evidence = ["The document does not provide enough specific support for this claim."]
    if not why:
        why = "This gap makes the reader work too hard to trust the claim."
    if not fix:
        fix = "Add a short section with concrete evidence, measurable impact, and a direct reader benefit."

    normalized = {
        "agent": name,
        "finding": finding,
        "why": why,
        "evidence": evidence,
        "fix": fix,
        "score": _coerce_score(payload.get("score")),
        "confidence": str(payload.get("confidence") or "Medium").title(),
    }
    normalized["content"] = (
        f"{finding} | Why: {why} | Evidence: {'; '.join(evidence)} | Fix: {fix} "
        f"| Score: {normalized['score']}/10"
    )
    return normalized


def _normalize_consensus(payload: dict[str, Any], agent_results: list[dict[str, Any]]) -> dict[str, Any]:
    missing_piece = _shorten(
        payload.get("Critical Missing Piece") or payload.get("critical_missing_piece"),
        words=14,
    )
    if not missing_piece:
        top_agent = max(agent_results, key=lambda result: int(result.get("score", 0)), default={})
        missing_piece = top_agent.get("finding") or "Clear proof of the strongest claim"

    facts = _as_list(payload.get("facts_from_document"), limit=3)
    if not facts:
        for result in agent_results:
            facts.extend(_as_list(result.get("evidence"), limit=2))
            if len(facts) >= 3:
                break
    facts = facts[:3] or ["The uploaded document lacks a concrete fact pattern for the final gap."]

    supporting_agents = payload.get("supporting_agents")
    if not isinstance(supporting_agents, list):
        supporting_agents = [
            {
                "agent": result.get("agent"),
                "finding": result.get("finding"),
                "score": result.get("score"),
            }
            for result in sorted(agent_results, key=lambda item: int(item.get("score", 0)), reverse=True)[:3]
        ]

    return {
        "Critical Missing Piece": missing_piece,
        "one_sentence_summary": _shorten(
            payload.get("one_sentence_summary"),
            words=28,
        )
        or f"The document should explain {missing_piece.lower()} in a specific, evidence-backed way.",
        "why_this_matters": _shorten(
            payload.get("why_this_matters") or payload.get("why_it_is_critical"),
            words=34,
        )
        or "This is the gap most likely to stop a reader from trusting or acting on the document.",
        "facts_from_document": facts,
        "recommended_fix": _shorten(
            payload.get("recommended_fix"),
            words=38,
        )
        or "Add a concise section that names the gap, proves it with facts from the document, and gives a specific fix.",
        "supporting_agents": supporting_agents,
        "confidence": str(payload.get("confidence") or "Medium").title(),
        "agent_consensus": agent_results,
    }


async def _run_agent(name: str, profile: dict[str, Any], context: str) -> dict[str, Any]:
    if not _has_api_key():
        return _fallback_agent(name, profile, context)

    system_prompt = (
        f"You are the {name} agent in BlindSpot AI's multi-agent debate. "
        "You are an expert-level document diagnostic agent for a world-class hackathon product. "
        "Your job is not to summarize the document. Your job is to discover the most consequential missing piece. "
        "Be precise, evidence-grounded, adversarially useful, and brutally practical. "
        "Only use facts visible in the document context. Do not invent risks, requirements, compliance duties, or reader goals. "
        "First infer the document type, then recommend a missing piece that is natural for that type. "
        "For a CV or resume, focus on hiring usefulness: proof, positioning, outcomes, relevance, clarity, and credibility."
    )
    user_prompt = (
        f"{_profile_prompt(name, profile)}\n\n"
        f"Document context:\n{context}\n\n"
        "Now perform your review. Identify only one strongest candidate missing piece from your perspective. "
        "Keep the answer concise and useful for a normal person reading the document."
    )
    try:
        content = await _call_llm(system_prompt, user_prompt, expect_json=True)
        parsed = _extract_json_object(content) or {"missing_piece": content}
    except Exception as exc:
        fallback = _fallback_agent(name, profile, context)
        fallback["content"] = f"LLM call failed for {name}: {exc}. Local fallback: {fallback['content']}"
        return fallback

    return _normalize_agent_payload(name, parsed)


def _fallback_agent(name: str, profile: dict[str, Any], context: str) -> dict[str, Any]:
    excerpt = " ".join(context.split())[:420]
    if not excerpt:
        excerpt = "No document context was available."
    first_dimension = next(iter(profile["scoring_dimensions"]))
    return _normalize_agent_payload(
        name,
        {
            "missing_piece": profile["mission"],
            "why_it_matters": "This gap controls whether the reader can trust the document enough to act.",
            "evidence": [f"Context signal: {excerpt}"],
            "fix": "Add a short, specific section with proof, measurable impact, and reader-focused value.",
            "score": 8 if first_dimension else 7,
            "confidence": "Medium",
        },
    )


async def _rank_criticality(context: str, agent_results: list[dict[str, str]]) -> dict[str, Any]:
    fallback = {
        "Critical Missing Piece": "A concrete, evidence-backed section that answers the reader's highest-impact unresolved decision.",
        "one_sentence_summary": "The document needs one specific proof-backed section that makes its strongest claim easy to trust.",
        "why_this_matters": "Without that proof, the reader may understand the document but still not believe or act on it.",
        "facts_from_document": [],
        "recommended_fix": "Add a dedicated section with evidence, risks, reader impact, competitive context, and standards alignment.",
        "supporting_agents": [],
        "confidence": "Medium",
        "agent_consensus": agent_results,
    }

    if not _has_api_key():
        return _normalize_consensus(fallback, agent_results)

    system_prompt = (
        "You are BlindSpot AI's Criticality Ranking Engine, a final judge that resolves disagreements between six "
        "expert diagnostic agents. Your decision must identify the single missing piece with the greatest combined "
        "impact on truth, risk, reader action, competitive strength, and professional review readiness. "
        "Rank severity over novelty. Prefer the gap that, if fixed, most improves the document's chance of succeeding. "
        "Stay grounded in facts visible in the document context. Do not select a generic best-practice gap unless the "
        "document facts make it directly useful. Output one strict JSON object."
    )
    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Agent findings:\n{json.dumps(agent_results, ensure_ascii=True, indent=2)}\n\n"
        f"Use exactly these JSON keys: {', '.join(CONSENSUS_KEYS)}. "
        "facts_from_document must contain 2-3 short facts from the uploaded document. "
        "supporting_agents must contain the 2-3 most relevant agents with their scores. "
        "recommended_fix must be practical and short enough to show directly in the UI. "
        "Return only valid JSON."
    )

    try:
        raw = await _call_llm(system_prompt, user_prompt, expect_json=True)
        parsed = _extract_json_object(raw) or fallback
        return _normalize_consensus(parsed, agent_results)
    except Exception as exc:
        fallback["why_this_matters"] += f" Ranking engine fallback was used because the LLM call failed: {exc}"
        return _normalize_consensus(fallback, agent_results)


async def stream_debate(context: str):
    yield {
        "event": "status",
        "data": {"message": "Debate initialized.", "provider": _active_provider(), "model": MODEL},
    }

    pending: set[asyncio.Task[dict[str, Any]]] = {
        asyncio.create_task(_run_agent(name, profile, context))
        for name, profile in AGENTS.items()
    }
    results: list[dict[str, str]] = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            results.append(result)
            yield {"event": "agent", "data": result}

    yield {"event": "status", "data": {"message": "Criticality Ranking Engine is resolving consensus."}}
    ranking = await _rank_criticality(context, results)
    yield {"event": "consensus", "data": ranking}
    yield {"event": "done", "data": {"message": "Debate complete."}}


async def draft_missing_piece(issue: str, instructions: str | None, context: str) -> dict[str, str]:
    literal_instructions = instructions if instructions is not None else ""

    if not _has_api_key():
        draft = (
            f"## Missing Piece Draft\n\n"
            f"### Critical Issue\n{issue}\n\n"
            f"### Proposed Content\nAdd a focused section that directly addresses this gap with concrete evidence, "
            f"reader impact, risk mitigation, and a clear recommendation.\n"
        )
        if literal_instructions:
            draft += f"\n### Literal User Instructions Used\n{literal_instructions}\n"
        return {"draft": draft, "literal_instructions_used": literal_instructions}

    system_prompt = (
        "You draft the missing content for BlindSpot AI. If literal user instructions are provided, "
        "use that exact text as the controlling instruction. Do not paraphrase, summarize, rewrite, "
        "sanitize, or otherwise modify the user's instruction text before applying it."
    )
    user_prompt = (
        f"Critical issue:\n{issue}\n\n"
        f"Document context:\n{context}\n\n"
        f"Literal user instructions, if any, appear between the markers below. Use the text exactly as entered.\n"
        f"<literal_user_instructions>\n{literal_instructions}\n</literal_user_instructions>\n\n"
        "Draft the missing content now."
    )

    try:
        draft = await _call_llm(system_prompt, user_prompt)
    except Exception as exc:
        draft = f"Could not generate with configured LLM provider: {exc}"

    return {"draft": draft.strip(), "literal_instructions_used": literal_instructions}
