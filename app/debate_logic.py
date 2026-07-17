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
    if AI_PROVIDER in {"groq", "openrouter", "openai"}:
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
    "Return a rigorous review using exactly these headings:\n"
    "1. Critical Missing Piece Candidate\n"
    "2. Why This Is The Highest-Leverage Gap\n"
    "3. Evidence From The Document\n"
    "4. What A Strong Fix Must Include\n"
    "5. Scores\n\n"
    "For Scores, provide three 1-10 scores using your assigned scoring dimensions and one final criticality score. "
    "Do not invent document facts. If evidence is absent, say what is absent and why that absence matters."
)


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


async def _run_agent(name: str, profile: dict[str, Any], context: str) -> dict[str, str]:
    if not _has_api_key():
        return _fallback_agent(name, profile, context)

    system_prompt = (
        f"You are the {name} agent in BlindSpot AI's multi-agent debate. "
        "You are an expert-level document diagnostic agent for a world-class hackathon product. "
        "Your job is not to summarize the document. Your job is to discover the most consequential missing piece. "
        "Be precise, evidence-grounded, adversarially useful, and brutally practical."
    )
    user_prompt = (
        f"{_profile_prompt(name, profile)}\n\n"
        f"Document context:\n{context}\n\n"
        "Now perform your review. Identify only one strongest candidate missing piece from your perspective."
    )
    try:
        content = await _call_llm(system_prompt, user_prompt)
    except Exception as exc:
        content = f"LLM call failed for {name}: {exc}. Local fallback: {_fallback_agent(name, profile, context)['content']}"

    return {"agent": name, "content": content.strip()}


def _fallback_agent(name: str, profile: dict[str, Any], context: str) -> dict[str, str]:
    excerpt = " ".join(context.split())[:420]
    if not excerpt:
        excerpt = "No document context was available."
    first_dimension = next(iter(profile["scoring_dimensions"]))
    content = (
        "1. Critical Missing Piece Candidate\n"
        f"{profile['mission']}\n\n"
        "2. Why This Is The Highest-Leverage Gap\n"
        "This gap likely controls whether a reader can trust the document enough to act on it. "
        "A strong submission needs decision-grade detail, not just confident framing.\n\n"
        "3. Evidence From The Document\n"
        f"Available context signal: {excerpt}\n\n"
        "4. What A Strong Fix Must Include\n"
        "Add a focused section with concrete claims, supporting evidence, risk handling, reader impact, "
        "competitive positioning, and measurable acceptance criteria.\n\n"
        "5. Scores\n"
        f"- {first_dimension}: 8/10\n"
        "- final_criticality: 9/10"
    )
    return {"agent": name, "content": content}


async def _rank_criticality(context: str, agent_results: list[dict[str, str]]) -> dict[str, Any]:
    fallback = {
        "Critical Missing Piece": "A concrete, evidence-backed section that answers the reader's highest-impact unresolved decision.",
        "criticality_score": 9,
        "why_it_is_critical": (
            "Multiple expert perspectives converged on a missing decision-grade section rather than a cosmetic edit. "
            "The gap affects trust, actionability, risk, and review readiness."
        ),
        "agent_consensus": agent_results,
        "recommended_fix": "Add a dedicated section with evidence, risks, reader impact, competitive context, and standards alignment.",
    }

    if not _has_api_key():
        return fallback

    system_prompt = (
        "You are BlindSpot AI's Criticality Ranking Engine, a final judge that resolves disagreements between six "
        "expert diagnostic agents. Your decision must identify the single missing piece with the greatest combined "
        "impact on truth, risk, reader action, competitive strength, and professional review readiness. "
        "Rank severity over novelty. Prefer the gap that, if fixed, most improves the document's chance of succeeding. "
        "Output one strict JSON object. The JSON object must include exactly these top-level keys: "
        '"Critical Missing Piece", "criticality_score", "why_it_is_critical", '
        '"agent_consensus", "recommended_fix".'
    )
    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Agent findings:\n{json.dumps(agent_results, ensure_ascii=True, indent=2)}\n\n"
        "Use this ranking rubric: severity, evidence gap, reader impact, risk exposure, competitive weakness, "
        "standards failure, and recoverability. Choose the single most critical missing piece. Return only valid JSON."
    )

    try:
        raw = await _call_llm(system_prompt, user_prompt, expect_json=True)
        parsed = json.loads(raw)
        if "Critical Missing Piece" not in parsed:
            parsed["Critical Missing Piece"] = fallback["Critical Missing Piece"]
        return parsed
    except Exception as exc:
        fallback["why_it_is_critical"] += f" Ranking engine fallback was used because the LLM call failed: {exc}"
        return fallback


async def stream_debate(context: str):
    yield {
        "event": "status",
        "data": {"message": "Debate initialized.", "provider": _active_provider(), "model": MODEL},
    }

    pending: set[asyncio.Task[dict[str, str]]] = {
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
