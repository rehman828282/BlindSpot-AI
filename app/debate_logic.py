import asyncio
import json
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AGENTS: dict[str, str] = {
    "Skeptic": "Find unsupported assumptions, vague claims, and missing proof.",
    "Evidence": "Find missing citations, data, examples, or factual substantiation.",
    "Risk": "Find the omission that creates the greatest operational, legal, financial, or reputational risk.",
    "User Perspective": "Find what the reader or end user most needs but cannot currently see.",
    "Competitor": "Find the missing competitive positioning, differentiation, or market comparison.",
    "Standards": "Find missing compliance, quality, accessibility, or professional standards content.",
}


def _has_api_key() -> bool:
    return bool(AsyncOpenAI and OPENAI_API_KEY and OPENAI_API_KEY.strip())


def _client():
    if AsyncOpenAI is None:
        raise RuntimeError("The OpenAI Python SDK is not installed.")
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


async def _call_openai(system_prompt: str, user_prompt: str, *, expect_json: bool = False) -> str:
    client = _client()
    response_format = {"type": "json_object"} if expect_json else None

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


async def _run_agent(name: str, specialty: str, context: str) -> dict[str, str]:
    if not _has_api_key():
        return _fallback_agent(name, specialty, context)

    system_prompt = (
        f"You are the {name} agent in BlindSpot AI's multi-agent debate. "
        "Identify the most important missing piece in the provided document context. "
        "Be specific, concise, and adversarially useful."
    )
    user_prompt = (
        f"Agent specialty: {specialty}\n\n"
        f"Document context:\n{context}\n\n"
        "Return: 1) your strongest candidate missing piece, 2) why it matters, "
        "3) the evidence in the document that led you there."
    )
    try:
        content = await _call_openai(system_prompt, user_prompt)
    except Exception as exc:
        content = f"OpenAI call failed for {name}: {exc}. Local fallback: {_fallback_agent(name, specialty, context)['content']}"

    return {"agent": name, "content": content.strip()}


def _fallback_agent(name: str, specialty: str, context: str) -> dict[str, str]:
    excerpt = " ".join(context.split())[:420]
    if not excerpt:
        excerpt = "No document context was available."
    content = (
        f"{specialty} The strongest candidate missing piece is a concrete, evidence-backed "
        f"section that resolves the document's main unanswered decision. Context signal: {excerpt}"
    )
    return {"agent": name, "content": content}


async def _rank_criticality(context: str, agent_results: list[dict[str, str]]) -> dict[str, Any]:
    fallback = {
        "Critical Missing Piece": "A concrete, evidence-backed section that answers the reader's highest-impact unresolved decision.",
        "criticality_score": 9,
        "why_it_is_critical": "Multiple review perspectives converged on a missing decision-grade section rather than a cosmetic edit.",
        "agent_consensus": agent_results,
        "recommended_fix": "Add a dedicated section with evidence, risks, reader impact, competitive context, and standards alignment.",
    }

    if not _has_api_key():
        return fallback

    system_prompt = (
        "You are BlindSpot AI's Criticality Ranking Engine. Rank the six agent findings and output "
        "one strict JSON object. The JSON object must include exactly these top-level keys: "
        '"Critical Missing Piece", "criticality_score", "why_it_is_critical", '
        '"agent_consensus", "recommended_fix".'
    )
    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Agent findings:\n{json.dumps(agent_results, ensure_ascii=True, indent=2)}\n\n"
        "Choose the single most critical missing piece. Return only valid JSON."
    )

    try:
        raw = await _call_openai(system_prompt, user_prompt, expect_json=True)
        parsed = json.loads(raw)
        if "Critical Missing Piece" not in parsed:
            parsed["Critical Missing Piece"] = fallback["Critical Missing Piece"]
        return parsed
    except Exception as exc:
        fallback["why_it_is_critical"] += f" Ranking engine fallback was used because the OpenAI call failed: {exc}"
        return fallback


async def stream_debate(context: str):
    yield {"event": "status", "data": {"message": "Debate initialized.", "model": MODEL}}

    pending: set[asyncio.Task[dict[str, str]]] = {
        asyncio.create_task(_run_agent(name, specialty, context))
        for name, specialty in AGENTS.items()
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
        draft = await _call_openai(system_prompt, user_prompt)
    except Exception as exc:
        draft = f"Could not generate with OpenAI: {exc}"

    return {"draft": draft.strip(), "literal_instructions_used": literal_instructions}
