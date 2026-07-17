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


def classify_document(context: str) -> dict[str, Any]:
    normalized = context.lower()
    scores: dict[str, int] = {}
    for track, config in TRACKS.items():
        if track == "general":
            continue
        scores[track] = sum(normalized.count(keyword) for keyword in config["keywords"])

    selected = max(scores, key=scores.get, default="general")
    if scores.get(selected, 0) < 2:
        selected = "general"

    config = TRACKS[selected]
    return {
        "track": selected,
        "label": config["label"],
        "framework": config["framework"],
        "confidence": "High" if scores.get(selected, 0) >= 5 else "Medium",
        "scores": scores,
    }


def _agent_profiles_for(track: str) -> dict[str, dict[str, Any]]:
    return TRACKS.get(track, TRACKS["general"])["agents"]

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
    "evidence_from_document",
    "facts_from_document",
    "consequence",
    "confidence_score",
    "recommended_fix",
    "supporting_agents",
    "confidence",
]

PITCH_DECK_AGENTS: dict[str, dict[str, Any]] = {
    "Skeptic Agent": {
        "mission": "Interrogate market sizing, financial projections, and core business assumptions for unbacked leaps.",
        "lens": "Think like a hostile venture partner searching for the first number or claim that breaks the story.",
        "red_flags": ["unbacked TAM or SAM", "unrealistic growth curve", "unclear pricing logic", "no margin assumptions", "missing why now"],
        "scoring_dimensions": {
            "assumption_fragility": "How easily the pitch fails if this assumption is challenged.",
            "investor_challenge_likelihood": "How likely investors are to challenge this gap.",
            "fundraising_damage": "How much the gap hurts fundability.",
        },
    },
    "Evidence Agent": {
        "mission": "Find the missing customer, traction, pilot, usage, revenue, testimonial, or LOI evidence.",
        "lens": "Think like a traction auditor. Features do not prove demand; external validation does.",
        "red_flags": ["no customer interviews", "no LOIs", "no usage metrics", "no pilot results", "no revenue or retention evidence"],
        "scoring_dimensions": {
            "proof_gap": "How much external validation is missing.",
            "credibility_gain": "How much investor trust would improve if fixed.",
            "screening_value": "How much the fix helps pass first review.",
        },
    },
    "Risk Agent": {
        "mission": "Find the missing execution risk, regulatory hurdle, dependency, or barrier-to-entry discussion.",
        "lens": "Think like an investor who wants to know what can kill execution after funding.",
        "red_flags": ["regulatory risk omitted", "supplier dependency", "unclear go-to-market cost", "weak moat", "complex implementation risk"],
        "scoring_dimensions": {
            "severity": "How badly this risk can damage the startup.",
            "probability": "How likely the risk is in realistic execution.",
            "detectability": "How visible the risk is to investors.",
        },
    },
    "Investor Agent": {
        "mission": "Judge whether the market, team, traction, and story give a venture investor a reason to keep reading.",
        "lens": "Think like a VC screening fast: fund-return potential, team fit, urgency, and proof of demand.",
        "red_flags": ["team-market fit unclear", "market not venture-scale", "no clear buyer", "weak fundraising narrative", "missing use of funds"],
        "scoring_dimensions": {
            "reader_blocker": "How much the gap blocks investor conviction.",
            "fund_return_relevance": "How much the gap affects venture-scale potential.",
            "team_confidence": "How much the gap affects belief in execution ability.",
        },
    },
    "Competitor Agent": {
        "mission": "Identify missing direct, indirect, or hidden competitors and the absent moat explanation.",
        "lens": "Think like a rival founder or investor comparing alternatives and substitutes.",
        "red_flags": ["competitor matrix too narrow", "no indirect alternatives", "weak moat", "no defensibility", "unclear differentiation"],
        "scoring_dimensions": {
            "differentiation_loss": "How much uniqueness is weakened.",
            "substitution_risk": "How easily customers can choose an alternative.",
            "moat_strength_gap": "How much defensibility is missing.",
        },
    },
    "Framework Standards Agent": {
        "mission": "Audit pitch structure against elite deck expectations and identify the missing slide or section.",
        "lens": "Think like a YC or Sequoia-style deck reviewer: problem, solution, market, traction, GTM, team, ask, why now.",
        "red_flags": ["missing traction slide", "missing GTM", "missing why now", "missing ask/use of funds", "weak problem framing"],
        "scoring_dimensions": {
            "deck_completeness": "How important the missing deck element is.",
            "review_readiness": "How much the fix improves investor review readiness.",
            "sequence_damage": "How much the gap breaks the pitch flow.",
        },
    },
}

RESUME_AGENTS: dict[str, dict[str, Any]] = {
    "Skeptic Agent": {
        "mission": "Scrutinize employment gaps, title jumps, vague claims, and unsupported career narratives.",
        "lens": "Think like a skeptical hiring manager validating whether the resume story is believable.",
        "red_flags": ["unexplained gap", "unbacked title change", "vague summary", "unclear role scope", "inflated claim"],
        "scoring_dimensions": {
            "assumption_fragility": "How easily a recruiter could doubt the claim.",
            "career_story_centrality": "How central the gap is to the candidate story.",
            "challenge_likelihood": "How likely a reviewer is to notice it.",
        },
    },
    "Evidence Agent": {
        "mission": "Find resume bullets that lack quantified impact, business outcomes, metrics, or proof.",
        "lens": "Think like a recruiter converting claims into evidence: percentages, revenue, time saved, users, scale, accuracy, performance.",
        "red_flags": ["job-description bullet", "no numbers", "no outcome", "no scale", "no proof of impact"],
        "scoring_dimensions": {
            "proof_gap": "How much measurable evidence is missing.",
            "credibility_gain": "How much stronger the resume becomes with metrics.",
            "shortlist_value": "How much the fix helps the candidate get shortlisted.",
        },
    },
    "Risk Agent": {
        "mission": "Detect hiring risks such as over-specialization, outdated skills, weak progression, or unclear role fit.",
        "lens": "Think like a hiring panel looking for reasons the candidate may not succeed in the target role.",
        "red_flags": ["skills mismatch", "outdated stack", "unclear progression", "too many unrelated projects", "no depth signal"],
        "scoring_dimensions": {
            "severity": "How much this risk hurts hiring confidence.",
            "probability": "How likely reviewers are to infer the risk.",
            "detectability": "How obvious the risk is during screening.",
        },
    },
    "Recruiter Agent": {
        "mission": "Simulate a six-second elite technical recruiter scan and find the missing role-fit signal.",
        "lens": "Think like a recruiter matching title, keywords, outcomes, seniority, and role relevance at speed.",
        "red_flags": ["target role unclear", "missing keywords", "summary not role-specific", "weak top third", "unclear seniority"],
        "scoring_dimensions": {
            "reader_blocker": "How much the gap blocks fast recruiter understanding.",
            "keyword_match": "How much the gap affects ATS and recruiter matching.",
            "role_fit_clarity": "How clearly the candidate fits the desired role.",
        },
    },
    "Applicant Competitor Agent": {
        "mission": "Compare the resume against top-tier applicants and find the missing differentiator.",
        "lens": "Think like a hiring manager comparing this resume in a stack of strong applicants.",
        "red_flags": ["generic profile", "no unique value", "projects lack outcomes", "no standout achievement", "weak positioning"],
        "scoring_dimensions": {
            "differentiation_loss": "How much the resume blends into the pile.",
            "substitution_risk": "How easy it is to pick a similar applicant.",
            "market_relevance": "How relevant the differentiator is to hiring demand.",
        },
    },
    "ATS Standards Agent": {
        "mission": "Validate ATS readability, section structure, action verbs, keyword clarity, and parse-friendly formatting.",
        "lens": "Think like an ATS and resume standards reviewer, not a compliance auditor.",
        "red_flags": ["missing standard sections", "weak action verbs", "unclear dates", "keyword gaps", "formatting may parse poorly"],
        "scoring_dimensions": {
            "ats_parse_risk": "How much the gap hurts machine parsing.",
            "standards_gap": "How far the resume is from modern resume structure.",
            "review_readiness": "How much the fix improves recruiter review.",
        },
    },
}

TRACKS = {
    "startup_pitch_deck": {
        "label": "Startup Pitch Deck",
        "framework": "Investor Pitch Review",
        "agents": PITCH_DECK_AGENTS,
        "keywords": [
            "tam",
            "sam",
            "som",
            "traction",
            "market",
            "investor",
            "funding",
            "revenue",
            "pitch",
            "deck",
            "go-to-market",
            "gtm",
            "competitor",
            "moat",
            "valuation",
            "seed",
            "series",
        ],
    },
    "resume": {
        "label": "Resume / CV",
        "framework": "Recruiter and ATS Review",
        "agents": RESUME_AGENTS,
        "keywords": [
            "resume",
            "cv",
            "experience",
            "education",
            "skills",
            "projects",
            "certifications",
            "developer",
            "engineer",
            "intern",
            "employment",
            "linkedin",
            "github",
        ],
    },
    "general": {
        "label": "General Document",
        "framework": "Critical Missing Piece Review",
        "agents": AGENTS,
        "keywords": [],
    },
}


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


def _profile_prompt(name: str, profile: dict[str, Any], classification: dict[str, Any]) -> str:
    red_flags = "\n".join(f"- {flag}" for flag in profile["red_flags"])
    scoring = "\n".join(
        f"- {dimension}: {description}"
        for dimension, description in profile["scoring_dimensions"].items()
    )
    return (
        f"Detected document type: {classification['label']}\n"
        f"Evaluation framework: {classification['framework']}\n\n"
        f"Agent: {name}\n"
        f"Mission: {profile['mission']}\n\n"
        f"Review lens: {profile['lens']}\n\n"
        f"High-priority red flags:\n{red_flags}\n\n"
        f"Scoring dimensions:\n{scoring}\n\n"
        f"{AGENT_OUTPUT_CONTRACT}"
    )


def _shorten(text: Any, *, words: int = 26) -> str:
    if isinstance(text, dict):
        for key in (
            "text",
            "summary",
            "title",
            "value",
            "description",
            "recommendation",
            "recommended_fix",
            "missing_piece",
        ):
            if text.get(key):
                text = text[key]
                break
        else:
            text = "; ".join(
                f"{key}: {value}"
                for key, value in text.items()
                if value not in (None, "", [], {})
            )
    elif isinstance(text, list):
        text = "; ".join(str(item) for item in text if item not in (None, "", [], {}))

    clean = " ".join(str(text or "").replace("\n", " ").split())
    parts = clean.split()
    if len(parts) <= words:
        return clean
    return " ".join(parts[:words]).rstrip(".,;:") + "..."


def _as_list(value: Any, *, limit: int = 3) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = [
            item
            for item in value.values()
            if item not in (None, "", [], {})
        ]
    elif value:
        items = [value]
    else:
        items = []
    return [_shorten(item, words=22) for item in items if str(item).strip()][:limit]


def _coerce_score(value: Any, *, default: int = 8) -> int:
    if isinstance(value, dict):
        for key in ("score", "value", "criticality", "rating"):
            if key in value:
                value = value[key]
                break
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

    finding = _shorten(
        payload.get("missing_piece")
        or payload.get("finding")
        or payload.get("Critical Missing Piece")
        or payload.get("critical_missing_piece"),
        words=12,
    )
    if not finding:
        finding = "Missing decision-ready proof"

    evidence = _as_list(
        payload.get("evidence")
        or payload.get("evidence_from_document")
        or payload.get("facts_from_document"),
        limit=3,
    )
    why = _shorten(
        payload.get("why_it_matters")
        or payload.get("why")
        or payload.get("why_this_matters"),
        words=24,
    )
    fix = _shorten(
        payload.get("fix")
        or payload.get("recommended_fix")
        or payload.get("recommendation"),
        words=26,
    )

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


def _coerce_percent(value: Any, *, default: int = 84) -> int:
    if isinstance(value, dict):
        for key in ("score", "value", "confidence", "percentage"):
            if key in value:
                value = value[key]
                break
    try:
        percent = int(float(str(value).replace("%", "").split("/")[0]))
    except Exception:
        percent = default
    if percent <= 10:
        percent *= 10
    return max(1, min(99, percent))


def _normalize_consensus(
    payload: dict[str, Any],
    agent_results: list[dict[str, Any]],
    classification: dict[str, Any],
) -> dict[str, Any]:
    missing_piece = _shorten(
        payload.get("Critical Missing Piece") or payload.get("critical_missing_piece"),
        words=14,
    )
    if not missing_piece:
        top_agent = max(agent_results, key=lambda result: int(result.get("score", 0)), default={})
        missing_piece = top_agent.get("finding") or "Clear proof of the strongest claim"

    facts = _as_list(payload.get("evidence_from_document") or payload.get("facts_from_document"), limit=3)
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
    else:
        supporting_agents = [
            {
                "agent": _shorten(agent.get("agent") if isinstance(agent, dict) else agent, words=5),
                "finding": _shorten(agent.get("finding") if isinstance(agent, dict) else "", words=10),
                "score": _coerce_score(agent.get("score") if isinstance(agent, dict) else None),
            }
            for agent in supporting_agents[:3]
        ]

    return {
        "document_type": classification["label"],
        "evaluation_framework": classification["framework"],
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
        "evidence_from_document": facts,
        "consequence": _shorten(
            payload.get("consequence"),
            words=30,
        )
        or "The reader may reject or ignore the document because the central proof is not clear enough.",
        "confidence_score": _coerce_percent(payload.get("confidence_score")),
        "recommended_fix": _shorten(
            payload.get("recommended_fix"),
            words=38,
        )
        or "Add a concise section that names the gap, proves it with facts from the document, and gives a specific fix.",
        "supporting_agents": supporting_agents,
        "confidence": str(payload.get("confidence") or "Medium").title(),
        "agent_consensus": agent_results,
    }


async def _run_agent(
    name: str,
    profile: dict[str, Any],
    context: str,
    classification: dict[str, Any],
) -> dict[str, Any]:
    if not _has_api_key():
        return _fallback_agent(name, profile, context)

    system_prompt = (
        f"You are the {name} agent in BlindSpot AI's multi-agent debate. "
        "You are an expert-level document diagnostic agent for a world-class hackathon product. "
        "Your job is not to summarize the document. Your job is to discover the most consequential missing piece. "
        "Be precise, evidence-grounded, adversarially useful, and brutally practical. "
        "Only use facts visible in the document context. Do not invent risks, requirements, compliance duties, or reader goals. "
        f"The document has already been classified as {classification['label']}; use the {classification['framework']} framework. "
        "Recommend a missing piece that is natural for this document type. "
        "For resumes/CVs, focus on hiring usefulness: proof, positioning, outcomes, relevance, clarity, ATS, and credibility. "
        "For startup pitch decks, focus on investor usefulness: market, traction, moat, team, GTM, ask, and proof of demand."
    )
    user_prompt = (
        f"{_profile_prompt(name, profile, classification)}\n\n"
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


async def _rank_criticality(
    context: str,
    agent_results: list[dict[str, Any]],
    classification: dict[str, Any],
) -> dict[str, Any]:
    fallback = {
        "Critical Missing Piece": "A concrete, evidence-backed section that answers the reader's highest-impact unresolved decision.",
        "one_sentence_summary": "The document needs one specific proof-backed section that makes its strongest claim easy to trust.",
        "why_this_matters": "Without that proof, the reader may understand the document but still not believe or act on it.",
        "evidence_from_document": [],
        "facts_from_document": [],
        "consequence": "The reader may reject or ignore the document because the central proof is not clear enough.",
        "confidence_score": 84,
        "recommended_fix": "Add a dedicated section with evidence, risks, reader impact, competitive context, and standards alignment.",
        "supporting_agents": [],
        "confidence": "Medium",
        "agent_consensus": agent_results,
    }

    if not _has_api_key():
        return _normalize_consensus(fallback, agent_results, classification)

    system_prompt = (
        "You are BlindSpot AI's Criticality Ranking Engine, a final judge that resolves disagreements between six "
        "expert diagnostic agents. Your decision must identify the single missing piece with the greatest combined "
        "impact on truth, risk, reader action, competitive strength, and professional review readiness. "
        f"The document type is {classification['label']} and the active framework is {classification['framework']}. "
        "Rank severity over novelty. Prefer the gap that, if fixed, most improves the document's chance of succeeding. "
        "Stay grounded in facts visible in the document context. Do not select a generic best-practice gap unless the "
        "document facts make it directly useful. Output one strict JSON object."
    )
    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Agent findings:\n{json.dumps(agent_results, ensure_ascii=True, indent=2)}\n\n"
        f"Use exactly these JSON keys: {', '.join(CONSENSUS_KEYS)}. "
        "evidence_from_document and facts_from_document must contain the same 2-3 short facts from the uploaded document. "
        "consequence must be one short sentence explaining the immediate downside if the gap is not fixed. "
        "confidence_score must be an integer percentage from 1 to 99 based on cross-agent agreement. "
        "supporting_agents must contain the 2-3 most relevant agents with their scores. "
        "recommended_fix must be practical and short enough to show directly in the UI. "
        "Return only valid JSON."
    )

    try:
        raw = await _call_llm(system_prompt, user_prompt, expect_json=True)
        parsed = _extract_json_object(raw) or fallback
        return _normalize_consensus(parsed, agent_results, classification)
    except Exception as exc:
        fallback["why_this_matters"] += f" Ranking engine fallback was used because the LLM call failed: {exc}"
        return _normalize_consensus(fallback, agent_results, classification)


async def stream_debate(context: str):
    classification = classify_document(context)
    agent_profiles = _agent_profiles_for(classification["track"])
    yield {
        "event": "status",
        "data": {
            "message": "Debate initialized.",
            "provider": _active_provider(),
            "model": MODEL,
            "document_type": classification["label"],
            "framework": classification["framework"],
            "classification_confidence": classification["confidence"],
        },
    }

    pending: set[asyncio.Task[dict[str, Any]]] = {
        asyncio.create_task(_run_agent(name, profile, context, classification))
        for name, profile in agent_profiles.items()
    }
    results: list[dict[str, Any]] = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            results.append(result)
            yield {"event": "agent", "data": result}

    yield {"event": "status", "data": {"message": "Criticality Ranking Engine is resolving consensus."}}
    ranking = await _rank_criticality(context, results, classification)
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
