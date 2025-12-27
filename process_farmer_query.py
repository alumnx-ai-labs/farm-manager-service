"""
Farm Manager Service
--------------------
Agentic workflow using LangGraph to route farmer queries to:
- Scheme service (government schemes)
- Crop service (text or image based)
- Collector (LLM synthesis into one final answer)

This file is intentionally defensive:
- Handles unstable backend schemas
- Handles transient network failures
- Avoids hallucinations and cross-domain leakage
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import os
import asyncio
import httpx
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Environment & Gemini Setup
# -------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def _validate_gemini_api_key() -> bool:
    """
    Best-effort Gemini validation.
    If this fails, the system must still function via rule-based fallback.
    """
    if not GEMINI_API_KEY:
        print("[Gemini] API key missing – falling back to rule-based routing.")
        return False

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai.GenerativeModel(GEMINI_MODEL_NAME)
        print("[Gemini] Gemini intent + collector enabled.")
        return True
    except Exception as e:
        print(f"[Gemini] Validation failed: {e!r}")
        return False


GEMINI_API_KEY_VALID = _validate_gemini_api_key()

gemini_model = (
    genai.GenerativeModel(GEMINI_MODEL_NAME)
    if GEMINI_API_KEY_VALID
    else None
)

COLLECTOR_MODEL = gemini_model  # same model reused for synthesis


# -------------------------------------------------------------------
# State Definition
# -------------------------------------------------------------------

class FarmerState(TypedDict):
    text: str
    imageUrl: str | None
    intent: Literal["scheme", "crop", "both"] | None
    entities: dict
    scheme_response: dict | None
    crop_response: dict | None
    service_response: dict


# -------------------------------------------------------------------
# Service URLs
# -------------------------------------------------------------------

SCHEME_SERVICE_URL = os.getenv(
    "SCHEME_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/query-government-schemes"
)

CROP_TEXT_SERVICE_URL = os.getenv(
    "CROP_TEXT_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/ask-consultant"
)

CROP_IMAGE_SERVICE_URL = os.getenv(
    "CROP_IMAGE_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/ask-with-image"
)

# -------------------------------------------------------------------
# Intent Classification (Fallback)
# -------------------------------------------------------------------

def _rule_based_intent(query: str) -> Literal["scheme", "crop", "both", "none"] | None:
    """
    Used only when Gemini is unavailable.
    """
    q = query.lower()

    scheme_keywords = [
        "scheme", "subsidy", "government", "yojana", "pm-kisan",
        "financial assistance", "grant", "loan"
    ]

    crop_keywords = [
        "crop", "disease", "pest", "yield", "fertilizer",
        "soil", "irrigation", "plant", "tree"
    ]

    has_scheme = any(k in q for k in scheme_keywords)
    has_crop = any(k in q for k in crop_keywords)

    if has_scheme and has_crop:
        return "both"
    if has_scheme:
        return "scheme"
    if has_crop:
        return "crop"

    return None


# -------------------------------------------------------------------
# Supervisor Node
# -------------------------------------------------------------------

def supervisor_node(state):
    """
    Decides intent: scheme / crop / both.
    Uses Gemini if available, otherwise rule-based.
    """
    query = (state.get("text") or "").strip()
    image_url = state.get("imageUrl")

    print(f"[Supervisor] query={query!r}, imageUrl={image_url!r}")

    # Image without text → crop by default
    if not query and image_url:
        return {**state, "intent": "crop", "entities": {}}

    if not query:
        return {**state, "intent": None, "entities": {}}

    if not gemini_model:
        intent = _rule_based_intent(query)
        print(f"[Supervisor] Rule-based intent={intent}")
        return {**state, "intent": intent, "entities": {}}

    prompt = f"""
Classify the farmer query into one of:
scheme | crop | both | none

Query:
{query}

Respond with ONLY one word.
"""

    try:
        response = gemini_model.generate_content(prompt)
        intent = (response.text or "").strip().lower()
        if intent not in {"scheme", "crop", "both"}:
            intent = None
        return {**state, "intent": intent, "entities": {}}
    except Exception:
        return {**state, "intent": None, "entities": {}}


# -------------------------------------------------------------------
# Scheme Node
# -------------------------------------------------------------------

async def scheme_node(state):
    """
    Fetches Andhra Pradesh government schemes.
    """
    query = state.get("text", "")
    print(f"[Scheme Node] query={query}")

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                SCHEME_SERVICE_URL,
                json={
                    "query": f"""
You are an expert on ANDHRA PRADESH agricultural government schemes.

Rules:
- Andhra Pradesh only
- Mention applicability per crop if relevant
- Do NOT hallucinate

Farmer question:
{query}
"""
                },
                timeout=30.0
            )
            r.raise_for_status()
            return {**state, "scheme_response": r.json()}
    except Exception as e:
        return {**state, "scheme_response": {"error": str(e)}}


# -------------------------------------------------------------------
# Crop Utilities
# -------------------------------------------------------------------

def normalize_crop_output(raw: dict) -> dict:
    """
    Backend may return:
    - { "answer": ... }
    - { "response": ... }
    Normalize to { "response": ... }
    """
    if not isinstance(raw, dict):
        return {"response": str(raw)}

    if "response" in raw:
        return raw

    if "answer" in raw:
        raw["response"] = raw.pop("answer")
        return raw

    return {"response": str(raw)}


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    data: dict | None = None,
    json: dict | None = None,
    retries: int = 3,
):
    """
    Retries transient network failures (Windows + TLS common issue).
    """
    for attempt in range(1, retries + 1):
        try:
            return await client.post(
                url,
                data=data,
                json=json,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        except httpx.ConnectError:
            if attempt == retries:
                raise
            await asyncio.sleep(1.5 * attempt)


# -------------------------------------------------------------------
# Crop Node
# -------------------------------------------------------------------

async def crop_node(state):
    """
    Handles crop advisory:
    - Image → ask-with-image (multipart/form-data)
    - Text → ask-consultant (JSON)
    """
    query = (state.get("text") or "").strip()
    image_url = state.get("imageUrl")

    print(f"[Crop Node] query={query!r}, imageUrl={image_url!r}")

    use_image = bool(image_url and image_url.startswith(("http://", "https://")))

    if use_image and not query:
        query = (
            "Analyze the crop shown in the image and identify possible diseases, "
            "nutrient deficiencies, or pest issues."
        )

    try:
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "farm-manager-service/1.0",
                "Accept": "application/json",
            }
        ) as client:

            if use_image:
                r = await post_with_retry(
                    client,
                    CROP_IMAGE_SERVICE_URL,
                    data={"query": query, "mediaUrl": image_url},
                )
            else:
                r = await post_with_retry(
                    client,
                    CROP_TEXT_SERVICE_URL,
                    json={
                        "query": f"""
You are an agricultural crop advisor.

Rules:
- Any crop
- No government schemes
- Give practical advice only

Farmer query:
{query}
"""
                    },
                )

            r.raise_for_status()
            crop_response = normalize_crop_output(r.json())

    except httpx.ConnectError:
        crop_response = {
            "error": "Crop service temporarily unreachable. Please try again."
        }
    except Exception as e:
        crop_response = {"error": str(e)}

    return {**state, "crop_response": crop_response}


# -------------------------------------------------------------------
# Collector (LLM Synthesis)
# -------------------------------------------------------------------

def synthesize_final_response(
    intent: Literal["scheme", "crop", "both"] | None,
    scheme: dict | None,
    crop: dict | None,
) -> str:
    """
    Produces the final farmer-facing response based on intent.
    - scheme  → scheme only
    - crop    → crop only
    - both    → intelligently combine
    """

    scheme_text = scheme.get("response") if isinstance(scheme, dict) else None
    crop_text = crop.get("response") if isinstance(crop, dict) else None

    # -------------------------
    # Intent-driven shortcuts
    # -------------------------
    if intent == "crop":
        return crop_text or "No crop advisory information available."

    if intent == "scheme":
        return scheme_text or "No government scheme information available."

    # intent == "both" or unknown → combine
    if not COLLECTOR_MODEL:
        return (
            "\n\n".join(filter(None, [crop_text, scheme_text]))
            or "No information available."
        )

    # -------------------------
    # LLM synthesis for BOTH
    # -------------------------
    prompt = f"""
You are an agricultural assistant responding to a farmer.

Combine the following information into ONE clear, helpful response.

Rules:
- Do NOT repeat information.
- Clearly separate crop advice vs government support.
- If one section is missing, answer with what is available.
- Use simple, farmer-friendly language.

Crop Advisory:
{crop_text or "N/A"}

Government Schemes:
{scheme_text or "N/A"}

Final Answer:
"""

    try:
        r = COLLECTOR_MODEL.generate_content(prompt)
        return (r.text or "").strip()
    except Exception:
        return (
            "\n\n".join(filter(None, [crop_text, scheme_text]))
            or "No information available."
        )

def collector_node(state):
    """
    Final aggregation node.
    Delegates decision-making to synthesize_final_response based on intent.
    """
    intent = state.get("intent")

    final_answer = synthesize_final_response(
        intent=intent,
        scheme=state.get("scheme_response"),
        crop=state.get("crop_response"),
    )

    return {
        **state,
        "service_response": {
            "response": final_answer
        }
    }


# -------------------------------------------------------------------
# Error Node
# -------------------------------------------------------------------

def error_node(state):
    return {**state, "service_response": {"error": "cannot classify"}}


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------

def route_decision(state):
    return state.get("intent") or "none"


def route_from_scheme(state):
    return "crop_node" if state.get("intent") == "both" else "collector"


# -------------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------------

workflow = StateGraph(FarmerState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("scheme_node", scheme_node)
workflow.add_node("crop_node", crop_node)
workflow.add_node("collector", collector_node)
workflow.add_node("error_node", error_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "scheme": "scheme_node",
        "crop": "crop_node",
        "both": "scheme_node",
        "none": "error_node",
    },
)

workflow.add_conditional_edges(
    "scheme_node",
    route_from_scheme,
    {
        "crop_node": "crop_node",
        "collector": "collector",
    },
)

workflow.add_edge("crop_node", "collector")
workflow.add_edge("collector", END)
workflow.add_edge("error_node", END)

app = workflow.compile()
