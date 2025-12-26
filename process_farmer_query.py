from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import httpx
import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# Configure Google Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _validate_gemini_api_key() -> bool:
    """
    Best-effort validation that the Gemini API key is configured.

    IMPORTANT: Even if validation fails, the service should continue working
    using a rule-based fallback so that farmer queries are still answered.
    """
    if not GEMINI_API_KEY:
        print("[Gemini] GEMINI_API_KEY is not set in the environment – Gemini features will be disabled, using rule-based fallback.")
        return False

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Optional lightweight sanity check. If it fails, we log and fall back.
        test_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
        _ = test_model  # avoid 'unused' warnings
        print("[Gemini] GEMINI_API_KEY configured. Gemini intent classification enabled.")
        return True
    except Exception as e:
        # Most failures here are due to invalid / unauthorized API keys or networking issues
        print(f"[Gemini] API key validation failed ({e!r}) – disabling Gemini and using rule-based fallback.")
        return False

GEMINI_API_KEY_VALID = _validate_gemini_api_key()

class FarmerState(TypedDict):
    text: str
    imageUrl: str | None
    intent: Literal["scheme", "crop", "both"] | None
    entities: dict
    scheme_response: dict | None
    crop_response: dict | None
    service_response: dict

# Service URLs (from environment or defaults)

SCHEME_SERVICE_URL = os.getenv(
    "SCHEME_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/query-government-schemes"
)

# 🔽🔽🔽 CHANGE START: split crop services by modality 🔽🔽🔽

CROP_TEXT_SERVICE_URL = os.getenv(
    "CROP_TEXT_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/ask-consultant"
)

CROP_IMAGE_SERVICE_URL = os.getenv(
    "CROP_IMAGE_SERVICE_URL",
    "https://api.alumnx.com/api/agrigpt/ask-with-image"
)

# 🔼🔼🔼 CHANGE END 🔼🔼🔼


# Initialize Gemini model for intent classification (only if key looked valid)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
gemini_model = (
    genai.GenerativeModel(GEMINI_MODEL_NAME) if GEMINI_API_KEY_VALID else None
)

def _rule_based_intent(query: str) -> Literal["scheme", "crop", "both", "none"] | None:
    """
    Simple keyword-based classifier used when Gemini is unavailable.
    This prevents noisy configuration errors and still routes queries reasonably.
    """
    q = query.lower()

    scheme_keywords = [
        "scheme", "subsidy", "government", "yojana", "pm-kisan", "pm kisan",
        "financial assistance", "grant", "loan", "benefit", "msme"
    ]
    crop_keywords = [
        "crop", "disease", "pest", "yield", "harvest", "cultivation",
        "sowing", "planting", "fertilizer", "fertiliser", "irrigation",
        "season", "soil", "variety"
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

def supervisor_node(state):
    query = state.get("text", "").strip()
    image_url = state.get("imageUrl")
    print(f"[Supervisor] In supervisor node. Query: {query!r}, ImageUrl: {image_url!r}")
    
    # Handle case where there's only an image but no text
    if not query and image_url:
        print("[Supervisor] Image provided without text – defaulting to 'crop' intent (images typically used for crop disease/pest identification).")
        return {**state, "intent": "crop", "entities": {}}
    
    # Handle empty/null queries with no image
    if not query:
        print("[Supervisor] Empty query received and no image – skipping classification.")
        return {**state, "intent": None, "entities": {}}
    
    # If Gemini is not available, use rule-based classifier and avoid noisy config errors
    if not GEMINI_API_KEY or not GEMINI_API_KEY_VALID or gemini_model is None:
        intent = _rule_based_intent(query)
        print(f"[Supervisor] Using rule-based intent classification. Intent: {intent!r}")
        return {**state, "intent": intent, "entities": {}}

    # LLM classifies intent (only when Gemini is healthy)
    # If there's an image, include it in the classification
    prompt_parts = []
    
    # Add image if available
    if image_url:
        try:
            import base64
            import mimetypes
            
            # Download the image
            with httpx.Client() as client:
                img_response = client.get(image_url, timeout=10.0)
                img_response.raise_for_status()
                image_data = img_response.content
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(image_url)
            if not mime_type:
                mime_type = "image/jpeg"  # Default fallback
            
            # Add image to prompt
            prompt_parts.append({
                "mime_type": mime_type,
                "data": base64.b64encode(image_data).decode()
            })
            print(f"[Supervisor] Image loaded for classification: {mime_type}")
        except Exception as e:
            print(f"[Supervisor] Warning: Could not load image for classification: {e}. Proceeding with text-only classification.")
    
    # Add text prompt
    text_prompt = f"""Classify the following farmer query into one of these intents: "scheme", "crop", "both" or "none".

Query: {query}

Rules:
- "scheme": Questions about government schemes, subsidies, financial assistance, programs in India.
- "crop": Questions about crop cultivation, seasons, growing conditions, crop-specific advice, crop diseases etc.
- "both": Queries that require information from both scheme and crop categories.
- "none": Queries that don't fit scheme or crop categories.

Examples:
1. Query : "What are the government schemes in Andhra Pradesh?" , intent : "scheme",
2. Query : "What is Citrus Canker?", intent : "crop",
3. Query : "Tell me what government schemes help me to address citrus canker?", intent : "both"
Respond with ONLY one word: scheme, crop, both, or none."""
    
    prompt_parts.append(text_prompt)

    try:
        response = gemini_model.generate_content(prompt_parts)
        raw_text = (response.text or "")
        print(f"[Supervisor] Raw Gemini response: {raw_text!r}")
        intent = raw_text.strip().lower()
        
        # Validate intent
        if intent not in ["scheme", "crop", "both", "none"]:
            print(f"[Supervisor] Unexpected intent value from Gemini: {intent!r}")
            intent = None
        elif intent == "none":
            print("[Supervisor] Gemini classified intent as 'none'.")
            intent = None
        else:
            print(f"[Supervisor] Classified intent: {intent!r}")

        return {**state, "intent": intent, "entities": {}}
    except Exception:
        # Other errors - log and return None intent
        import traceback
        print("[Supervisor] Unexpected error during intent classification:")
        traceback.print_exc()
        return {**state, "intent": None, "entities": {}, "service_response": {"error": "Gemini classification failed"}}

async def scheme_node(state):
    query = state.get("text", "")
    print(f"in scheme node. query: {query}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SCHEME_SERVICE_URL,
                json={"query": query},
                timeout=30.0
            )
            response.raise_for_status()
            scheme_response = response.json()
    except Exception as e:
        scheme_response = {"error": str(e)}
    
    return {**state, "scheme_response": scheme_response}

async def crop_node(state):
    query = state.get("text", "")
    image_url = state.get("imageUrl")
    print(f"in crop node. query: {query}, imageUrl: {image_url}")

    # 🔽🔽🔽 CHANGE START: decide modality 🔽🔽🔽
    use_image = bool(image_url and image_url.startswith(("http://", "https://")))
    # 🔼🔼🔼 CHANGE END 🔼🔼🔼

    try:
        async with httpx.AsyncClient() as client:

            # 🔽🔽🔽 CHANGE START: route to correct endpoint 🔽🔽🔽
            if use_image:
                url = CROP_IMAGE_SERVICE_URL
                payload = {
                    "query": query,
                    "mediaUrl": image_url
                }
                # ask-with-image expects form-style or JSON depending on backend
                response = await client.post(
                    url,
                    json=payload,  # switched to JSON for consistency
                    timeout=30.0
                )
            else:
                url = CROP_TEXT_SERVICE_URL
                payload = {
                    "query": query
                }
                # ask-consultant is TEXT ONLY
                response = await client.post(
                    url,
                    json=payload,
                    timeout=30.0
                )
            # 🔼🔼🔼 CHANGE END 🔼🔼🔼

            response.raise_for_status()
            crop_response = response.json()

    except Exception as e:
        import traceback
        print(f"[Crop Node] Error calling crop service: {e}")
        traceback.print_exc()
        crop_response = {"error": str(e)}

    return {**state, "crop_response": crop_response}



def error_node(state):
    return {**state, "service_response": {"error": "cannot classify"}}

def collector_node(state):
    """Collect and combine responses from scheme and/or crop nodes"""
    print("in collector node")
    intent = state.get("intent")
    scheme_response = state.get("scheme_response")
    crop_response = state.get("crop_response")
    
    # Ensure responses were generated
    if intent == "scheme":
        if scheme_response is None:
            service_response = {"error": "Scheme service did not generate a response"}
        else:
            service_response = {"scheme": scheme_response}
    elif intent == "crop":
        if crop_response is None:
            service_response = {"error": "Crop service did not generate a response"}
        else:
            service_response = {"crop": crop_response}
    elif intent == "both":
        # Combine both responses
        combined = {}
        if scheme_response is not None:
            combined["scheme"] = scheme_response
        else:
            combined["scheme"] = {"error": "Scheme service did not generate a response"}
        
        if crop_response is not None:
            combined["crop"] = crop_response
        else:
            combined["crop"] = {"error": "Crop service did not generate a response"}
        
        service_response = combined
    else:
        # For error cases or unknown intents
        service_response = state.get("service_response", {"error": "Unknown intent or classification error"})
    
    return {**state, "service_response": service_response}

# Routing function from supervisor
def route_decision(state):
    intent = state.get("intent")
    if intent is None:
        return "none"
    return intent

# Routing function from scheme_node - if intent is "both", go to crop_node, else go to collector
def route_from_scheme(state):
    intent = state.get("intent")
    if intent == "both":
        return "crop_node"
    return "collector"

# Build graph
workflow = StateGraph(FarmerState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("scheme_node", scheme_node)
workflow.add_node("crop_node", crop_node)
workflow.add_node("collector", collector_node)
workflow.add_node("error_node", error_node)

workflow.set_entry_point("supervisor")

# Supervisor routes based on intent classification
workflow.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "scheme": "scheme_node",
        "crop": "crop_node",
        "both": "scheme_node",  # Route to scheme_node first, then it will route to crop_node
        "none": "error_node"
    }
)

# scheme_node routes: if intent is "both", go to crop_node, else go to collector
workflow.add_conditional_edges(
    "scheme_node",
    route_from_scheme,
    {
        "crop_node": "crop_node",
        "collector": "collector"
    }
)

# crop_node always routes to collector
workflow.add_edge("crop_node", "collector")

# Collector and error node are terminal
workflow.add_edge("collector", END)
workflow.add_edge("error_node", END)

app = workflow.compile()