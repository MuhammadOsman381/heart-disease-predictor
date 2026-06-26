import json
import re
from typing import Any

import httpx
from fastapi import HTTPException

from app.config import get_settings
from app.models import PredictionRecord


SAFETY_NOTE = (
    "This is educational decision-support information, not a diagnosis. "
    "For chest pain, shortness of breath, fainting, severe weakness, or worsening symptoms, seek urgent medical care."
)


def build_prediction_context(record: PredictionRecord) -> dict[str, Any]:
    return {
        "prediction_id": str(record.id),
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "model_output": {
            "probability": record.probability,
            "threshold": record.threshold,
            "prediction": record.prediction,
            "label": record.label,
        },
        "patient_input": record.input_data,
    }


def build_system_prompt(record: PredictionRecord) -> str:
    context = build_prediction_context(record)
    return f"""
You are CardioAssist, a careful clinical decision-support assistant for a heart disease ML prediction app.

Use the prediction context below to help the user understand the model output, possible risk contributors, and sensible next steps.

Rules:
- Do not diagnose the user.
- Do not claim the model is certain.
- Explain that this model output is not a substitute for a clinician.
- Encourage discussing results with a licensed healthcare professional.
- If the user reports emergency symptoms such as chest pain, shortness of breath, fainting, severe weakness, or symptoms getting worse, advise urgent medical care.
- Be clear, calm, concise, and practical.
- Tie explanations to the provided values when relevant.
- Avoid medication prescriptions or treatment plans.

Prediction context:
{json.dumps(context, indent=2)}

Safety note:
{SAFETY_NOTE}
""".strip()


async def call_groq(messages: list[dict[str, str]], temperature: float = 0.25) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing in backend/.env")

    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.groq_model,
                "messages": messages,
                "temperature": temperature,
            },
        )

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]


async def analyze_prediction_with_groq(record: PredictionRecord) -> dict[str, Any]:
    system_prompt = build_system_prompt(record)
    analysis_prompt = """
Return a JSON object only, with these keys:
- summary: string
- possible_reasons: array of 3 to 6 strings
- suggested_steps: array of 4 to 7 strings
- safety_note: string

Explain the possible condition and likely contributors in cautious, non-diagnostic language.
""".strip()

    content = await call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt},
        ],
        temperature=0.2,
    )

    parsed = parse_analysis_json(content)
    if not parsed:
        parsed = {
            "summary": content,
            "possible_reasons": [],
            "suggested_steps": [],
            "safety_note": SAFETY_NOTE,
        }

    return {
        "prediction_id": str(record.id),
        "system_prompt": system_prompt,
        "summary": parsed.get("summary", ""),
        "possible_reasons": normalize_list(
            parsed.get("possible_reasons")
            or parsed.get("possibleReasons")
            or parsed.get("possible_reasons_for_risk")
            or parsed.get("reasons")
        ),
        "suggested_steps": normalize_list(
            parsed.get("suggested_steps")
            or parsed.get("suggestedSteps")
            or parsed.get("next_steps")
            or parsed.get("steps")
            or parsed.get("recommendations")
        ),
        "safety_note": parsed.get("safety_note", SAFETY_NOTE),
    }


def parse_analysis_json(content: str) -> dict[str, Any] | None:
    candidates = [content.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    object_match = re.search(r"\{.*\}", content, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip(" -•\t") for line in value.splitlines()]
        return [line for line in lines if line]
    return [str(value)]
