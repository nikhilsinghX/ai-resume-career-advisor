"""
Agent 1: Resume Reviewer

Responsible for scoring the resume, identifying strengths/weaknesses,
and diffing the resume's skills against retrieved job-description
context (RAG) to produce a missing-skills list.
"""
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import CHAT_MODEL
from prompts.templates import RESUME_REVIEWER_SYSTEM_PROMPT, RESUME_ANALYSIS_PROMPT


class ResumeReviewerAgent:
    def __init__(self, temperature: float = 0.2):
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=temperature)

    def analyze(self, resume_text: str, jd_context: str, target_role: str) -> dict:
        prompt = RESUME_ANALYSIS_PROMPT.format(
            resume_text=resume_text[:8000],  # guard against oversized context
            jd_context=jd_context[:6000] if jd_context else "No job description provided.",
            target_role=target_role or "Not specified",
        )

        messages = [
            SystemMessage(content=RESUME_REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return _safe_parse_json(response.content)


def _safe_parse_json(raw) -> dict:
    """
    LLMs occasionally wrap JSON in markdown fences despite instructions.
    Newer Gemini models may also return content as a list of blocks
    instead of a plain string. Normalize both cases before parsing.
    """
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        raw = "".join(parts)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse model output as JSON: {e}",
            "raw_output": raw,
        }