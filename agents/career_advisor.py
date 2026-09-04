"""
Agent 2: Career Advisor

Takes the Resume Reviewer's output and produces forward-looking
guidance: an interview prep roadmap, and (stretch goal) a
personalized multi-week/month learning plan to close skill gaps.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import CHAT_MODEL
from prompts.templates import (
    CAREER_ADVISOR_SYSTEM_PROMPT,
    INTERVIEW_PREP_PROMPT,
    LEARNING_PLAN_PROMPT,
)
from agents.resume_reviewer import _safe_parse_json


class CareerAdvisorAgent:
    def __init__(self, temperature: float = 0.4):
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=temperature)

    def generate_interview_prep(
        self, resume_text: str, target_role: str, missing_skills: list, matched_skills: list
    ) -> dict:
        prompt = INTERVIEW_PREP_PROMPT.format(
            resume_text=resume_text[:4000],
            target_role=target_role or "Not specified",
            missing_skills=", ".join(missing_skills) if missing_skills else "None identified",
            matched_skills=", ".join(matched_skills) if matched_skills else "None identified",
        )
        messages = [
            SystemMessage(content=CAREER_ADVISOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return _safe_parse_json(response.content)

    def generate_learning_plan(
        self, target_role: str, missing_skills: list, matched_skills: list, timeframe: str = "3 months"
    ) -> dict:
        """Stretch goal: personalized N-month learning plan."""
        prompt = LEARNING_PLAN_PROMPT.format(
            target_role=target_role or "Not specified",
            missing_skills=", ".join(missing_skills) if missing_skills else "None identified",
            matched_skills=", ".join(matched_skills) if matched_skills else "None identified",
            timeframe=timeframe,
        )
        messages = [
            SystemMessage(content=CAREER_ADVISOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return _safe_parse_json(response.content)
