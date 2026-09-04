"""
Prompt Engineering layer.
Keeping prompts centralized here (rather than inline in agents)
makes them easy to iterate on and evaluate independently of code logic.
"""

RESUME_REVIEWER_SYSTEM_PROMPT = """You are an expert technical resume reviewer and ATS
(Applicant Tracking System) specialist with 15+ years of experience across tech,
product, and business roles. You give precise, evidence-based feedback — never vague
platitudes. You always ground your evaluation in the actual text of the resume and,
when provided, the actual text of the target job description.
"""

RESUME_ANALYSIS_PROMPT = """Analyze the following resume against the target job description context.

RESUME:
{resume_text}

TARGET JOB DESCRIPTION CONTEXT (retrieved via RAG, may be partial):
{jd_context}

TARGET ROLE: {target_role}

Perform the following analysis and return it as valid JSON with EXACTLY this schema:

{{
  "resume_score": <integer 0-100>,
  "score_breakdown": {{
    "relevant_experience": <integer 0-25>,
    "skills_match": <integer 0-25>,
    "quantified_impact": <integer 0-20>,
    "clarity_and_formatting": <integer 0-15>,
    "keyword_optimization": <integer 0-15>
  }},
  "strengths": [<3-5 short strings>],
  "weaknesses": [<3-5 short strings>],
  "missing_skills": [<list of specific skills/technologies/keywords present in the
      job description but absent or weak in the resume>],
  "matched_skills": [<list of skills the resume already demonstrates that match the role>],
  "quick_fixes": [<3-5 concrete, actionable rewrite suggestions, e.g.
      "Change 'responsible for API development' to 'Built 12 REST APIs serving 50K
      daily requests, reducing latency by 30%'">]
}}

Rules:
- Base "missing_skills" strictly on what's in the job description context. If no JD
  context was provided, infer typical requirements for the stated target role.
- Every weakness must be specific to THIS resume, not generic advice.
- Return ONLY the JSON object, no markdown fences, no commentary.
"""

CAREER_ADVISOR_SYSTEM_PROMPT = """You are a supportive but candid career coach who
specializes in translating resume gaps into actionable growth plans. You never
guilt-trip the user about weaknesses; you convert them into a concrete plan.
"""

INTERVIEW_PREP_PROMPT = """Based on this resume analysis, generate an interview
preparation roadmap.

RESUME SUMMARY: {resume_text}
TARGET ROLE: {target_role}
MISSING SKILLS: {missing_skills}
MATCHED SKILLS: {matched_skills}

Return valid JSON with EXACTLY this schema:

{{
  "likely_interview_topics": [<5-8 topics/technologies likely to come up, based on
      the target role and matched skills>],
  "technical_questions": [<5 realistic technical interview questions tailored to
      this candidate's actual background>],
  "behavioral_questions": [<3 behavioral questions relevant to gaps or transitions
      visible in the resume>],
  "questions_to_ask_interviewer": [<3 thoughtful questions the candidate could ask>],
  "one_week_prep_plan": [
    {{"day": "Day 1-2", "focus": <string>}},
    {{"day": "Day 3-4", "focus": <string>}},
    {{"day": "Day 5-6", "focus": <string>}},
    {{"day": "Day 7", "focus": <string>}}
  ]
}}

Return ONLY the JSON object, no markdown fences, no commentary.
"""

LEARNING_PLAN_PROMPT = """Create a personalized learning plan to close the following
skill gaps for someone targeting the role of {target_role}.

MISSING SKILLS: {missing_skills}
CURRENT SKILLS: {matched_skills}
TIMEFRAME: {timeframe}

Return valid JSON with EXACTLY this schema:

{{
  "plan_summary": <1-2 sentence overview of the strategy>,
  "milestones": [
    {{
      "period": <e.g. "Month 1", "Week 1-2">,
      "goals": [<2-4 specific goals>],
      "resources": [<2-3 concrete resource suggestions - course names, project ideas,
          certifications; be specific rather than generic>],
      "deliverable": <a concrete artifact to produce, e.g. "Deploy a small RAG app
          using LangChain and FAISS">
    }}
  ]
}}

Return ONLY the JSON object, no markdown fences, no commentary.
"""
