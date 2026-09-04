"""
AI Resume & Career Advisor
---------------------------
Architecture: User -> RAG (job description retrieval) -> LLM Agents -> Response

Agent Workflow:
  1. Resume Reviewer Agent  -> scores resume, finds missing/matched skills
  2. Career Advisor Agent   -> interview prep roadmap + personalized learning plan

Run with:
    streamlit run app.py
"""
import streamlit as st

from config import GOOGLE_API_KEY
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_multiple_pdfs
from rag.vector_store import build_vector_store, get_retriever, retrieve_relevant_jd_context
from agents.resume_reviewer import ResumeReviewerAgent
from agents.career_advisor import CareerAdvisorAgent

st.set_page_config(page_title="AI Resume & Career Advisor", page_icon="📄", layout="wide")


# ---------- Sidebar: inputs ----------
st.sidebar.title("📄 AI Resume & Career Advisor")
st.sidebar.caption("Upload your resume + target job descriptions to get a tailored analysis.")

if not GOOGLE_API_KEY:
    st.sidebar.error("GOOGLE_API_KEY not set. Add it to a .env file (see .env.example).")

target_role = st.sidebar.text_input("Target Role", placeholder="e.g. Backend Software Engineer")

resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_files = st.sidebar.file_uploader(
    "Upload Job Description(s) (PDF)", type=["pdf"], accept_multiple_files=True
)

timeframe = st.sidebar.selectbox("Learning plan timeframe", ["1 month", "3 months", "6 months"], index=1)

run_button = st.sidebar.button("🚀 Analyze Resume", type="primary", use_container_width=True)


# ---------- Session state ----------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "interview_prep" not in st.session_state:
    st.session_state.interview_prep = None
if "learning_plan" not in st.session_state:
    st.session_state.learning_plan = None


st.title("AI Resume & Career Advisor")
st.caption("RAG-grounded resume scoring, skill-gap analysis, and interview prep — powered by LangChain + Google Gemini.")


# ---------- Main pipeline ----------
if run_button:
    if not resume_file:
        st.error("Please upload a resume PDF first.")
        st.stop()
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY — set it in your .env file.")
        st.stop()

    with st.spinner("Reading resume..."):
        resume_text = extract_text_from_pdf(resume_file)

    jd_context = ""
    if jd_files:
        with st.spinner("Indexing job description(s) for RAG retrieval..."):
            jd_docs = extract_text_from_multiple_pdfs(jd_files)
            vector_store = build_vector_store(jd_docs)
            retriever = get_retriever(vector_store)
            query = target_role or "job requirements and required skills"
            jd_context = retrieve_relevant_jd_context(retriever, query)

    with st.spinner("Resume Reviewer Agent analyzing your resume..."):
        reviewer = ResumeReviewerAgent()
        analysis = reviewer.analyze(resume_text, jd_context, target_role)
        st.session_state.analysis = analysis

    if "error" not in analysis:
        with st.spinner("Career Advisor Agent building your interview roadmap..."):
            advisor = CareerAdvisorAgent()
            st.session_state.interview_prep = advisor.generate_interview_prep(
                resume_text,
                target_role,
                analysis.get("missing_skills", []),
                analysis.get("matched_skills", []),
            )

        with st.spinner("Career Advisor Agent building your personalized learning plan..."):
            st.session_state.learning_plan = advisor.generate_learning_plan(
                target_role,
                analysis.get("missing_skills", []),
                analysis.get("matched_skills", []),
                timeframe,
            )


# ---------- Display results ----------
analysis = st.session_state.analysis

if analysis and "error" in analysis:
    st.error(analysis["error"])
    with st.expander("Raw model output"):
        st.code(analysis.get("raw_output", ""))

elif analysis:
    tab1, tab2, tab3 = st.tabs(["📊 Resume Score", "🎤 Interview Prep", "📚 Learning Plan"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Resume Score", f"{analysis.get('resume_score', 0)} / 100")
            breakdown = analysis.get("score_breakdown", {})
            for k, v in breakdown.items():
                st.progress(min(v / 25, 1.0) if v <= 25 else min(v / 100, 1.0), text=f"{k.replace('_', ' ').title()}: {v}")

        with col2:
            st.subheader("✅ Strengths")
            for s in analysis.get("strengths", []):
                st.markdown(f"- {s}")

            st.subheader("⚠️ Weaknesses")
            for w in analysis.get("weaknesses", []):
                st.markdown(f"- {w}")

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            st.subheader("❌ Missing Skills")
            missing = analysis.get("missing_skills", [])
            if missing:
                st.markdown("\n".join(f"- {m}" for m in missing))
            else:
                st.caption("None identified (upload a job description for a sharper diff).")
        with colB:
            st.subheader("✔️ Matched Skills")
            matched = analysis.get("matched_skills", [])
            if matched:
                st.markdown("\n".join(f"- {m}" for m in matched))
            else:
                st.caption("None identified.")

        st.divider()
        st.subheader("🔧 Quick Fixes")
        for fix in analysis.get("quick_fixes", []):
            st.markdown(f"- {fix}")

    with tab2:
        prep = st.session_state.interview_prep
        if prep and "error" not in prep:
            st.subheader("Likely Interview Topics")
            st.write(", ".join(prep.get("likely_interview_topics", [])))

            st.subheader("Technical Questions")
            for i, q in enumerate(prep.get("technical_questions", []), 1):
                st.markdown(f"{i}. {q}")

            st.subheader("Behavioral Questions")
            for i, q in enumerate(prep.get("behavioral_questions", []), 1):
                st.markdown(f"{i}. {q}")

            st.subheader("Questions to Ask the Interviewer")
            for i, q in enumerate(prep.get("questions_to_ask_interviewer", []), 1):
                st.markdown(f"{i}. {q}")

            st.subheader("1-Week Prep Plan")
            for item in prep.get("one_week_prep_plan", []):
                st.markdown(f"**{item.get('day')}:** {item.get('focus')}")
        else:
            st.info("Run the analysis to generate your interview prep roadmap.")

    with tab3:
        plan = st.session_state.learning_plan
        if plan and "error" not in plan:
            st.write(plan.get("plan_summary", ""))
            for milestone in plan.get("milestones", []):
                with st.expander(f"📌 {milestone.get('period')}"):
                    st.markdown("**Goals:**")
                    for g in milestone.get("goals", []):
                        st.markdown(f"- {g}")
                    st.markdown("**Resources:**")
                    for r in milestone.get("resources", []):
                        st.markdown(f"- {r}")
                    st.markdown(f"**Deliverable:** {milestone.get('deliverable')}")
        else:
            st.info("Run the analysis to generate your personalized learning plan.")

else:
    st.info("👈 Upload your resume (and optionally target job descriptions) in the sidebar, then click **Analyze Resume**.")
