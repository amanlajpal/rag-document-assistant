from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def build_pdf(filename: str, title: str, sections: list):
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = [Paragraph(title, title_style), Spacer(1, 10)]

    for heading, text in sections:
        story.append(Paragraph(heading, h2_style))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 8))

    doc.build(story)


def generate_all_sample_docs():
    out_dir = Path(__file__).resolve().parent.parent / "sample_data"
    out_dir.mkdir(exist_ok=True)

    # 1. Remote Work Policy
    remote_work_sections = [
        (
            "1. Purpose & Scope",
            "This policy outlines the guidelines and requirements for remote and telecommuting work at NovaTech Global. "
            "All full-time software engineers, product managers, and technical specialists are eligible for remote work.",
        ),
        (
            "2. Core Working Hours",
            "To ensure seamless collaboration across distributed teams, core working hours are established between "
            "10:00 AM and 4:00 PM Eastern Standard Time (EST). Team members must be reachable via Slack during these hours.",
        ),
        (
            "3. Home Office & Equipment Reimbursement",
            "NovaTech Global provides a one-time home office setup stipend of $1,500 to purchase ergonomic desks, chairs, "
            "and external 4K monitors. Additionally, a monthly high-speed internet stipend of $75 is credited to each employee.",
        ),
        (
            "4. Security, VPN & Data Protection",
            "All remote workers must connect to the corporate network via the approved NovaTech WireGuard VPN. "
            "Full-disk encryption (BitLocker on Windows, FileVault on macOS) is mandatory. Company data must never be accessed "
            "over unsecured public Wi-Fi without VPN active.",
        ),
        (
            "5. Paid Time Off (PTO) & Leave",
            "Employees receive 22 days of Paid Time Off (PTO) annually, plus 10 paid statutory holidays. Leave requests "
            "exceeding 3 consecutive business days must be submitted at least two weeks in advance.",
        ),
    ]
    build_pdf(
        str(out_dir / "company_remote_work_policy.pdf"),
        "NovaTech Global — Remote Work & Telecommuting Policy",
        remote_work_sections,
    )

    # 2. Employee Benefits Guide
    benefits_sections = [
        (
            "1. Health, Dental & Vision Insurance",
            "NovaTech Global covers 100% of medical, dental, and vision health premiums for all full-time employees, "
            "and 80% coverage for eligible dependents through the BlueCross BlueShield Comprehensive PPO network.",
        ),
        (
            "2. Annual Wellness & Fitness Credit",
            "Employees receive an annual $600 wellness credit. This budget can be used for gym memberships, fitness trackers, "
            "mental health counseling, yoga classes, or ergonomic accessories.",
        ),
        (
            "3. 401(k) Retirement Plan & Company Match",
            "NovaTech Global provides a 401(k) retirement savings plan through Fidelity. The company matches 100% of employee "
            "contributions up to 5% of base salary, with immediate 100% vesting from day one.",
        ),
        (
            "4. Learning & Professional Development",
            "Every employee is allocated a $2,000 annual professional development budget for books, certifications (AWS, GCP), "
            "and technical conference tickets.",
        ),
        (
            "5. Parental Leave",
            "NovaTech Global offers 16 weeks of 100% paid parental leave for all new parents following birth, adoption, or foster placement.",
        ),
    ]
    build_pdf(
        str(out_dir / "employee_benefits_guide.pdf"),
        "NovaTech Global — Comprehensive Employee Benefits Guide",
        benefits_sections,
    )

    # 3. AI Engineering Standards
    ai_sections = [
        (
            "1. Architecture Overview",
            "NovaTech Global builds production GenAI systems using LangChain, LangGraph, ChromaDB, and Groq high-speed LPU inference. "
            "All conversational document assistants must implement Retrieval-Augmented Generation (RAG).",
        ),
        (
            "2. Chunking & Ingestion Standards",
            "All document ingest pipelines must use RecursiveCharacterTextSplitter with a chunk size of 1000 characters "
            "and an overlap of 200 characters. Every chunk must retain document name and page number metadata.",
        ),
        (
            "3. Conversational State & Checkpointing",
            "Stateful LLM assistants must utilize LangGraph MemorySaver checkpoints indexed by thread_id to ensure conversation "
            "history and context are preserved across follow-up queries.",
        ),
        (
            "4. Model Selection Guidelines",
            "For inference, utilize Groq Llama-3.1-8b-instant for fast low-latency tasks, or open models such as OpenAI GPT-OSS "
            "when complex reasoning and multi-step tool calling are required.",
        ),
    ]
    build_pdf(
        str(out_dir / "ai_engineering_handbook.pdf"),
        "NovaTech Global — AI Engineering & RAG Architecture Standards",
        ai_sections,
    )

    print("Successfully generated all sample PDFs in sample_data directory!")


if __name__ == "__main__":
    generate_all_sample_docs()
