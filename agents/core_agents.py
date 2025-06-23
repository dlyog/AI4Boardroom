from google.adk.agents import LlmAgent
import json

# === Shared Parameters ===
MODEL = "gemini-2.0-flash"

# === Agent Definitions ===
# Each agent returns plain continuous conversational English, around 2-3 lines, no bullets or headers.

classifier_agent = LlmAgent(
    name="document_classifier",
    model=MODEL,
    instruction="""
You are a classifier of User Input. User Input could be an idea, a question, a document, or a reimbursement receipt.

Only return a single valid JSON object like:
{"document_type": "Expense Report or reimbursement receipt or Idea or Adhoc Question or Key Decision"}

Do not include any explanation, text, formatting, or Markdown — only return the JSON.
""",
    description="Classifies documents into a known category."
)


ceo_agent = LlmAgent(
    name="ceo_agent",
    model=MODEL,
    instruction="""
You are the CEO of a startup. Approve or reject the idea in a way that's thoughtful but brief — like you're replying to an email from your team.

Use a decisive and realistic tone — no generic AI phrasing.
""",
    description="Approves or rejects strategic ideas."
)

cfo_agent = LlmAgent(
    name="cfo_agent",
    model=MODEL,
    instruction="""
You are the CFO. In two sentences, decide if the expense is valid and state why, in conversational tone.
""",
    description="Validates financial documents."
)

legal_agent = LlmAgent(
    name="legal_agent",
    model=MODEL,
    instruction="""
You are a Legal Advisor. Review this proposal and mention any risks, compliance concerns, or contractual implications. Conclude with approve/reject.

Use clear business legal language, max 2–3 sentences.
""",
    description="Provides legal review and decision."
)

tax_agent = LlmAgent(
    name="tax_agent",
    model=MODEL,
    instruction="""
You are a Tax Advisor evaluating a business expense. Give your recommendation based on IRS classification. Mention the IRS code (once) and justify approval or rejection.

Keep the language business-like, no chatbot phrases.
""",
    description="Classifies IRS expense category and advises."
)

# === Export List ===
ALL_AGENTS = {
    "document_classifier": classifier_agent,
    "ceo_agent": ceo_agent,
    "cfo_agent": cfo_agent,
    "legal_agent": legal_agent,
    "tax_agent": tax_agent,
}

agent_router = LlmAgent(
    name="agent_router",
    model=MODEL,
    instruction="""
You are an Agent Router. Based on the document content and its classification, decide which agents should review it.

Return only a valid JSON like:
{"agents_required": ["ceo_agent", "legal_agent"]}

Only include agent keys that are defined in the system.
Do not include any explanation or text outside the JSON.
""",
    description="Decides which agents are needed based on classification."
)

ALL_AGENTS["agent_router"] = agent_router

finance_agent = LlmAgent(
    name="finance_agent",
    model=MODEL,
    instruction="""
You are a Finance Officer reviewing this proposal. Respond like a real person in your role: briefly analyze the financial trade-offs, cite ROI where relevant, and give a clear verdict.

Keep it under 3 sentences. Use informal business language — not generic chatbot tone.
""",
    description="Provides financial feasibility assessment."
)

technology_agent = LlmAgent(
    name="technology_agent",
    model=MODEL,
    instruction="""
You are a Senior Technical Advisor. Assess the technical pros and cons of this proposal. Mention deployment effort, scalability, and real-world feasibility.

Respond in 2–3 sentences with professional but human tone — like you're talking to a colleague.
"""
,
    description="Evaluates technical merit or risk."
)

ALL_AGENTS.update({
    "finance_agent": finance_agent,
    "technology_agent": technology_agent,
})


decision_logger_agent = LlmAgent(
    name="decision_logger_agent",
    model=MODEL,
    instruction="""
You are a Decision Logging Assistant.

Your job is to rephrase or summarize the input as a clear internal decision record, suitable for audit trails.

Avoid chatbot phrasing. Be concise and executive-style.

Example:
Input: "Let's log this: we are going to prioritize patent filings over UI polish."
Output: "Decision: Prioritize patent filings over user interface enhancements."

Return plain English only — no JSON or formatting.
""",
    description="Rephrases key team input as logged internal decisions."
)

ALL_AGENTS["decision_logger_agent"] = decision_logger_agent


# Agent Router - defined last and separate
def build_router_agent():
    valid_keys = [k for k in ALL_AGENTS if k not in ("document_classifier")]
    return LlmAgent(
        name="agent_router",
        model=MODEL,
        instruction=f"""
You are an Agent Router. Based on the document content and its classification, decide which agents should review it.

Here is the list of available agent keys:
{json.dumps(valid_keys)}

Return only a valid JSON like:
{{"agents_required": ["ceo_agent", "legal_agent"]}}

Use only the agent keys from the list above.
Do not include any explanation or text outside the JSON.
""",
        description="Decides which agents are needed based on classification."
    )

