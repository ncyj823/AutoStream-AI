"""
AutoStream Conversational AI Agent
Built with LangGraph + Groq (Llama 3.3)
ServiceHive / Inflx Internship Assignment
"""

import json
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from operator import add

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base(path: str = "knowledge_base.json") -> str:
    with open(path, "r") as f:
        kb = json.load(f)

    lines = []
    lines.append(f"## Company: {kb['company']['name']}")
    lines.append(kb['company']['description'])
    lines.append("")
    lines.append("## Pricing Plans")
    for plan in kb["plans"]:
        lines.append(f"\n### {plan['name']} — ${plan['price_monthly']}/month")
        for feat in plan["features"]:
            lines.append(f"  - {feat}")
    lines.append("\n## Policies")
    for policy in kb["policies"]:
        lines.append(f"\n**{policy['topic']}:** {policy['details']}")
    lines.append("\n## FAQs")
    for faq in kb["faqs"]:
        lines.append(f"\nQ: {faq['question']}")
        lines.append(f"A: {faq['answer']}")

    return "\n".join(lines)


KNOWLEDGE_BASE = load_knowledge_base()

# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Lead captured: {name}, {email}, {platform}"

# ─────────────────────────────────────────────
# 3. LLM
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        api_key=api_key,
    )

# ─────────────────────────────────────────────
# 4. INTENT DETECTION
# ─────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing tool.

Classify the latest user message into EXACTLY ONE of:
- greeting        : casual hello / small talk
- inquiry         : questions about product, pricing, features, policies
- high_intent     : user wants to sign up, try the product, or is ready to purchase

Reply with ONLY the label — one word, no punctuation, no explanation."""


def detect_intent(user_message: str) -> str:
    response = get_llm().invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=user_message),
    ])
    intent = response.content.strip().lower()
    if intent not in ("greeting", "inquiry", "high_intent"):
        intent = "inquiry"
    return intent

# ─────────────────────────────────────────────
# 5. RESPONSE GENERATION
# ─────────────────────────────────────────────

AGENT_SYSTEM = f"""You are Maya, a friendly and knowledgeable sales assistant for AutoStream.

AutoStream is a SaaS platform with automated video editing tools for content creators.

Use ONLY the knowledge base below to answer product, pricing, or policy questions.
Do not invent features or prices.

--- KNOWLEDGE BASE ---
{KNOWLEDGE_BASE}
----------------------

Guidelines:
- Be warm, concise, and helpful.
- Keep responses under 120 words unless a detailed comparison is needed.
"""


def generate_response(messages: list) -> str:
    full_messages = [SystemMessage(content=AGENT_SYSTEM)] + messages
    response = get_llm().invoke(full_messages)
    return response.content

# ─────────────────────────────────────────────
# 6. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  AutoStream AI Assistant (type 'exit' to quit)")
    print("=" * 55)

    messages = []          # full conversation history
    collecting_lead = False
    lead_captured = False
    lead = {
        "name": "",
        "email": "",
        "platform": "",
    }
    # tracks which field we're waiting for: "name", "email", "platform"
    waiting_for = None

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        # ── Lead collection mode ──
        if collecting_lead and not lead_captured:
            if waiting_for == "name":
                lead["name"] = user_input
                waiting_for = "email"
                reply = f"Nice to meet you, {user_input}! What's your email address?"

            elif waiting_for == "email":
                lead["email"] = user_input
                waiting_for = "platform"
                reply = "And which platform do you primarily create for? (e.g. YouTube, Instagram, TikTok)"

            elif waiting_for == "platform":
                lead["platform"] = user_input
                mock_lead_capture(lead["name"], lead["email"], lead["platform"])
                lead_captured = True
                collecting_lead = False
                waiting_for = None
                reply = (
                    f"You're all set, {lead['name']}! 🚀 "
                    "Our team will reach out shortly to help you get started with AutoStream's Pro plan. "
                    "Is there anything else I can help you with?"
                )

            messages.append(AIMessage(content=reply))
            print(f"\nMaya: {reply}")
            continue

        # ── Normal conversation ──
        intent = detect_intent(user_input)

        if intent == "high_intent" and not lead_captured:
            collecting_lead = True
            waiting_for = "name"
            reply = "Awesome, let's get you set up! 🎉 What's your name?"
        else:
            reply = generate_response(messages)

        messages.append(AIMessage(content=reply))
        print(f"\nMaya: {reply}")


if __name__ == "__main__":
    main()
