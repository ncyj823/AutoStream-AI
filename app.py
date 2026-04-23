"""
AutoStream AI Agent — Streamlit UI
ServiceHive / Inflx Internship Assignment
"""

import json
import os
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()



st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered",
)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0c0c0e;
    --surface: #141416;
    --surface2: #1c1c1f;
    --border: #2a2a2f;
    --accent: #e8ff47;
    --accent2: #47c8ff;
    --text: #f0f0f0;
    --muted: #6b6b7a;
    --user-bg: #1e1e24;
    --maya-bg: #141418;
    --success: #47ffaa;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background-color: var(--bg);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Header */
.agent-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.agent-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.5px;
}
.agent-header h1 span {
    color: var(--accent);
}
.agent-header p {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0.4rem 0 0 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    margin-bottom: 1.5rem;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}

/* Message bubbles */
.message-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.message-row.user { align-items: flex-end; }
.message-row.maya { align-items: flex-start; }

.message-bubble {
    max-width: 78%;
    padding: 0.75rem 1rem;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.55;
}
.message-bubble.user {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-bottom-right-radius: 4px;
    color: var(--text);
}
.message-bubble.maya {
    background: var(--maya-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text);
}

.message-label {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    padding: 0 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Intent badge */
.intent-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.intent-greeting  { background: #1a2a1a; color: #47ff9a; border: 1px solid #47ff9a44; }
.intent-inquiry   { background: #1a1a2a; color: var(--accent2); border: 1px solid #47c8ff44; }
.intent-high_intent { background: #2a2a0a; color: var(--accent); border: 1px solid #e8ff4744; }

/* Lead capture card */
.lead-card {
    background: linear-gradient(135deg, #141418, #1a1a10);
    border: 1px solid #e8ff4733;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}
.lead-card-title {
    color: var(--accent);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.lead-card-row {
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    padding: 3px 0;
    border-bottom: 1px solid #2a2a2f55;
}
.lead-card-row span:last-child { color: var(--text); }

/* Progress indicator */
.progress-bar {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.progress-step { color: var(--accent); }

/* Input area */
.stTextInput > div > div > input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #e8ff4720 !important;
}

/* Button */
.stButton > button {
    background-color: var(--accent) !important;
    color: #0c0c0e !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollable chat area */
.chat-scroll {
    max-height: 460px;
    overflow-y: auto;
    padding-right: 4px;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)



def load_knowledge_base(path: str = "knowledge_base.json") -> str:
    with open(path, "r") as f:
        kb = json.load(f)
    lines = []
    lines.append(f"## Company: {kb['company']['name']}")
    lines.append(kb['company']['description'])
    lines.append("\n## Pricing Plans")
    for plan in kb["plans"]:
        lines.append(f"\n### {plan['name']} — ${plan['price_monthly']}/month")
        for feat in plan["features"]:
            lines.append(f"  - {feat}")
    lines.append("\n## Policies")
    for policy in kb["policies"]:
        lines.append(f"\n**{policy['topic']}:** {policy['details']}")
    lines.append("\n## FAQs")
    for faq in kb["faqs"]:
        lines.append(f"\nQ: {faq['question']}\nA: {faq['answer']}")
    return "\n".join(lines)

KNOWLEDGE_BASE = load_knowledge_base()



def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to your .env file.")
        st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, api_key=api_key)



INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing tool.
Classify the latest user message into EXACTLY ONE of:
- greeting        : casual hello / small talk
- inquiry         : questions about product, pricing, features, policies
- high_intent     : user wants to sign up, try the product, or is ready to purchase
Reply with ONLY the label — one word, no punctuation."""

def detect_intent(user_message: str) -> str:
    response = get_llm().invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=user_message),
    ])
    intent = response.content.strip().lower()
    return intent if intent in ("greeting", "inquiry", "high_intent") else "inquiry"



def build_context_summary(messages: list) -> str:
    """Build a brief context note from past conversation for multi-turn awareness."""
    topics = []
    for m in messages:
        if isinstance(m, HumanMessage):
            txt = m.content.lower()
            if "youtube" in txt: topics.append("user mentioned YouTube")
            if "instagram" in txt: topics.append("user mentioned Instagram")
            if "tiktok" in txt: topics.append("user mentioned TikTok")
            if "pro" in txt: topics.append("user showed interest in Pro plan")
            if "basic" in txt: topics.append("user asked about Basic plan")
            if "price" in txt or "cost" in txt or "pricing" in txt:
                topics.append("user asked about pricing")
    if not topics:
        return ""
    unique = list(dict.fromkeys(topics))
    return "Context from earlier in this conversation: " + "; ".join(unique) + "."

AGENT_SYSTEM_TEMPLATE = """You are Maya, a friendly and knowledgeable sales assistant for AutoStream.
AutoStream is a SaaS platform with automated video editing tools for content creators.

Use ONLY the knowledge base below to answer product, pricing, or policy questions.
Do not invent features or prices.

--- KNOWLEDGE BASE ---
{kb}
----------------------

{context}

Guidelines:
- Be warm, concise, and helpful.
- Reference earlier context naturally when relevant (e.g. "Since you're on YouTube...").
- Keep responses under 120 words unless a detailed comparison is needed.
"""

def generate_response(messages: list) -> str:
    context = build_context_summary(messages)
    system = AGENT_SYSTEM_TEMPLATE.format(kb=KNOWLEDGE_BASE, context=context)
    full_messages = [SystemMessage(content=system)] + messages
    response = get_llm().invoke(full_messages)
    return response.content



def save_conversation_log(session_id: str, messages: list, lead: dict):
    """Save conversation + lead data to a JSON file."""
    os.makedirs("logs", exist_ok=True)
    log = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "lead_captured": lead.get("name") != "",
        "lead": lead,
        "conversation": [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ],
    }
    path = f"logs/{session_id}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def mock_lead_capture(name: str, email: str, platform: str):
    print(f"\n✅ Lead captured: {name}, {email}, {platform}\n")



def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "intents" not in st.session_state:
        st.session_state.intents = []        # parallel list to messages, user turns only
    if "collecting_lead" not in st.session_state:
        st.session_state.collecting_lead = False
    if "lead_captured" not in st.session_state:
        st.session_state.lead_captured = False
    if "waiting_for" not in st.session_state:
        st.session_state.waiting_for = None
    if "lead" not in st.session_state:
        st.session_state.lead = {"name": "", "email": "", "platform": ""}

init_session()


st.markdown("""
<div class="agent-header">
    <h1>Auto<span>Stream</span> AI</h1>
    <p>Social-to-Lead Agentic Workflow · Inflx by ServiceHive</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align:center">
        <span class="status-pill">
            <span class="status-dot"></span>
            Maya · Online
        </span>
    </div>
    """, unsafe_allow_html=True)



chat_html = '<div class="chat-scroll"><div class="chat-container">'

intent_idx = 0
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        intent = ""
        if intent_idx < len(st.session_state.intents):
            raw = st.session_state.intents[intent_idx]
            label_map = {
                "greeting": "Greeting",
                "inquiry": "Inquiry",
                "high_intent": "High Intent 🔥",
            }
            label = label_map.get(raw, raw)
            intent = f'<span class="intent-badge intent-{raw}">{label}</span>'
            intent_idx += 1
        chat_html += f"""
        <div class="message-row user">
            <div class="message-label">You {intent}</div>
            <div class="message-bubble user">{msg.content}</div>
        </div>"""
    else:
        chat_html += f"""
        <div class="message-row maya">
            <div class="message-label">Maya</div>
            <div class="message-bubble maya">{msg.content}</div>
        </div>"""

chat_html += "</div></div>"
st.markdown(chat_html, unsafe_allow_html=True)



if st.session_state.collecting_lead and not st.session_state.lead_captured:
    wf = st.session_state.waiting_for
    steps = {
        "name": "● Step 1/3 — Waiting for name",
        "email": "● Step 2/3 — Waiting for email",
        "platform": "● Step 3/3 — Waiting for platform",
    }
    st.markdown(f'<div class="progress-bar"><span class="progress-step">{steps.get(wf,"")}</span></div>', unsafe_allow_html=True)



if st.session_state.lead_captured:
    l = st.session_state.lead
    st.markdown(f"""
    <div class="lead-card">
        <div class="lead-card-title">✅ Lead Captured · mock_lead_capture() fired</div>
        <div class="lead-card-row"><span>Name</span><span>{l['name']}</span></div>
        <div class="lead-card-row"><span>Email</span><span>{l['email']}</span></div>
        <div class="lead-card-row"><span>Platform</span><span>{l['platform']}</span></div>
        <div class="lead-card-row"><span>Session</span><span>#{st.session_state.session_id}</span></div>
    </div>
    """, unsafe_allow_html=True)



st.markdown("<br>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        placeholder_map = {
            "name": "Enter your name...",
            "email": "Enter your email...",
            "platform": "YouTube, Instagram, TikTok...",
        }
        placeholder = placeholder_map.get(
            st.session_state.waiting_for, "Ask about pricing, features, or say hi..."
        )
        user_input = st.text_input("", placeholder=placeholder, label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("Send")


if submitted and user_input.strip():
    user_text = user_input.strip()
    st.session_state.messages.append(HumanMessage(content=user_text))

    
    if st.session_state.collecting_lead and not st.session_state.lead_captured:
        st.session_state.intents.append("high_intent")

        wf = st.session_state.waiting_for
        if wf == "name":
            st.session_state.lead["name"] = user_text
            st.session_state.waiting_for = "email"
            reply = f"Nice to meet you, {user_text}! What's your email address?"

        elif wf == "email":
            st.session_state.lead["email"] = user_text
            st.session_state.waiting_for = "platform"
            reply = "And which platform do you primarily create for? (e.g. YouTube, Instagram, TikTok)"

        elif wf == "platform":
            st.session_state.lead["platform"] = user_text
            mock_lead_capture(
                st.session_state.lead["name"],
                st.session_state.lead["email"],
                user_text,
            )
            save_conversation_log(
                st.session_state.session_id,
                st.session_state.messages,
                st.session_state.lead,
            )
            st.session_state.lead_captured = True
            st.session_state.collecting_lead = False
            st.session_state.waiting_for = None
            reply = (
                f"You're all set, {st.session_state.lead['name']}! 🚀 "
                "Our team will reach out shortly to help you get started with AutoStream's Pro plan. "
                "Is there anything else I can help you with?"
            )

    
    else:
        intent = detect_intent(user_text)
        st.session_state.intents.append(intent)

        if intent == "high_intent" and not st.session_state.lead_captured:
            st.session_state.collecting_lead = True
            st.session_state.waiting_for = "name"
            reply = "Awesome, let's get you set up! 🎉 What's your name?"
        else:
            reply = generate_response(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=reply))

   
    save_conversation_log(
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.lead,
    )

    st.rerun()



with st.sidebar:
    st.markdown("### 🎬 AutoStream Agent")
    st.markdown(f"**Session** `#{st.session_state.session_id}`")
    st.markdown(f"**Turns** `{len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])}`")
    st.markdown(f"**Model** `llama-3.3-70b`")
    st.markdown(f"**Framework** `LangGraph · Groq`")

    st.divider()

    if st.session_state.lead_captured:
        st.markdown("**Lead Status** ✅ Captured")
        st.markdown(f"**Name** {st.session_state.lead['name']}")
        st.markdown(f"**Platform** {st.session_state.lead['platform']}")
    elif st.session_state.collecting_lead:
        st.markdown("**Lead Status** 🔄 Collecting...")
    else:
        st.markdown("**Lead Status** ⏳ Waiting")

    st.divider()

    if st.button("🔄 New Conversation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built for ServiceHive · Inflx Assignment")
