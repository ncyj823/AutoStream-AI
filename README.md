# 🎬 AutoStream AI Agent
### Social-to-Lead Agentic Workflow · ServiceHive / Inflx Internship Assignment

A production-grade conversational AI agent that converts social media conversations into qualified business leads — built with **LangGraph**, **Groq (Llama 3.3)**, and a **Streamlit chat UI**.

---

## 📸 Demo

> Agent detects high-intent, collects lead info across turns, fires `mock_lead_capture()` only after all 3 fields are collected.

```
You    → "Hi, what's in the Pro plan?"
Maya   → "The Pro plan is $79/month and includes unlimited videos, 4K export, AI captions..."

You    → "That sounds great, I want to sign up for my YouTube channel"
Maya   → [Intent: HIGH_INTENT 🔥] "Awesome, let's get you set up! What's your name?"

You    → "Nancy"
Maya   → "Nice to meet you, Nancy! What's your email address?"

You    → "nancy@example.com"
Maya   → "Which platform do you primarily create for?"

You    → "YouTube"
✅      → Lead captured: Nancy, nancy@example.com, YouTube
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/ncyj823/autostream-agent.git
cd autostream-agent
```

### 2. Create virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API key

Get a free API key from [console.groq.com](https://console.groq.com) → API Keys → Create Key

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

### 5. Run the Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 5b. Run CLI version (optional)

```bash
python agent.py
```

---

## 🗂️ Project Structure

```
autostream-agent/
├── app.py                  # Streamlit chat UI (main interface)
├── agent.py                # CLI agent (LangGraph core)
├── knowledge_base.json     # RAG knowledge base — pricing, policies, FAQs
├── requirements.txt        # All dependencies
├── .env.example            # API key template
├── logs/                   # Auto-generated conversation logs (JSON)
└── README.md
```

---

## 🧠 Architecture Explanation

The agent is built on **LangGraph**, a stateful graph execution framework built on top of LangChain. LangGraph was chosen over AutoGen because it provides explicit, inspectable control over conversation flow — each node in the graph represents a discrete step (intent detection, response generation, lead collection), making the system deterministic and easy to debug. AutoGen's multi-agent model adds unnecessary complexity for a focused single-agent, single-task workflow.

**State management** is handled through a combination of LangGraph's StateGraph and Streamlit's session_state. The state persists the full message history, the current intent label, and all three lead fields (name, email, platform) along with flags for collecting_lead and lead_captured. A waiting_for variable tracks exactly which field is expected next, ensuring the lead capture tool never fires prematurely.

**RAG** is implemented by loading knowledge_base.json at startup and injecting it as a static context block into the system prompt. The agent is explicitly instructed not to invent information outside this knowledge base. For production, this would be replaced with a vector store (FAISS or Pinecone) for semantic retrieval over a larger corpus.

**Intent detection** uses a separate LLM call with a strict 3-class classification prompt (greeting / inquiry / high_intent), keeping concerns separated from response generation. This also allows intent to be displayed in the UI per turn.

**Multi-turn context** is handled by scanning conversation history for signals (platform mentions, plan references) and injecting a brief context summary into the system prompt, allowing Maya to reference earlier parts of the conversation naturally.

---

## 💡 Agent Capabilities

| Feature | Implementation |
|---|---|
| Intent Detection | Separate LLM call → greeting / inquiry / high_intent |
| RAG Knowledge Base | knowledge_base.json injected into system prompt |
| Lead Capture Tool | Fires only after name + email + platform collected |
| Tool Guard | waiting_for state variable prevents premature firing |
| Multi-turn Context | History scanning → context summary injected per turn |
| Conversation Logger | Every session saved to logs/{session_id}.json |
| Streamlit UI | Dark chat interface with intent badges + lead card |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| UI | Streamlit |
| RAG | JSON knowledge base + prompt injection |
| State | LangGraph StateGraph + Streamlit session_state |
| Logging | Python JSON file logger |

---

## 🔌 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the architecture uses the **WhatsApp Business Cloud API** (Meta) with a webhook-based FastAPI server.

### Step-by-step integration

**1. Register a WhatsApp Business App**

Create a Meta Developer account, set up a WhatsApp Business App, and obtain a phone number ID and permanent access token.

**2. Deploy a FastAPI webhook server**

```python
from fastapi import FastAPI, Request
from agent import build_graph
from langchain_core.messages import HumanMessage, AIMessage
import redis, json

app = FastAPI()
graph = build_graph()
r = redis.Redis()

@app.get("/webhook")
async def verify(hub_mode: str, hub_challenge: str, hub_verify_token: str):
    if hub_verify_token == "your_verify_token":
        return int(hub_challenge)

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    entry = body["entry"][0]["changes"][0]["value"]
    message = entry["messages"][0]
    sender_id = message["from"]
    user_text = message["text"]["body"]

    raw = r.get(f"session:{sender_id}")
    state = json.loads(raw) if raw else {
        "messages": [], "intent": "", "lead_name": "",
        "lead_email": "", "lead_platform": "",
        "lead_captured": False, "collecting_lead": False,
    }

    state["messages"].append(HumanMessage(content=user_text))
    state = graph.invoke(state)
    r.setex(f"session:{sender_id}", 3600, json.dumps(state))

    reply = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    send_whatsapp_message(sender_id, reply)
    return {"status": "ok"}
```

**3. Register webhook with Meta**

In the Meta Developer portal set webhook URL to `https://yourdomain.com/webhook`, set verify token, and subscribe to messages events.

**4. Production considerations**

| Concern | Solution |
|---|---|
| Session persistence | Redis with TTL per user |
| Scale | Deploy on Railway / Render / AWS Lambda |
| Local testing | ngrok or Cloudflare Tunnel |
| CRM integration | Replace mock_lead_capture() with HubSpot / Salesforce API |

---

## 📊 Evaluation Checklist

| Criterion | Status |
|---|---|
| Agent reasoning & intent detection | ✅ 3-class LLM classifier per turn |
| Correct use of RAG | ✅ JSON KB injected, no hallucination |
| Clean state management | ✅ LangGraph + session_state + waiting_for guard |
| Proper tool calling logic | ✅ Fires only after all 3 fields collected |
| Code clarity & structure | ✅ Modular nodes, typed state, clean separation |
| Real-world deployability | ✅ WhatsApp webhook architecture with Redis + FastAPI |

---

## 👤 Author

**Nancy**
- GitHub: [github.com/ncyj823](https://github.com/ncyj823)
- LinkedIn: [linkedin.com/in/nancy-9b3688220](https://linkedin.com/in/nancy-9b3688220)
- Email: nancyjha24@gmail.com
