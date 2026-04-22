# AutoStream AI Agent
### ServiceHive / Inflx — ML Intern Assignment

A production-grade conversational AI agent that converts social media conversations into qualified leads for **AutoStream**, a SaaS video editing platform for content creators.

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd autostream-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Open .env and paste your Groq API key
```

Or export it directly:
```bash
export GROQ_API_KEY=gsk_...
```

### 3. Run the agent

```bash
python agent.py
```

### Sample conversation to test

```
You: Hi there!
You: What's included in the Pro plan?
You: That sounds perfect for my YouTube channel, I want to sign up
You: Nancy
You: nancy@example.com
You: YouTube
```

---

## Project Structure

```
autostream-agent/
├── agent.py            # Core LangGraph agent
├── knowledge_base.json # RAG knowledge base (pricing, policies, FAQs)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Architecture Explanation (~200 words)

The agent is built on **LangGraph**, a stateful graph execution framework built on top of LangChain. LangGraph was chosen over AutoGen because it provides explicit, inspectable control over conversation flow — each node in the graph represents a discrete step (intent detection, response generation, lead collection), making the system deterministic and easy to debug. AutoGen's multi-agent model would add unnecessary complexity for a single-agent, single-task workflow.

**State management** is handled through LangGraph's `StateGraph` with a typed `AgentState` dictionary that persists across all turns. It stores the full message history, the current intent label, and all three lead fields (`name`, `email`, `platform`) along with flags for `collecting_lead` and `lead_captured`. This ensures the agent never loses context between turns and the lead capture tool is never triggered prematurely — only after all three fields are collected.

**RAG** is implemented by loading `knowledge_base.json` at startup and injecting it as a static context block into the system prompt. For a production system, this would be replaced with a vector store (e.g. FAISS or Pinecone) for semantic retrieval over a larger document corpus.

The **intent router** uses a separate LLM call with a strict classification prompt, ensuring the main response node is never burdened with classification logic — keeping concerns separated and the system easy to extend.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) with a webhook-based architecture:

### Step-by-step integration

**1. Register a WhatsApp Business App**
- Create a Meta Developer account and set up a WhatsApp Business App
- Get a phone number ID and a permanent access token

**2. Set up a webhook endpoint**
Deploy a lightweight web server (FastAPI recommended) that exposes a `POST /webhook` endpoint:

```python
from fastapi import FastAPI, Request
from agent import build_graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()
graph = build_graph()

# In-memory session store (use Redis in production)
sessions: dict[str, AgentState] = {}

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    
    # Extract sender and message from WhatsApp payload
    entry = body["entry"][0]["changes"][0]["value"]
    message = entry["messages"][0]
    sender_id = message["from"]
    user_text = message["text"]["body"]
    
    # Load or initialise session state for this user
    if sender_id not in sessions:
        sessions[sender_id] = {
            "messages": [], "intent": "", "lead_name": "",
            "lead_email": "", "lead_platform": "",
            "lead_captured": False, "collecting_lead": False,
        }
    
    state = sessions[sender_id]
    state["messages"] = state["messages"] + [HumanMessage(content=user_text)]
    state = graph.invoke(state)
    sessions[sender_id] = state
    
    # Get last AI response and send back via WhatsApp API
    reply = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
    )
    send_whatsapp_message(sender_id, reply)
    return {"status": "ok"}
```

**3. Register the webhook with Meta**
- In the Meta Developer portal, set your webhook URL to `https://yourdomain.com/webhook`
- Set a verify token and handle the `GET /webhook` verification handshake

**4. Production considerations**
- Replace the in-memory `sessions` dict with **Redis** for persistence and horizontal scaling
- Use **ngrok** or **Cloudflare Tunnel** for local testing before deploying
- Deploy the FastAPI app on **Railway, Render, or AWS Lambda** for production
- Store leads in a proper CRM (HubSpot, Salesforce) instead of the mock function

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | Separate LLM node with strict 3-class classifier |
| RAG | JSON knowledge base injected into system prompt |
| State management | LangGraph `AgentState` persists across all turns |
| Tool calling guard | Lead tool fires only after all 3 fields collected |
| Code clarity | Modular nodes, typed state, clear routing logic |
| Deployability | WhatsApp webhook architecture documented above |

---

## Security Basics For GitHub

- Never commit `.env` or any real API key.
- Keep only placeholder secrets in `.env.example`.
- Rotate keys immediately if they were ever committed or shared.
- Enable secret scanning and push protection in GitHub repository settings.
- Use least-privilege API keys and separate keys for dev/prod.
