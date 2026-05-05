# News Analyst Agent

A multi-step LLM agent that accepts a news topic, researches it using real web search, and produces a structured intelligence briefing — all driven by the Groq API (LLaMA 3.3 70B).

---

## What It Does

The agent breaks a single user input ("What is happening with X?") into six sequential steps where each step's output becomes the next step's input:

```
[User topic]
     │
     ▼
Step 1 ── LLM    ── topic → 4 targeted search queries
     │
     ▼
Step 2 ── TOOL   ── queries → raw web articles     (Serper API)
     │
     ▼
Step 3 ── LLM    ── raw articles → structured summaries
     │
     ▼
Step 4 ── LLM    ── summaries → themes + contradictions
     │
     ▼
Step 5 ── LLM    ── themes → Markdown briefing draft
     │
     ▼
Step 6 ── LLM×2  ── draft → critique → final polished briefing
     │
     ▼
[briefing_*.md  +  state_*.json]
```

No step can be removed without breaking the chain. Step 3 cannot run without Step 2's articles. Step 5 cannot run without Step 4's themes.

---

## File Structure

```
news_agent/
├── agent.py        — orchestrator: initialises state, runs all steps, saves output
├── steps.py        — the six pipeline step functions
├── tools.py        — external API wrappers (Groq LLM + Serper search)
├── requirements.txt
└── README.md
```

Each step is an isolated function in `steps.py` with a clear signature:
- **Input**: reads specific keys from the shared `state` dict
- **Output**: writes new keys into `state` and returns it

---

## Setup

### 1. Clone / unzip and install dependencies

```bash
cd news_agent
pip install -r requirements.txt
```

### 2. Get your API keys

| Service | Free tier? | Where to get it |
|---------|-----------|-----------------|
| **Groq** | Yes (free, no card needed) | https://console.groq.com → API Keys |
| **Serper** | Yes (2500 searches/month free) | https://serper.dev → Dashboard |

### 3. Set environment variables

**Mac/Linux:**
```bash
export GROQ_API_KEY="your-groq-key-here"
export SERPER_API_KEY="your-serper-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your-groq-key-here"
$env:SERPER_API_KEY="your-serper-key-here"
```

---

## Running the Agent

**With a topic as an argument:**
```bash
python agent.py "US Federal Reserve interest rate decisions 2025"
```

**Interactive mode:**
```bash
python agent.py
# → Enter a news topic to analyse: _
```

### Example topics that work well
```
python agent.py "ipl 2025"
python agent.py "India Pakistan tensions 2025"
python agent.py "Tesla sales decline"
python agent.py "OPEC oil production cuts"
```

---

## Output Files

Two files are written to the current directory after every run:

| File | Contents |
|------|----------|
| `briefing_<topic>_<timestamp>.md` | The final structured briefing (for human readers) |
| `state_<topic>_<timestamp>.json` | The complete shared state — all intermediate outputs from every step |

---

## Chain Design (step-by-step)

### Step 1 — Query Planner (LLM)
**Reads:** `state["topic"]`
**Writes:** `state["queries"]`, `state["angles"]`

Asks the LLM to produce 4 search queries covering four angles: recent news, background/context, expert opinion, and future outlook. A single broad search would miss important perspectives; decomposing the topic first makes Step 2 retrieve more useful data.

**System prompt forces:** JSON-only output with a fixed schema so Step 2 can iterate over `queries` reliably.

---

### Step 2 — Web Search (Tool Call — Serper API)
**Reads:** `state["queries"]`
**Writes:** `state["raw_results"]`, `state["all_articles"]`

Calls the Serper API for each query. Results are deduplicated by URL. If one query fails (network error, rate limit), the step logs the failure and continues with the remaining queries — the chain does not crash.

**Why a tool, not an LLM?** LLMs hallucinate recent events. Serper returns real, dated web content.

---

### Step 3 — Article Summariser (LLM)
**Reads:** `state["all_articles"]`, `state["topic"]`
**Writes:** `state["summaries"]`

Normalises every article into the same schema: `{headline, key_facts[], sentiment, source_name}`. The raw Serper snippets are inconsistent in length and relevance; this step makes them uniform so Step 4 can reason across all of them in a predictable way.

**Fallback:** If JSON parsing fails, constructs minimal summaries from raw fields so the chain continues.

---

### Step 4 — Theme Extractor (LLM)
**Reads:** `state["summaries"]`, `state["topic"]`
**Writes:** `state["themes"]`

Reasons *across* all summaries to identify 3–5 cross-cutting themes, contradictions between sources, consensus points, overall sentiment, and key named entities. This is the analytical core — it finds patterns invisible in any single article.

---

### Step 5 — Briefing Writer (LLM)
**Reads:** `state["themes"]`, `state["summaries"]`, `state["topic"]`
**Writes:** `state["draft_briefing"]`

Writes a structured Markdown briefing using a fixed template enforced in the system prompt. Synthesis (Step 4) and writing are kept separate so the writer can focus entirely on prose quality given already-structured inputs.

---

### Step 6 — Critic + Refiner (LLM × 2)
**Reads:** `state["draft_briefing"]`, `state["topic"]`
**Writes:** `state["critique"]`, `state["final_briefing"]`

Two sub-calls: first a critic identifies weaknesses, missing angles, and bias flags; then a refiner addresses those specific points and appends an "Editorial Notes" section documenting what changed. Self-review in one call produces worse results than separating critique from revision.

---

## Handling Failures

| Failure | Behaviour |
|---------|-----------|
| One search query fails | Logs warning, continues with remaining queries |
| Zero articles returned | Prints diagnostic message, exits gracefully |
| LLM returns malformed JSON | Falls back to minimal valid structure, chain continues |
| Missing API key | Clear error message before any API call is made |

---

## Known Limitations

1. **Snippet-only content** — Serper returns article snippets, not full text. The agent analyses summaries of summaries, which may miss nuance.
2. **4-query breadth limit** — Only 16 articles (4 per query) are retrieved. Niche or fast-moving topics may not be well-covered.
3. **No source credibility weighting** — A tabloid and a Reuters article are treated identically.
4. **Single-language only** — Queries and responses are English only.
5. **Groq context window** — Very long topics with many articles may approach token limits at Step 3.

---

## Demo Talking Points

**"Show me what Step 3 received and what it returned"**
→ Open `state_*.json`, find `"all_articles"` (Step 3 input) and `"summaries"` (Step 3 output). The LLM converted unstructured snippets into a uniform JSON array.

**"What happens if the tool call in Step 2 fails?"**
→ `step2_web_search` wraps each individual query in try/except. A failed query logs a warning and gets an empty list — the chain continues with whatever articles were retrieved. If *all* queries fail, `run_agent` checks for an empty `all_articles` and exits with a helpful message.

**"Why did you choose this prompt for Step 4?"**
→ The schema-enforced JSON output (`themes[]`, `contradictions[]`, `consensus`) gives Step 5's writer structured data it can address section-by-section, rather than having to re-parse free text.

**"Where does this chain break?"**
→ Step 3 is most fragile: if the LLM produces malformed JSON for a large article set, the fallback summaries lose the `key_facts` structure. Step 4 then has less to work with, producing shallower themes.
