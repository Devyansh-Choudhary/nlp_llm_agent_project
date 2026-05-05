"""
steps.py — The six pipeline steps of the News Analyst Agent

Each function:
  - receives the shared state dict
  - reads only what it needs from state
  - writes its output back into state under a named key
  - returns the updated state

Step 1  (LLM)  : topic          → search queries
Step 2  (Tool) : queries         → raw web search results
Step 3  (LLM)  : raw articles    → structured summaries
Step 4  (LLM)  : summaries       → themes + contradictions
Step 5  (LLM)  : themes          → briefing draft
Step 6  (LLM)  : draft + critique → final polished briefing
"""

import json
from datetime import datetime
from tools import call_groq, parse_json_response, serper_search


# ── Step 1 — Query Planner ─────────────────────────────────────────────────────

def step1_generate_queries(state: dict) -> dict:
    """
    LLM Call 1: Decompose the user's topic into 4 targeted search queries.

    Why a separate step?
    A single broad search on the raw topic misses important angles.
    The LLM breaks the topic into: recent events, background/context,
    expert opinion, and future implications — ensuring the search phase
    retrieves diverse, complementary information.

    Input  : state["topic"]
    Output : state["queries"]  — list of 4 query strings
             state["angles"]   — list of 4 angle labels (for traceability)
    """
    print("\n[Step 1/6] Generating targeted search queries...")

    SYSTEM = """You are a research strategist. Given a news topic, generate exactly 4 focused
web search queries that together cover the topic from four complementary angles:
  1. Recent news and current developments
  2. Background, history, and context
  3. Expert analysis or opinion
  4. Future outlook or implications

Respond ONLY with a valid JSON object — no explanation, no markdown:
{
  "queries": ["query 1", "query 2", "query 3", "query 4"],
  "angles":  ["recent news", "background", "expert opinion", "future implications"]
}"""

    USER = f'Topic: "{state["topic"]}"'

    raw    = call_groq(SYSTEM, USER)
    parsed = parse_json_response(raw)

    state["queries"] = parsed["queries"]
    state["angles"]  = parsed["angles"]

    print(f"  ✓ {len(state['queries'])} queries generated:")
    for angle, q in zip(state["angles"], state["queries"]):
        print(f"     [{angle}] {q}")

    return state


# ── Step 2 — Web Search (Tool Call) ───────────────────────────────────────────

def step2_web_search(state: dict) -> dict:
    """
    TOOL CALL: Execute web searches for each query via the Serper API.

    Why a tool, not an LLM?
    LLMs cannot retrieve real-time information; they hallucinate recent facts.
    Serper returns actual, dated web results — grounding the rest of the chain
    in real sources rather than model-generated fiction.

    Input  : state["queries"]
    Output : state["raw_results"]   — dict: query → list of article dicts
             state["all_articles"]  — deduplicated flat list of all articles
    """
    print("\n[Step 2/6] Searching the web (tool call — Serper API)...")

    raw_results = {}
    for query in state["queries"]:
        try:
            results = serper_search(query, num_results=4)
            raw_results[query] = results
            print(f"  ✓ {len(results)} results for: '{query[:55]}...'")
        except Exception as exc:
            # Graceful degradation: log the failure, continue the chain
            print(f"  ⚠ Search failed for '{query[:55]}...': {exc}")
            raw_results[query] = []

    state["raw_results"] = raw_results

    # Flatten + deduplicate by URL
    seen, unique = set(), []
    for results in raw_results.values():
        for article in results:
            if article["source"] not in seen:
                seen.add(article["source"])
                unique.append(article)

    state["all_articles"] = unique
    print(f"  ✓ {len(unique)} unique articles collected across all queries")

    return state


# ── Step 3 — Article Summarizer ───────────────────────────────────────────────

def step3_summarize_articles(state: dict) -> dict:
    """
    LLM Call 2: Convert raw article snippets into structured per-article summaries.

    Why a separate step?
    The raw Serper output is unstructured text snippets of inconsistent length
    and relevance. This step normalises every article into the same schema so
    that Step 4 can reason over a uniform, comparable dataset rather than
    messy free-form strings.

    Input  : state["all_articles"], state["topic"]
    Output : state["summaries"] — list of structured summary dicts
    """
    print("\n[Step 3/6] Summarising articles...")

    # Build a numbered article block for the prompt
    articles_block = ""
    for i, art in enumerate(state["all_articles"], 1):
        articles_block += (
            f"\n[Article {i}]\n"
            f"Title   : {art['title']}\n"
            f"Source  : {art['source']}\n"
            f"Date    : {art['date']}\n"
            f"Snippet : {art['snippet']}\n"
        )

    SYSTEM = """You are a news analyst. For each article, extract a structured summary.

Respond ONLY with a valid JSON array — no explanation, no markdown:
[
  {
    "article_id"  : 1,
    "headline"    : "one-sentence summary of what the article says",
    "key_facts"   : ["concrete fact 1", "concrete fact 2", "concrete fact 3"],
    "sentiment"   : "positive | negative | neutral | mixed",
    "source_name" : "domain name only (e.g. bbc.com)"
  }
]
Include one object per article. Keep key_facts as specific claims, not opinions."""

    USER = f'Topic: "{state["topic"]}"\n\nArticles:\n{articles_block}'

    raw = call_groq(SYSTEM, USER)

    try:
        summaries = parse_json_response(raw)
    except (json.JSONDecodeError, KeyError):
        # Fallback: build minimal summaries so the chain does not break
        print("  ⚠ JSON parse failed — using fallback summaries")
        summaries = [
            {
                "article_id": i + 1,
                "headline":   art["title"],
                "key_facts":  [art["snippet"]],
                "sentiment":  "neutral",
                "source_name": art["source"],
            }
            for i, art in enumerate(state["all_articles"])
        ]

    state["summaries"] = summaries
    print(f"  ✓ {len(summaries)} articles summarised")

    return state


# ── Step 4 — Theme Extractor ──────────────────────────────────────────────────

def step4_extract_themes(state: dict) -> dict:
    """
    LLM Call 3: Identify cross-cutting themes, contradictions, and consensus.

    Why a separate step?
    Step 3 produced one summary per article — a flat list.
    This step reasons *across* all summaries to find patterns invisible
    in any single article: recurring themes, conflicting narratives,
    and what all sources agree on. This higher-order analysis is what
    a briefing writer needs before drafting.

    Input  : state["summaries"], state["topic"]
    Output : state["themes"] — theme analysis dict
    """
    print("\n[Step 4/6] Extracting themes and patterns...")

    SYSTEM = """You are a senior intelligence analyst. Given structured article summaries,
identify the major cross-cutting themes in the coverage.

Respond ONLY with a valid JSON object — no explanation, no markdown:
{
  "themes": [
    {
      "name"               : "short theme label",
      "description"        : "2–3 sentence explanation of this theme",
      "supporting_articles": [1, 2],
      "significance"       : "why this matters beyond the immediate news"
    }
  ],
  "contradictions"     : ["description of any conflicting claims between sources"],
  "consensus"          : "what all or most sources agree on (1–2 sentences)",
  "overall_sentiment"  : "positive | negative | neutral | mixed",
  "key_entities"       : ["most-mentioned people, organisations, or places"]
}
Identify 3–5 themes. Be specific — name facts, not vague generalities."""

    USER = (
        f'Topic: "{state["topic"]}"\n\n'
        f"Summaries:\n{json.dumps(state['summaries'], indent=2)}"
    )

    raw    = call_groq(SYSTEM, USER)
    parsed = parse_json_response(raw)

    state["themes"] = parsed
    print(f"  ✓ {len(parsed.get('themes', []))} themes identified:")
    for t in parsed.get("themes", []):
        print(f"     • {t['name']}")

    return state


# ── Step 5 — Briefing Writer ──────────────────────────────────────────────────

def step5_write_briefing(state: dict) -> dict:
    """
    LLM Call 4: Draft a structured Markdown briefing from the theme analysis.

    Why a separate step?
    Synthesis (Step 4) and writing are distinct cognitive tasks.
    Asking one LLM call to both find themes AND write a polished document
    produces worse output than letting a dedicated writing step receive
    already-structured theme data and focus solely on prose quality.

    Input  : state["themes"], state["summaries"], state["topic"]
    Output : state["draft_briefing"] — full Markdown string
    """
    print("\n[Step 5/6] Writing briefing draft...")

    today   = datetime.now().strftime("%B %d, %Y")
    context = {
        "topic":         state["topic"],
        "date":          today,
        "article_count": len(state["summaries"]),
        "themes":        state["themes"],
        "summaries":     state["summaries"],
    }

    SYSTEM = f"""You are an intelligence briefing writer. Produce a professional news briefing
in the exact Markdown structure below. Use a neutral, precise analytical tone.
Cite specific facts from the summaries — avoid generalities.

Required structure (use exactly these headings):
# [TOPIC] — News Briefing
**Date:** {today}
**Articles Reviewed:** [N]
**Overall Sentiment:** [sentiment from themes analysis]

## Executive Summary
[2–3 paragraphs: what is happening, why it matters, what to watch]

## Key Themes
### [Theme 1 name]
[2–3 paragraphs with specific facts]
### [Theme 2 name]
...

## Points of Tension
[Paragraph on contradictions or gaps between sources]

## Key Entities to Watch
- [bullet list]

## Sources Referenced
- [bullet list of source_name from summaries]
"""

    USER = f"Research data:\n{json.dumps(context, indent=2)}"

    state["draft_briefing"] = call_groq(SYSTEM, USER, temperature=0.4)
    print("  ✓ Draft briefing written")

    return state


# ── Step 6 — Critic + Refiner ─────────────────────────────────────────────────

def step6_critique_and_refine(state: dict) -> dict:
    """
    LLM Calls 5 & 6: Critique the draft, then produce a refined final version.

    Why a separate step?
    The draft from Step 5 was written to cover the given data.
    A dedicated critic can identify gaps, unsupported claims, and biased
    framing that the writer step could not self-detect. The refiner then
    addresses those specific weaknesses — producing a qualitatively better
    output than any single draft-and-publish step could.

    Input  : state["draft_briefing"], state["topic"]
    Output : state["critique"]       — structured critique dict
             state["final_briefing"] — polished final Markdown string
    """
    print("\n[Step 6/6] Critiquing and refining the briefing...")

    # ── 6a: Editorial critique ─────────────────────────────────────────────
    CRITIC_SYSTEM = """You are an editorial critic reviewing a news briefing.
Identify specific, actionable weaknesses only.

Respond ONLY with a valid JSON object — no markdown:
{
  "weaknesses"          : ["specific flaw 1", "specific flaw 2"],
  "missing_angles"      : ["angle not covered by the current draft"],
  "bias_flags"          : ["phrase or framing that appears biased or unbalanced"],
  "suggested_additions" : ["concrete thing that should be added or clarified"]
}"""

    critic_raw = call_groq(
        CRITIC_SYSTEM,
        f'Topic: "{state["topic"]}"\n\nBriefing draft:\n{state["draft_briefing"]}',
    )
    try:
        critique = parse_json_response(critic_raw)
    except (json.JSONDecodeError, KeyError):
        critique = {"weaknesses": [], "missing_angles": [],
                    "bias_flags": [], "suggested_additions": []}

    state["critique"] = critique
    print(f"  ✓ Critique: {len(critique.get('weaknesses', []))} weaknesses, "
          f"{len(critique.get('missing_angles', []))} missing angles")

    # ── 6b: Refined final briefing ─────────────────────────────────────────
    REFINER_SYSTEM = """You are a senior editor. Revise the news briefing using the critique below.
Keep the same Markdown structure. Address each weakness and missing angle you can
given the available data. Append this section at the end:

## Editorial Notes
[2–3 sentences: what was changed, what limitations remain]

Return the complete, revised briefing in Markdown — nothing else."""

    state["final_briefing"] = call_groq(
        REFINER_SYSTEM,
        f"Original draft:\n{state['draft_briefing']}\n\n"
        f"Editorial critique:\n{json.dumps(critique, indent=2)}",
        temperature=0.3,
    )
    print("  ✓ Final briefing refined and ready")

    return state
