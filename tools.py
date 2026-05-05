"""
agent.py — Orchestrator for the News Analyst Agent

Initialises shared state, runs all six steps in order, saves outputs.
Import run_agent() from here to use the agent programmatically,
or run this file directly from the command line.
"""

import sys
import json
import os
from datetime import datetime

from steps import (
    step1_generate_queries,
    step2_web_search,
    step3_summarize_articles,
    step4_extract_themes,
    step5_write_briefing,
    step6_critique_and_refine,
)


# ── State initialiser ─────────────────────────────────────────────────────────

def init_state(topic: str) -> dict:
    """
    Create the shared state dictionary that accumulates results across all steps.
    Every key is pre-declared as None so the full schema is visible upfront.
    """
    return {
        # ── User input ─────────────────────────────────────────────────────
        "topic":          topic,
        "timestamp":      datetime.now().isoformat(),

        # ── Step 1 outputs ─────────────────────────────────────────────────
        "queries":        None,   # list[str] — 4 search queries
        "angles":         None,   # list[str] — 4 angle labels

        # ── Step 2 outputs ─────────────────────────────────────────────────
        "raw_results":    None,   # dict: query → list[article_dict]
        "all_articles":   None,   # list[article_dict] — deduplicated flat list

        # ── Step 3 outputs ─────────────────────────────────────────────────
        "summaries":      None,   # list[summary_dict]

        # ── Step 4 outputs ─────────────────────────────────────────────────
        "themes":         None,   # dict: theme analysis

        # ── Step 5 outputs ─────────────────────────────────────────────────
        "draft_briefing": None,   # str: Markdown draft

        # ── Step 6 outputs ─────────────────────────────────────────────────
        "critique":       None,   # dict: editorial critique
        "final_briefing": None,   # str: polished final Markdown
    }


# ── Output writer ─────────────────────────────────────────────────────────────

def save_outputs(state: dict) -> tuple[str, str]:
    """
    Write the final briefing (Markdown) and full state (JSON) to disk.
    Returns the paths of both files.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = state["topic"].lower().replace(" ", "_")[:30]

    md_path   = f"briefing_{topic_slug}_{timestamp}.md"
    json_path = f"state_{topic_slug}_{timestamp}.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(state["final_briefing"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    return md_path, json_path


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_agent(topic: str) -> dict:
    """
    Run the full 6-step News Analyst pipeline for the given topic.

    Chain dependency map:
      step1 reads  : topic
      step2 reads  : queries         (from step1)
      step3 reads  : all_articles    (from step2)
      step4 reads  : summaries       (from step3)
      step5 reads  : themes          (from step4)
      step6 reads  : draft_briefing  (from step5)

    Returns the completed state dict.
    """
    print("\n" + "=" * 62)
    print("  NEWS ANALYST AGENT")
    print(f"  Topic    : {topic}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    state = init_state(topic)

    # ── Run the chain ──────────────────────────────────────────────────────
    state = step1_generate_queries(state)
    state = step2_web_search(state)

    # Guard: if no articles came back, abort gracefully
    if not state["all_articles"]:
        print("\n⚠  No articles retrieved. Possible causes:")
        print("   • Invalid or missing SERPER_API_KEY")
        print("   • Topic returned no results — try rephrasing")
        return state

    state = step3_summarize_articles(state)
    state = step4_extract_themes(state)
    state = step5_write_briefing(state)
    state = step6_critique_and_refine(state)

    # ── Save structured outputs ────────────────────────────────────────────
    md_path, json_path = save_outputs(state)

    print("\n" + "=" * 62)
    print("  COMPLETE")
    print(f"  Briefing (Markdown) : {md_path}")
    print(f"  Full state (JSON)   : {json_path}")
    print("=" * 62 + "\n")

    # Print briefing preview to terminal
    preview_lines = state["final_briefing"].splitlines()[:20]
    print("\n--- BRIEFING PREVIEW (first 20 lines) ---\n")
    print("\n".join(preview_lines))
    print("\n[...] (see full file for complete briefing)")

    return state


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Validate environment variables before starting
    missing = [k for k in ("GROQ_API_KEY", "SERPER_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Error: Missing environment variable(s): {', '.join(missing)}")
        print("See README.md for setup instructions.")
        sys.exit(1)

    if len(sys.argv) >= 2:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter a news topic to analyse: ").strip()

    if not topic:
        print("Error: Topic cannot be empty.")
        sys.exit(1)

    run_agent(topic)
