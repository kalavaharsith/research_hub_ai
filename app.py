"""
ResearchHub AI Agent - Python Flask Backend
All routes powered by Anthropic Claude API
"""

import os
import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

app = Flask(__name__, static_folder='static')

# ─────────────────────────────────────────────
# Anthropic API helper (pure stdlib – no SDK)
# ─────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def get_api_key(request_data: dict = None) -> str:
    """Get API key from request body first, then env."""
    if request_data and request_data.get("api_key"):
        return request_data["api_key"]
    return os.environ.get("ANTHROPIC_API_KEY", "")


def claude(system: str, user: str, max_tokens: int = 1500, api_key: str = "") -> str:
    """Call Claude API and return text response."""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}]
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"API error {e.code}: {body}")


def claude_chat(messages: list, system: str, max_tokens: int = 1500, api_key: str = "") -> str:
    """Multi-turn chat call."""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"API error {e.code}: {body}")


# ─────────────────────────────────────────────
# CORS helper
# ─────────────────────────────────────────────
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.after_request
def add_cors(resp):
    return cors(resp)


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:p>", methods=["OPTIONS"])
def options(p=""):
    return cors(jsonify({"ok": True}))


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─────────────────────────────────────────────
# 1. /api/search  – intelligent research query
# ─────────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    system = (
        "You are an expert AI research assistant with broad academic knowledge. "
        "When given a research query, provide a comprehensive, well-structured response that includes:\n"
        "1. A brief direct answer\n"
        "2. Key concepts and background\n"
        "3. Recent developments or findings\n"
        "4. Research implications\n"
        "5. Suggested further reading directions\n"
        "Use clear headings, be thorough but accessible, and cite general knowledge areas."
    )
    try:
        key = get_api_key(data)
        result = claude(system, query, 1500, key)
        return jsonify({"result": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 2. /api/chat  – multi-turn research chat
# ─────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "Messages are required"}), 400

    system = (
        "You are an expert AI research assistant with deep expertise across science, technology, humanities, "
        "and social sciences. Engage in thoughtful, multi-turn research conversations. "
        "Help users explore topics deeply, ask clarifying questions when useful, "
        "suggest related research angles, and explain complex concepts clearly. "
        "Reference methodologies, key researchers, and landmark studies where relevant."
    )
    try:
        key = get_api_key(data)
        result = claude_chat(messages, system, 1500, key)
        return jsonify({"result": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 3. /api/summarize  – smart text summarization
# ─────────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "brief")
    length = data.get("length", "medium")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    mode_instructions = {
        "brief": "Provide a concise, clear summary capturing the essential points.",
        "detailed": "Provide a detailed, comprehensive summary preserving all key arguments, evidence, and conclusions.",
        "bullets": "Provide a well-structured bullet-point summary with clear categorical sections and sub-points.",
        "academic": "Provide an academic-style abstract-like summary covering: Background, Objective, Methods (if applicable), Key Findings, and Implications.",
        "eli5": "Explain this text as if teaching a curious 12-year-old, using simple language, relatable analogies, and avoiding jargon."
    }
    length_instructions = {
        "short": "Keep the summary to 1–2 paragraphs (under 150 words).",
        "medium": "Aim for 3–4 paragraphs (150–300 words).",
        "long": "Provide a thorough summary of 4–6 paragraphs (300–500 words)."
    }

    system = (
        f"You are an expert research summarizer. {mode_instructions.get(mode, mode_instructions['brief'])} "
        f"{length_instructions.get(length, length_instructions['medium'])} "
        "Be accurate, preserve nuance, and ensure the summary stands alone without needing the original."
    )
    try:
        key = get_api_key(data)
        result = claude(system, f"Summarize the following:\n\n{text}", 1000, key)
        return jsonify({"result": result, "mode": mode, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 4. /api/insights  – deep research insights
# ─────────────────────────────────────────────
@app.route("/api/insights", methods=["POST"])
def insights():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    insight_type = data.get("type", "full")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    type_prompts = {
        "gaps": "Identify and analyze RESEARCH GAPS: What questions remain unanswered? What areas need more study? What methodological gaps exist? What populations or contexts are under-studied?",
        "findings": "Extract and analyze KEY FINDINGS: What are the landmark discoveries? What consensus has emerged? What are the most cited or impactful results?",
        "trends": "Analyze EMERGING TRENDS: What directions is this field moving? What new methodologies are gaining traction? What sub-fields are growing?",
        "methods": "Analyze RESEARCH METHODOLOGIES: What approaches are used? What are their strengths and limitations? What innovative methods are emerging?",
        "full": "Provide a COMPREHENSIVE RESEARCH ANALYSIS covering: (1) Overview & History, (2) Key Findings, (3) Current Trends, (4) Research Gaps, (5) Methodological Landscape, (6) Future Directions, (7) Key Researchers/Institutions."
    }

    system = (
        "You are a senior research analyst with expertise in synthesizing academic literature across disciplines. "
        + type_prompts.get(insight_type, type_prompts["full"])
        + " Be specific, evidence-based, and provide actionable research directions. "
        "Use clear numbered sections and provide concrete examples."
    )

    try:
        key = get_api_key(data)
        result = claude(system, f"Analyze this research topic:\n\n{topic}", 1500, key)

        # Extract simple trending topics from result
        topics = extract_topics(topic, result)

        return jsonify({
            "result": result,
            "trending": topics,
            "metrics": {
                "coverage": min(95, 40 + len(result) // 50),
                "depth": min(90, 35 + len(result) // 60),
                "relevance": 94
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_topics(main_topic, text):
    """Extract subtopics from insight text."""
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    seen = set()
    topics = []
    for w in words:
        if w.lower() not in seen and len(w) > 5 and w != main_topic:
            seen.add(w.lower())
            topics.append(w)
        if len(topics) >= 6:
            break
    if not topics:
        topics = [main_topic, "Methodology", "Key Findings", "Future Research", "Applications", "Limitations"]
    return topics[:6]


# ─────────────────────────────────────────────
# 5. /api/keywords  – keyword extraction
# ─────────────────────────────────────────────
@app.route("/api/keywords", methods=["POST"])
def keywords():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    kw_type = data.get("type", "keywords")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    type_prompts = {
        "keywords": (
            "Extract the most important KEYWORDS from this text. "
            "For each keyword provide: Category | Term | Importance (1-10) | Brief context.\n"
            "Group them into: Core Concepts, Methods/Approaches, Domain Terms, Action Terms."
        ),
        "themes": (
            "Identify the main THEMES in this text. "
            "For each theme: Theme Name | Supporting evidence | Significance. "
            "Organize from primary to secondary themes."
        ),
        "entities": (
            "Extract NAMED ENTITIES: People, Organizations, Locations, Dates, Events, Products. "
            "Format: Type | Name | Context/Role | Frequency of mention."
        ),
        "all": (
            "Perform COMPLETE TEXT ANALYSIS extracting:\n"
            "1. Top Keywords (with importance scores)\n"
            "2. Main Themes\n"
            "3. Named Entities (people, orgs, places)\n"
            "4. Key Concepts\n"
            "5. Sentiment/Tone indicators\n"
            "Present in clear organized sections."
        )
    }

    system = (
        "You are an expert computational linguist and text analysis specialist. "
        + type_prompts.get(kw_type, type_prompts["keywords"])
        + " Be thorough, precise, and ensure extracted items are truly significant to the text."
    )

    try:
        key = get_api_key(data)
        result = claude(system, f"Analyze this text:\n\n{text}", 1000, key)

        # Build keyword cloud list
        cloud_words = build_keyword_cloud(text, result)

        return jsonify({
            "result": result,
            "cloud": cloud_words,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def build_keyword_cloud(text, analysis):
    """Build keyword frequency cloud."""
    stop = {"the","and","for","that","this","with","from","are","was","were","been",
            "have","has","had","will","would","could","should","may","might","must",
            "shall","can","not","but","what","when","where","which","who","how",
            "its","our","their","your","all","any","each","more","also","into",
            "about","over","such","then","than","they","them","these","those","being"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', (text + " " + analysis).lower())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:25]
    return [{"word": w, "count": c} for w, c in sorted_words]


# ─────────────────────────────────────────────
# 6. /api/citations  – citation helper
# ─────────────────────────────────────────────
@app.route("/api/citations", methods=["POST"])
def citations():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "format")
    style = data.get("style", "APA")
    if not text:
        return jsonify({"error": "Citation text is required"}), 400

    mode_prompts = {
        "format": (
            f"FORMAT this citation in {style} style. "
            "Provide: (1) The correctly formatted citation, "
            "(2) A breakdown of each element, "
            "(3) Any corrections made and why, "
            "(4) Common mistakes to avoid with this type of citation."
        ),
        "suggest": (
            f"SUGGEST 5 relevant academic sources for this research topic, formatted in {style} style. "
            "For each source include: Full citation, Why it's relevant, Key contribution, Access note. "
            "Focus on seminal works and recent high-impact studies."
        ),
        "validate": (
            f"VALIDATE these citations against {style} style guidelines. "
            "For each citation: (1) Is it valid? (2) What errors exist? "
            "(3) Provide the corrected version. "
            "(4) Explain each correction. "
            "Be specific about punctuation, ordering, and formatting rules."
        )
    }

    system = (
        f"You are an expert academic librarian and citation specialist with mastery of all major citation styles. "
        + mode_prompts.get(mode, mode_prompts["format"])
        + " Be precise, educational, and provide complete, copy-ready citations."
    )

    try:
        key = get_api_key(data)
        result = claude(system, text, 1200, key)
        return jsonify({
            "result": result,
            "style": style,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 7. /api/compare  – compare & contrast
# ─────────────────────────────────────────────
@app.route("/api/compare", methods=["POST"])
def compare():
    data = request.get_json(force=True)
    concept_a = (data.get("concept_a") or "").strip()
    concept_b = (data.get("concept_b") or "").strip()
    mode = data.get("mode", "similarities")
    if not concept_a or not concept_b:
        return jsonify({"error": "Both concepts are required"}), 400

    mode_prompts = {
        "similarities": (
            "Analyze SIMILARITIES AND DIFFERENCES using a structured approach:\n"
            "1. Key Similarities (with explanations)\n"
            "2. Key Differences (with explanations)\n"
            "3. Overlapping Applications\n"
            "4. When to choose each"
        ),
        "pros-cons": (
            "Compare PROS AND CONS:\n"
            f"For {concept_a}: Advantages, Disadvantages, Ideal Use Cases\n"
            f"For {concept_b}: Advantages, Disadvantages, Ideal Use Cases\n"
            "Then: Overall recommendation based on context"
        ),
        "academic": (
            "Provide a rigorous ACADEMIC COMPARISON covering:\n"
            "1. Theoretical Foundations\n"
            "2. Historical Development\n"
            "3. Empirical Evidence\n"
            "4. Scholarly Debates\n"
            "5. Synthesis and Reconciliation"
        ),
        "table": (
            "Create a STRUCTURED COMPARISON TABLE with these dimensions:\n"
            "Definition, Origin/History, Core Principles, Methodology, Applications, "
            "Strengths, Limitations, Key Proponents, Current Status, Future Outlook.\n"
            "Format clearly with | separators for each row."
        ),
        "critical": (
            "Provide a CRITICAL ANALYSIS:\n"
            "1. Underlying assumptions of each\n"
            "2. Strengths and weaknesses of each argument\n"
            "3. Points of contention in the literature\n"
            "4. Synthesis: How can they complement each other?\n"
            "5. Your analytical verdict"
        )
    }

    system = (
        "You are an expert research analyst skilled in rigorous comparative analysis. "
        + mode_prompts.get(mode, mode_prompts["similarities"])
        + " Be balanced, thorough, and provide concrete examples. "
        "Support claims with reasoning and academic context."
    )

    try:
        key = get_api_key(data)
        result = claude(system, f"Compare and contrast:\n\nConcept A: {concept_a}\n\nConcept B: {concept_b}", 1500, key)
        return jsonify({
            "result": result,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 8. /api/history  – get search history
# ─────────────────────────────────────────────
# History is stored client-side; this endpoint is for any server-side ops
@app.route("/api/health", methods=["GET"])
def health():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "api_configured": bool(api_key and len(api_key) > 10),
        "timestamp": datetime.now().isoformat()
    })


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 ResearchHub AI Backend running on http://localhost:{port}")
    print("   Set ANTHROPIC_API_KEY env var before starting\n")
    app.run(host="0.0.0.0", port=port, debug=False)
