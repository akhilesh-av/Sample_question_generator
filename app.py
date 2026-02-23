"""
Flask question generator: topic + difficulty selection, Groq API, JSON storage, logging.
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from groq import Groq

load_dotenv()

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent))
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Logging
LOG_FILE = LOGS_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL", "llama-3.1-8b-instant")

TOPICS = ["Maths", "Reasoning", "GK", "English"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]


def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    return Groq(api_key=GROQ_API_KEY)


def build_prompt(topic: str, difficulty: str) -> str:
    return f"""You are an expert question-setter for graduate-level public service exams (e.g. UPSC, state PSC, SSC). Generate exactly 10 questions on the topic "{topic}" at "{difficulty}" difficulty.

Quality requirements:
- Write at graduate / competitive exam standard. Do NOT write simple, basic, or school-level questions.
- Questions must test understanding, application, and analytical ability—suitable for civil services and similar exams.
- Avoid trivial recall; include scenario-based, conceptual, or multi-step items where appropriate.
- Keep wording precise and professional.

Output ONLY valid JSON, no other text. Use this exact structure:
{{"questions": [{{"number": 1, "question": "full question text", "answer": "correct answer"}}, ... for all 10 items]}}

Return only the JSON object."""


def parse_questions_response(text: str) -> list[dict]:
    """Parse Groq response into list of {number, question, answer}."""
    # Try raw JSON first
    text = text.strip()
    # Remove markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: Q1/A1 style
    result = []
    q_pattern = re.compile(r"(?:Q(\d+)|Question\s*\d*[:.]?)\s*(.+?)(?=Q\d+|Question\s*\d|A\d+|Answer\s*\d|$)", re.S | re.I)
    a_pattern = re.compile(r"(?:A(\d+)|Answer\s*\d*[:.]?)\s*(.+?)(?=A\d+|Answer\s*\d|Q\d+|Question\s*\d|$)", re.S | re.I)
    for m in q_pattern.finditer(text):
        num, q = m.group(1), (m.group(2) or "").strip()
        if not q:
            continue
        result.append({"number": int(num) if num else len(result) + 1, "question": q, "answer": ""})
    for m in a_pattern.finditer(text):
        num, a = m.group(1), (m.group(2) or "").strip()
        if not num or not result:
            continue
        idx = int(num) - 1
        if 0 <= idx < len(result):
            result[idx]["answer"] = a
    if result:
        return result

    # Last resort: split by numbered lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines[:20], 1):
        if i <= 10:
            result.append({"number": i, "question": line, "answer": lines[i + 9] if i + 9 < len(lines) else ""})
    return result[:10]


def generate_questions(topic: str, difficulty: str) -> list[dict]:
    client = get_groq_client()
    prompt = build_prompt(topic, difficulty)
    logger.info("Generating questions: topic=%s, difficulty=%s", topic, difficulty)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    raw = completion.choices[0].message.content or ""
    logger.info("Groq response received, length=%s", len(raw))
    questions = parse_questions_response(raw)
    if len(questions) > 10:
        questions = questions[:10]
    elif len(questions) < 10:
        logger.warning("Parsed only %s questions, expected 10", len(questions))
    for i, q in enumerate(questions, 1):
        if "number" not in q:
            q["number"] = i
        if "question" not in q:
            q["question"] = str(q.get("q", q.get("text", "")))
        if "answer" not in q:
            q["answer"] = str(q.get("a", q.get("answer_key", "")))
    return questions


@app.route("/")
def index():
    return render_template("index.html", topics=TOPICS, difficulties=DIFFICULTIES)


@app.route("/generate", methods=["POST"])
def generate():
    topic = request.form.get("topic", "").strip()
    difficulty = request.form.get("difficulty", "").strip()
    if not topic or topic not in TOPICS:
        flash("Please select a valid topic.", "error")
        return redirect(url_for("index"))
    if not difficulty or difficulty not in DIFFICULTIES:
        flash("Please select a valid difficulty.", "error")
        return redirect(url_for("index"))

    try:
        questions = generate_questions(topic, difficulty)
    except Exception as e:
        logger.exception("Generate failed: %s", e)
        flash(f"Generation failed: {e}", "error")
        return redirect(url_for("index"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\-]", "_", topic)
    q_file = DATA_DIR / f"questions_{safe_topic}_{difficulty}_{ts}.json"
    a_file = DATA_DIR / f"answers_{safe_topic}_{difficulty}_{ts}.json"

    questions_data = [
        {"number": q["number"], "question": q["question"], "answer": q["answer"]}
        for q in questions
    ]
    answers_data = [
        {"number": q["number"], "answer": q["answer"]}
        for q in questions
    ]

    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(
            {"topic": topic, "difficulty": difficulty, "generated_at": ts, "questions": questions_data},
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(a_file, "w", encoding="utf-8") as f:
        json.dump(
            {"topic": topic, "difficulty": difficulty, "generated_at": ts, "answers": answers_data},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Saved questions to %s and answers to %s", q_file.name, a_file.name)

    return render_template(
        "questions.html",
        topic=topic,
        difficulty=difficulty,
        questions=questions,
        questions_file=q_file.name,
        answers_file=a_file.name,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
