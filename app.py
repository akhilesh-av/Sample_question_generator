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

# ==============================
# Base Configuration
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# ==============================
# Ensure Required Directories
# ==============================

def ensure_directories():
    for directory in [DATA_DIR, LOGS_DIR]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory ready: {directory}")
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}")
            raise

ensure_directories()

# ==============================
# Logging Configuration
# ==============================

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

# ==============================
# Flask App Setup
# ==============================

app = Flask(__name__, template_folder=str(BASE_DIR))
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# ==============================
# Groq Configuration
# ==============================

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
- Graduate / competitive exam standard only.
- Test analytical ability and application.
- Avoid trivial recall.
- Professional wording.

Output ONLY valid JSON, no extra text.
Structure:
{{"questions": [{{"number": 1, "question": "full question text", "answer": "correct answer"}}, ...]}}

Return only the JSON object."""


# ==============================
# Response Parsing
# ==============================

def parse_questions_response(text: str) -> list[dict]:
    text = text.strip()

    # Remove markdown block if exists
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

    # Fallback parsing
    result = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines[:10], 1):
        result.append({"number": i, "question": line, "answer": ""})

    return result


# ==============================
# Generate Questions
# ==============================

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

    if len(questions) != 10:
        logger.warning("Expected 10 questions but got %s", len(questions))

    for i, q in enumerate(questions, 1):
        q.setdefault("number", i)
        q.setdefault("question", "")
        q.setdefault("answer", "")

    return questions[:10]


# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    return render_template("index.html", topics=TOPICS, difficulties=DIFFICULTIES)


@app.route("/generate", methods=["POST"])
def generate():
    topic = request.form.get("topic", "").strip()
    difficulty = request.form.get("difficulty", "").strip()

    if topic not in TOPICS:
        flash("Please select a valid topic.", "error")
        return redirect(url_for("index"))

    if difficulty not in DIFFICULTIES:
        flash("Please select a valid difficulty.", "error")
        return redirect(url_for("index"))

    try:
        questions = generate_questions(topic, difficulty)
    except Exception as e:
        logger.exception("Generation failed: %s", e)
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


# ==============================
# Run Application
# ==============================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
