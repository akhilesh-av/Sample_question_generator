"""
Flask question generator: topic + difficulty selection, Groq API, JSON storage, logging.
Serverless-safe version (Vercel compatible)
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
# PATH CONFIGURATION (IMPORTANT FIX)
# ==============================

# Detect serverless environment
IS_SERVERLESS = os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")

if IS_SERVERLESS:
    BASE_DIR = Path("/tmp")  # writable in serverless
else:
    BASE_DIR = Path(__file__).resolve().parent  # local dev

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# ==============================
# ENSURE DIRECTORIES
# ==============================

def ensure_directories():
    for directory in [DATA_DIR, LOGS_DIR]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Directory creation failed: {directory} -> {e}")
            raise

ensure_directories()

# ==============================
# LOGGING
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
# FLASK APP
# ==============================

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent))
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# ==============================
# GROQ CONFIG
# ==============================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL", "llama-3.1-8b-instant")

TOPICS = ["Maths", "Reasoning", "GK", "English"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]


def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def build_prompt(topic: str, difficulty: str) -> str:
    return f """You are an expert question-setter for graduate-level public service exams.

Generate exactly 10 questions on "{topic}" at "{difficulty}" difficulty.

Requirements:
- Graduate level
- Analytical
- No basic questions
- Professional wording

Output ONLY valid JSON:
{{"questions":[{{"number":1,"question":"...","answer":"..."}}]}}
"""


# ==============================
# RESPONSE PARSER
# ==============================

def parse_questions_response(text: str) -> list[dict]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return [{"number": i + 1, "question": q, "answer": ""} for i, q in enumerate(lines[:10])]


# ==============================
# QUESTION GENERATOR
# ==============================

def generate_questions(topic: str, difficulty: str) -> list[dict]:
    client = get_groq_client()
    prompt = build_prompt(topic, difficulty)

    logger.info("Generating questions: %s | %s", topic, difficulty)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )

    raw = completion.choices[0].message.content or ""
    questions = parse_questions_response(raw)

    for i, q in enumerate(questions, 1):
        q.setdefault("number", i)
        q.setdefault("question", "")
        q.setdefault("answer", "")

    return questions[:10]


# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():
    return render_template("index.html", topics=TOPICS, difficulties=DIFFICULTIES)


@app.route("/generate", methods=["POST"])
def generate():
    topic = request.form.get("topic", "").strip()
    difficulty = request.form.get("difficulty", "").strip()

    if topic not in TOPICS:
        flash("Invalid topic", "error")
        return redirect(url_for("index"))

    if difficulty not in DIFFICULTIES:
        flash("Invalid difficulty", "error")
        return redirect(url_for("index"))

    try:
        questions = generate_questions(topic, difficulty)
    except Exception as e:
        logger.exception("Generation failed")
        flash(str(e), "error")
        return redirect(url_for("index"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\-]", "_", topic)

    q_file = DATA_DIR / f"questions_{safe_topic}_{difficulty}_{ts}.json"
    a_file = DATA_DIR / f"answers_{safe_topic}_{difficulty}_{ts}.json"

    with open(q_file, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)

    with open(a_file, "w", encoding="utf-8") as f:
        json.dump({"answers": [{"number": q["number"], "answer": q["answer"]} for q in questions]}, f, indent=2)

    logger.info("Saved files → %s , %s", q_file, a_file)

    return render_template(
        "questions.html",
        topic=topic,
        difficulty=difficulty,
        questions=questions,
        questions_file=q_file.name,
        answers_file=a_file.name,
    )


# ==============================
# ENTRYPOINT
# ==============================

if __name__ == "__main__":
    app.run(debug=True, port=5001)
