"""
Lost & Found Smart Portal — Flask application with SQLite, uploads, and scored matching.

Future extensions (structure only — not implemented here):
  - Azure SQL: replace get_db() with pyodbc/SQLAlchemy; keep column names.
  - Auth: add users table + Flask-Login; protect report routes.
  - Email: call send_mail() after INSERT when NOTIFY_ON_MATCH is True.
  - Maps: add latitude/longitude columns and pass to a map template partial.
"""

import os
import sqlite3
import uuid
from datetime import date
from typing import Any, Optional

from flask import Flask, g, render_template, request
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["DATABASE"] = os.path.join(app.instance_path, "lost_found.db")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload

# Images saved under static/uploads/ so url_for('static', filename='uploads/…') works.
UPLOAD_SUBDIR = "uploads"
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", UPLOAD_SUBDIR)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# Match score: Strong vs Possible (0–100 scale).
STRONG_SCORE_MIN = 70
POSSIBLE_SCORE_MIN = 45


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_upload_folder() -> None:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def save_upload(file_storage) -> Optional[str]:
    """
    Save an uploaded image with a unique prefix; return stored filename or None.
    Uses werkzeug.secure_filename for safe names.
    """
    if not file_storage or not file_storage.filename:
        return None
    if not allowed_file(file_storage.filename):
        return None
    ensure_upload_folder()
    base = secure_filename(file_storage.filename)
    if not base:
        return None
    unique = f"{uuid.uuid4().hex[:12]}_{base}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    file_storage.save(path)
    return unique


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    """Create tables with full schema (new installs)."""
    os.makedirs(app.instance_path, exist_ok=True)
    ensure_upload_folder()
    db = sqlite3.connect(app.config["DATABASE"])
    try:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS lost_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT,
                image TEXT,
                contact_name TEXT,
                contact_phone TEXT,
                contact_email TEXT
            );

            CREATE TABLE IF NOT EXISTS found_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT,
                image TEXT,
                contact_name TEXT,
                contact_phone TEXT,
                contact_email TEXT
            );
            """
        )
        db.commit()
    finally:
        db.close()


def migrate_legacy_columns() -> None:
    """
    Add new columns when upgrading an older database (SQLite ALTER TABLE).
    """
    db = sqlite3.connect(app.config["DATABASE"])
    try:
        for table in ("lost_items", "found_items"):
            existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if "image" not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN image TEXT")
            if "contact_name" not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN contact_name TEXT")
            if "contact_phone" not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN contact_phone TEXT")
            if "contact_email" not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN contact_email TEXT")
        db.commit()
    finally:
        db.close()


init_db()
migrate_legacy_columns()


# -----------------------------------------------------------------------------
# Matching: normalize, partial text, date proximity, score 0–100
# -----------------------------------------------------------------------------

MIN_PARTIAL_LEN = 3


def normalize_text(value: str) -> str:
    if not value:
        return ""
    return " ".join(value.lower().strip().split())


def texts_match_flexible(a: str, b: str) -> bool:
    """Case-insensitive exact or substring match (for gating pairs)."""
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    if len(na) < MIN_PARTIAL_LEN or len(nb) < MIN_PARTIAL_LEN:
        return na == nb
    return na in nb or nb in na


def text_similarity_points(a: str, b: str, max_points: float) -> float:
    """
    Score one text field up to max_points (40 for name, 40 for location).
    Exact normalized match = full points; partial (substring) = reduced.
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return max_points if na == nb else 0.0
    if na == nb:
        return max_points
    if len(na) < MIN_PARTIAL_LEN or len(nb) < MIN_PARTIAL_LEN:
        return max_points if na == nb else 0.0
    if na in nb or nb in na:
        return max_points * 0.72
    return 0.0


def days_apart(date_str_lost: str, date_str_found: str) -> Optional[int]:
    try:
        dl = date.fromisoformat(str(date_str_lost).strip()[:10])
        df = date.fromisoformat(str(date_str_found).strip()[:10])
        return abs((dl - df).days)
    except ValueError:
        return None


def date_similarity_points(d1: str, d2: str) -> float:
    """Up to 20 points: same day best, then decay by gap."""
    gap = days_apart(d1, d2)
    if gap is None:
        return 0.0
    if gap == 0:
        return 20.0
    if gap <= 3:
        return 16.0
    if gap <= 7:
        return 10.0
    if gap <= 14:
        return 5.0
    return 0.0


def compute_match_score(
    lost_name: str,
    lost_loc: str,
    lost_date: str,
    found_name: str,
    found_loc: str,
    found_date: str,
) -> Optional[dict[str, Any]]:
    """
    Return None if not a candidate pair.
    Otherwise dict with score (0–100), match_type, days_apart, and sub-scores.
    """
    if not texts_match_flexible(lost_name, found_name):
        return None
    if not texts_match_flexible(lost_loc, found_loc):
        return None

    name_pts = text_similarity_points(lost_name, found_name, 40.0)
    loc_pts = text_similarity_points(lost_loc, found_loc, 40.0)
    date_pts = date_similarity_points(lost_date, found_date)
    total = int(round(min(100.0, name_pts + loc_pts + date_pts)))

    if total < POSSIBLE_SCORE_MIN:
        return None

    match_type = "Strong Match" if total >= STRONG_SCORE_MIN else "Possible Match"
    gap = days_apart(lost_date, found_date)

    return {
        "score": total,
        "match_type": match_type,
        "days_apart": gap,
        "name_points": round(name_pts, 1),
        "location_points": round(loc_pts, 1),
        "date_points": round(date_pts, 1),
    }


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "location": row["location"],
        "date": row["date"],
        "description": row["description"] or "",
        "image": row["image"] if row["image"] else None,
        "contact_name": row["contact_name"] or "",
        "contact_phone": row["contact_phone"] or "",
        "contact_email": row["contact_email"] or "",
    }


def find_matches() -> list[dict[str, Any]]:
    db = get_db()
    lost_rows = db.execute("SELECT * FROM lost_items ORDER BY id").fetchall()
    found_rows = db.execute("SELECT * FROM found_items ORDER BY id").fetchall()

    pairs: list[dict[str, Any]] = []
    for lr in lost_rows:
        for fr in found_rows:
            result = compute_match_score(
                lr["name"],
                lr["location"],
                lr["date"],
                fr["name"],
                fr["location"],
                fr["date"],
            )
            if result is None:
                continue
            pairs.append(
                {
                    "lost": row_to_dict(lr),
                    "found": row_to_dict(fr),
                    "score": result["score"],
                    "match_type": result["match_type"],
                    "days_apart": result["days_apart"],
                }
            )

    # Best matches first: higher score, then closer dates, stable ids.
    pairs.sort(
        key=lambda p: (
            -p["score"],
            p["days_apart"] if p["days_apart"] is not None else 999,
            p["lost"]["id"],
            p["found"]["id"],
        )
    )
    return pairs


# -----------------------------------------------------------------------------
# Form context helper (repopulate on validation error)
# -----------------------------------------------------------------------------


def form_defaults(
    name: str = "",
    location: str = "",
    date_str: str = "",
    description: str = "",
    contact_name: str = "",
    contact_phone: str = "",
    contact_email: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "location": location,
        "date": date_str,
        "description": description,
        "contact_name": contact_name,
        "contact_phone": contact_phone,
        "contact_email": contact_email,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/report/lost", methods=["GET", "POST"])
def report_lost() -> Any:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip()
        date_str = request.form.get("date", "").strip()
        description = request.form.get("description", "").strip()
        contact_name = request.form.get("contact_name", "").strip()
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip()

        ctx = form_defaults(
            name,
            location,
            date_str,
            description,
            contact_name,
            contact_phone,
            contact_email,
        )

        if not name or not location or not date_str:
            return render_template(
                "report_lost.html",
                message="Please fill in item name, location, and date.",
                **ctx,
            ), 400

        image_name = save_upload(request.files.get("image"))

        db = get_db()
        db.execute(
            """
            INSERT INTO lost_items (
                name, location, date, description, image,
                contact_name, contact_phone, contact_email
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                location,
                date_str,
                description or None,
                image_name,
                contact_name or None,
                contact_phone or None,
                contact_email or None,
            ),
        )
        db.commit()
        return render_template(
            "report_lost.html",
            message="Lost item saved successfully.",
        )

    return render_template("report_lost.html")


@app.route("/report/found", methods=["GET", "POST"])
def report_found() -> Any:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip()
        date_str = request.form.get("date", "").strip()
        description = request.form.get("description", "").strip()
        contact_name = request.form.get("contact_name", "").strip()
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip()

        ctx = form_defaults(
            name,
            location,
            date_str,
            description,
            contact_name,
            contact_phone,
            contact_email,
        )

        if not name or not location or not date_str:
            return render_template(
                "report_found.html",
                message="Please fill in item name, location, and date.",
                **ctx,
            ), 400

        image_name = save_upload(request.files.get("image"))

        db = get_db()
        db.execute(
            """
            INSERT INTO found_items (
                name, location, date, description, image,
                contact_name, contact_phone, contact_email
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                location,
                date_str,
                description or None,
                image_name,
                contact_name or None,
                contact_phone or None,
                contact_email or None,
            ),
        )
        db.commit()
        return render_template(
            "report_found.html",
            message="Found item saved successfully.",
        )

    return render_template("report_found.html")


@app.route("/matches")
def view_matches() -> str:
    matches = find_matches()
    return render_template("results.html", matches=matches)


application = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
