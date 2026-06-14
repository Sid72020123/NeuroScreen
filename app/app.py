import json
import os
from datetime import datetime
from io import BytesIO

import librosa
import numpy as np
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys

# Add the project root directory to the Python path to resolve the import error.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
VOICE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "voice_model.pkl")
ALLOWED_AUDIO_EXTENSIONS = {"wav"}
VOICE_FEATURE_COLUMNS = [
    "Jitter",
    "Shimmer",
    "Pitch_STD",
    "ZCR",
    "MFCC_1",
    "MFCC_2",
    "MFCC_3",
]
VOICE_ANALYSIS_THRESHOLDS = {
    "Jitter": 0.02,
    "Shimmer": 0.05,
    "Pitch_STD": 28.0,
}

os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_object(Config)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "SQLITE_URL",
    f"sqlite:///{os.path.join(INSTANCE_DIR, 'neuroscreen.db')}",
)
app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {"pool_pre_ping": True})

# Keep the Flask instance folder aligned with the on-disk layout in this repo.
app.instance_path = INSTANCE_DIR

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to continue."
login_manager.login_message_category = "warning"


@app.context_processor
def inject_year():
    """Inject the current year into all templates."""
    return {"year": datetime.utcnow().year}


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    sessions = db.relationship(
        "ScreeningSession",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )


class ScreeningSession(db.Model):
    __tablename__ = "screening_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    voice_score = db.Column(db.Float, nullable=True)
    spiral_score = db.Column(db.Float, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    voice_metrics = db.Column(db.Text, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def ensure_database_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        columns = {
            column["name"] for column in inspector.get_columns("screening_sessions")
        }
        if "voice_metrics" not in columns:
            with db.engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE screening_sessions ADD COLUMN voice_metrics TEXT")
                )


def wants_json_response():
    accept_header = request.headers.get("Accept", "")
    requested_with = request.headers.get("X-Requested-With", "")
    return "application/json" in accept_header or requested_with == "XMLHttpRequest"


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audio_extension_allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS
    )


def load_voice_model():
    import joblib

    if not os.path.exists(VOICE_MODEL_PATH):
        return None
    return joblib.load(VOICE_MODEL_PATH)


_VOICE_MODEL = None


def get_voice_model():
    global _VOICE_MODEL
    if _VOICE_MODEL is None:
        _VOICE_MODEL = load_voice_model()
    return _VOICE_MODEL


def extract_voice_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            return {
                "Jitter": 0.0,
                "Shimmer": 0.0,
                "Pitch_STD": 0.0,
                "ZCR": 0.0,
                "MFCC_1": 0.0,
                "MFCC_2": 0.0,
                "MFCC_3": 0.0,
            }

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=50,
            fmax=300,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        min_len = min(len(f0), len(rms), len(voiced_flag))
        f0 = f0[:min_len]
        rms = rms[:min_len]
        voiced_flag = voiced_flag[:min_len]

        f0_voiced = f0[voiced_flag]
        rms_voiced = rms[voiced_flag]

        if len(f0_voiced) > 1:
            periods = 1.0 / f0_voiced
            mean_period = np.mean(periods)
            jitter = (
                np.mean(np.abs(np.diff(periods))) / mean_period
                if mean_period > 0
                else 0.0
            )
            pitch_std = float(np.std(f0_voiced))
        else:
            jitter = 0.0
            pitch_std = 0.0

        if len(rms_voiced) > 1:
            mean_rms = np.mean(rms_voiced)
            shimmer = (
                np.mean(np.abs(np.diff(rms_voiced))) / mean_rms if mean_rms > 0 else 0.0
            )
        else:
            shimmer = 0.0

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        mfcc_means = np.mean(mfccs, axis=1)

        return {
            "Jitter": float(jitter),
            "Shimmer": float(shimmer),
            "Pitch_STD": float(pitch_std),
            "ZCR": float(zcr),
            "MFCC_1": float(mfcc_means[0]) if len(mfcc_means) > 0 else 0.0,
            "MFCC_2": float(mfcc_means[1]) if len(mfcc_means) > 1 else 0.0,
            "MFCC_3": float(mfcc_means[2]) if len(mfcc_means) > 2 else 0.0,
        }
    except Exception:
        return {
            "Jitter": 0.0,
            "Shimmer": 0.0,
            "Pitch_STD": 0.0,
            "ZCR": 0.0,
            "MFCC_1": 0.0,
            "MFCC_2": 0.0,
            "MFCC_3": 0.0,
        }


def calculate_voice_score(features_dict):
    model = get_voice_model()
    feature_values = [
        [features_dict.get(column, 0.0) for column in VOICE_FEATURE_COLUMNS]
    ]

    if model is not None and hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(feature_values)[0][1])
    elif model is not None and hasattr(model, "predict"):
        probability = float(model.predict(feature_values)[0])
    else:
        jitter = features_dict.get("Jitter", 0.0)
        shimmer = features_dict.get("Shimmer", 0.0)
        pitch_std = features_dict.get("Pitch_STD", 0.0)
        probability = min(
            1.0, (jitter * 12.0 + shimmer * 8.0 + pitch_std / 120.0) / 3.0
        )

    return round(max(0.0, min(probability * 100.0, 100.0)), 1)


def update_final_score(session_record):
    scores = [
        score
        for score in [session_record.voice_score, session_record.spiral_score]
        if score is not None
    ]
    session_record.final_score = round(sum(scores) / len(scores), 1) if scores else None


def generate_clinical_analysis(features_dict):
    jitter = parse_float(features_dict.get("Jitter")) or 0.0
    shimmer = parse_float(features_dict.get("Shimmer")) or 0.0
    pitch_std = parse_float(features_dict.get("Pitch_STD")) or 0.0

    jitter_threshold = VOICE_ANALYSIS_THRESHOLDS["Jitter"]
    shimmer_threshold = VOICE_ANALYSIS_THRESHOLDS["Shimmer"]
    pitch_threshold = VOICE_ANALYSIS_THRESHOLDS["Pitch_STD"]

    jitter_line = (
        f"Jitter is elevated at {jitter:.4f}, above the baseline of {jitter_threshold:.4f}."
        if jitter > jitter_threshold
        else f"Jitter is within the expected range at {jitter:.4f}, at or below the {jitter_threshold:.4f} baseline."
    )

    shimmer_line = (
        f"Shimmer is elevated at {shimmer:.4f}, suggesting more amplitude instability than the {shimmer_threshold:.4f} baseline."
        if shimmer > shimmer_threshold
        else f"Shimmer remains controlled at {shimmer:.4f}, staying near or below the {shimmer_threshold:.4f} baseline."
    )

    pitch_line = (
        f"Pitch variability is higher than expected at {pitch_std:.2f} Hz, above the {pitch_threshold:.2f} Hz baseline."
        if pitch_std > pitch_threshold
        else f"Pitch variability is relatively stable at {pitch_std:.2f} Hz and remains within the {pitch_threshold:.2f} Hz baseline."
    )

    return [jitter_line, shimmer_line, pitch_line]


def format_score(value):
    if value is None:
        return "Pending"
    return f"{value:.1f}%"


def format_timestamp(timestamp_value):
    if timestamp_value is None:
        return ""
    return timestamp_value.strftime("%Y-%m-%d %I:%M %p")


def serialize_session(session_record):
    voice_metrics = {}
    if session_record.voice_metrics:
        try:
            voice_metrics = json.loads(session_record.voice_metrics)
        except json.JSONDecodeError:
            voice_metrics = {}

    timestamp_text = format_timestamp(session_record.timestamp)
    voice_analysis = generate_clinical_analysis(voice_metrics) if voice_metrics else []
    return {
        "id": session_record.id,
        "date": (
            session_record.timestamp.strftime("%Y-%m-%d")
            if session_record.timestamp
            else ""
        ),
        "time": (
            session_record.timestamp.strftime("%I:%M %p")
            if session_record.timestamp
            else ""
        ),
        "full_timestamp": timestamp_text,
        "voice_score": session_record.voice_score,
        "spiral_score": session_record.spiral_score,
        "final_score": session_record.final_score,
        "voice_text": format_score(session_record.voice_score),
        "spiral_text": format_score(session_record.spiral_score),
        "final_text": format_score(session_record.final_score),
        "voice_metrics": voice_metrics,
        "voice_analysis": voice_analysis,
    }


def resolve_session(session_id):
    if session_id is not None:
        session_record = ScreeningSession.query.filter_by(
            id=session_id, user_id=current_user.id
        ).first()
        if session_record is not None:
            return session_record

    return (
        ScreeningSession.query.filter_by(user_id=current_user.id)
        .order_by(ScreeningSession.timestamp.desc(), ScreeningSession.id.desc())
        .first()
    )


def render_auth_page(template_name, **context):
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template(template_name, **context)


def build_report_pdf(session_record):
    session_data = serialize_session(session_record)
    features_dict = session_data["voice_metrics"]
    analysis_points = generate_clinical_analysis(features_dict)

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    page_width = pdf.w - 20

    pdf.set_fill_color(13, 110, 253)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(
        page_width, 16, "NeuroScreen Clinical Report", ln=True, align="C", fill=True
    )

    pdf.ln(4)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_text_color(15, 23, 42)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 22, "Header Image Placeholder", border=1, fill=True, align="C")
    pdf.set_xy(90, pdf.get_y())
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        page_width - 70,
        6,
        f"Session ID: {session_record.id}\nDate: {session_data['full_timestamp']}\nPatient: {session_record.user.username}",
        border=1,
        fill=True,
    )

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Clinical Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        f"Voice score: {session_data['voice_text']}\nSpiral score: {session_data['spiral_text']}\nFinal score: {session_data['final_text']}",
        border=1,
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Voice Diagnostics", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(55, 8, "Metric", border=1, fill=True)
    pdf.cell(45, 8, "Measured", border=1, fill=True)
    pdf.cell(0, 8, "Clinical Note", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)

    metric_notes = {
        "Jitter": "Voice timing stability",
        "Shimmer": "Amplitude stability",
        "Pitch_STD": "Pitch variation",
        "ZCR": "Speech waveform activity",
        "MFCC_1": "Spectral shape",
        "MFCC_2": "Spectral shape",
        "MFCC_3": "Spectral shape",
    }

    for metric_name in VOICE_FEATURE_COLUMNS:
        measured_value = features_dict.get(metric_name, 0.0)
        pdf.cell(55, 8, metric_name, border=1)
        pdf.cell(45, 8, f"{measured_value:.4f}", border=1)
        pdf.cell(0, 8, metric_notes.get(metric_name, ""), border=1, ln=True)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Explainable AI Analysis", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for point in analysis_points:
        pdf.multi_cell(0, 6, f"- {point}", ln=1)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Spiral Assessment", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Spiral results are reserved for the drawing analysis workflow. This section is intentionally left available for future image-based scoring and clinician review.",
        border=1,
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Clinical interpretation note: This report is a screening summary and should be reviewed alongside the patient history, examination findings, and any additional neurological assessment.",
    )

    buffer = BytesIO()
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin1")
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Name and password are required.", "danger")
            return render_template("register.html")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user is not None:
            flash("That account already exists.", "danger")
            return render_template("register.html")

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user is None or not check_password_hash(user.password_hash, password):
            flash("Invalid name or password.", "danger")
            return render_template("login.html")

        login_user(user)
        flash("Signed in.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        session_record = ScreeningSession(user_id=current_user.id)
        db.session.add(session_record)
        db.session.commit()
        flash("New screening session created.", "success")
        return redirect(url_for("session_hub", session_id=session_record.id))

    recent_session = (
        ScreeningSession.query.filter_by(user_id=current_user.id)
        .order_by(ScreeningSession.timestamp.desc(), ScreeningSession.id.desc())
        .first()
    )
    total_sessions = ScreeningSession.query.filter_by(user_id=current_user.id).count()

    return render_template(
        "dashboard.html",
        recent_session=serialize_session(recent_session) if recent_session else None,
        total_sessions=total_sessions,
    )


@app.route("/session/", defaults={"session_id": None})
@app.route("/session/<int:session_id>/")
@login_required
def session_hub(session_id):
    session_record = resolve_session(session_id)
    if session_record is None:
        flash("Start a new screening session first.", "warning")
        return redirect(url_for("dashboard"))

    return render_template(
        "session.html",
        session_record=serialize_session(session_record),
    )


@app.route("/test/voice/", defaults={"session_id": None})
@app.route("/test/voice/<int:session_id>/")
@login_required
def voice_test(session_id):
    session_record = resolve_session(session_id)
    if session_record is None:
        flash("Start a new screening session first.", "warning")
        return redirect(url_for("dashboard"))

    return render_template(
        "voice_test.html",
        session_record=serialize_session(session_record),
    )


@app.route("/test/spiral/", defaults={"session_id": None})
@app.route("/test/spiral/<int:session_id>/")
@login_required
def spiral_test(session_id):
    session_record = resolve_session(session_id)
    if session_record is None:
        flash("Start a new screening session first.", "warning")
        return redirect(url_for("dashboard"))

    return render_template(
        "spiral_test.html",
        session_record=serialize_session(session_record),
    )


@app.route("/upload_voice/", methods=["POST"])
@login_required
def upload_voice():
    session_id = request.form.get("session_id", type=int)
    session_record = resolve_session(session_id)
    if session_record is None:
        message = "That screening session could not be found."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 404
        flash(message, "danger")
        return redirect(url_for("dashboard"))

    audio_file = request.files.get("file")
    if audio_file is None or audio_file.filename == "":
        message = "Please record or upload a voice sample first."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 400
        flash(message, "danger")
        return redirect(url_for("voice_test", session_id=session_record.id))

    if not audio_extension_allowed(audio_file.filename):
        message = "Only WAV audio files are supported."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 400
        flash(message, "danger")
        return redirect(url_for("voice_test", session_id=session_record.id))

    temp_filename = secure_filename(audio_file.filename or "voice_sample.wav")
    temp_path = os.path.join(INSTANCE_DIR, f"{session_record.id}_{temp_filename}")

    try:
        audio_file.save(temp_path)
        features = extract_voice_features(temp_path)
        voice_score = calculate_voice_score(features)

        session_record.voice_score = voice_score
        session_record.voice_metrics = json.dumps(features, sort_keys=True)
        update_final_score(session_record)
        db.session.commit()

        redirect_url = url_for("session_hub", session_id=session_record.id)
        if wants_json_response():
            return jsonify(
                {
                    "success": True,
                    "message": "Voice test saved.",
                    "voice_score": voice_score,
                    "voice_metrics": features,
                    "clinical_analysis": generate_clinical_analysis(features),
                    "redirect_url": redirect_url,
                }
            )

        flash("Voice test saved.", "success")
        return redirect(redirect_url)
    except Exception as exc:
        db.session.rollback()
        message = f"The voice sample could not be processed: {exc}"
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 500
        flash(message, "danger")
        return redirect(url_for("voice_test", session_id=session_record.id))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/history")
@login_required
def history():
    sessions = (
        ScreeningSession.query.filter_by(user_id=current_user.id)
        .order_by(ScreeningSession.timestamp.desc(), ScreeningSession.id.desc())
        .all()
    )

    serialized_sessions = [
        serialize_session(session_record) for session_record in sessions
    ]

    return render_template(
        "history.html",
        sessions=serialized_sessions,
    )


@app.route("/api/generate_report/", methods=["GET"])
@login_required
def generate_report():
    session_id = request.args.get("session_id", type=int)
    if session_id is None:
        abort(400)

    session_record = ScreeningSession.query.filter_by(
        id=session_id, user_id=current_user.id
    ).first()
    if session_record is None:
        abort(404)

    pdf_buffer = build_report_pdf(session_record)
    download_name = f"neuroscreen_report_session_{session_record.id}.pdf"

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )


ensure_database_schema()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
