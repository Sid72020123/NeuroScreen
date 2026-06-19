import json
import os
import warnings

# Suppress TensorFlow C++ logs (e.g. CUDA cuInit errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Suppress Keras structure warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message=".*The structure of `inputs` doesn't match the expected structure.*")
# Suppress expected fallback behavior in nolds nonlinear math equations
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nolds")

from datetime import datetime, timezone
from io import BytesIO
from zoneinfo import ZoneInfo
import base64
import cv2
import parselmouth
from parselmouth.praat import call
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nolds
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

import sys

# Add the project root directory to the Python path to resolve the import error.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
SPIRAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "spiral_model.keras")
VOICE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "xgboost_pruned_model.pkl")
VOICE_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "standard_scaler_pruned.pkl")
CUSTOM_THRESHOLD = 0.39
ALLOWED_AUDIO_EXTENSIONS = {"wav"}

ALL_22_FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

# Features to drop for the ML model prediction
FEATURES_TO_DROP = [
    "Jitter:DDP",
    "MDVP:PPQ",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "MDVP:Jitter(%)",
]

# Curated list of the most impactful features for PDF reporting
PDF_VOICE_FEATURES = [
    "spread1",
    "spread2",
    "D2",
    "PPE",
    "MDVP:Fo(Hz)",
    "HNR",
    "RPDE",
    "DFA",
]

# Top 3 features for the main dashboard's XAI visualization
TOP_XAI_FEATURES = {
    "spread2": {
        "name": "Vocal Fold Spread (spread2)",
        "healthy_avg": 0.1,
        "pd_avg": 0.4,
        "unit": "",
    },
    "D2": {
        "name": "Correlation Dimension (D2)",
        "healthy_avg": 1.5,
        "pd_avg": 2.5,
        "unit": "",
    },
    "RPDE": {
        "name": "Recurrence Period Density (RPDE)",
        "healthy_avg": 0.4,
        "pd_avg": 0.6,
        "unit": "",
    },
}


ALL_BIOMARKER_REFERENCE = {
    "MDVP:Fo(Hz)": {"type": "range", "desc": "Baseline pitch varies by age/gender."},
    "MDVP:Fhi(Hz)": {"type": "range", "desc": "Maximum vocal pitch."},
    "MDVP:Flo(Hz)": {"type": "range", "desc": "Minimum vocal pitch."},
    "MDVP:Jitter(%)": {"type": "threshold", "value": 0.007, "direction": "higher_is_bad", "desc": "< 0.007%"},
    "MDVP:Jitter(Abs)": {"type": "threshold", "value": 0.00005, "direction": "higher_is_bad", "desc": "< 0.00005"},
    "MDVP:RAP": {"type": "threshold", "value": 0.003, "direction": "higher_is_bad", "desc": "< 0.003"},
    "MDVP:PPQ": {"type": "threshold", "value": 0.003, "direction": "higher_is_bad", "desc": "< 0.003"},
    "Jitter:DDP": {"type": "threshold", "value": 0.009, "direction": "higher_is_bad", "desc": "< 0.009"},
    "MDVP:Shimmer": {"type": "threshold", "value": 0.05, "direction": "higher_is_bad", "desc": "< 0.05"},
    "MDVP:Shimmer(dB)": {"type": "threshold", "value": 0.3, "direction": "higher_is_bad", "desc": "< 0.3 dB"},
    "Shimmer:APQ3": {"type": "threshold", "value": 0.015, "direction": "higher_is_bad", "desc": "< 0.015"},
    "Shimmer:APQ5": {"type": "threshold", "value": 0.02, "direction": "higher_is_bad", "desc": "< 0.02"},
    "MDVP:APQ": {"type": "threshold", "value": 0.025, "direction": "higher_is_bad", "desc": "< 0.025"},
    "Shimmer:DDA": {"type": "threshold", "value": 0.046, "direction": "higher_is_bad", "desc": "< 0.046"},
    "NHR": {"type": "threshold", "value": 0.02, "direction": "higher_is_bad", "desc": "< 0.02"},
    "HNR": {"type": "threshold", "value": 20.0, "direction": "lower_is_bad", "desc": "> 20.0 dB"},
    "RPDE": {"type": "threshold", "value": 0.5, "direction": "higher_is_bad", "desc": "< 0.5"},
    "DFA": {"type": "threshold", "value": 0.7, "direction": "higher_is_bad", "desc": "< 0.7"},
    "spread1": {"type": "threshold", "value": -5.0, "direction": "higher_is_bad", "desc": "< -5.0"},
    "spread2": {"type": "threshold", "value": 0.2, "direction": "higher_is_bad", "desc": "< 0.2"},
    "D2": {"type": "threshold", "value": 2.2, "direction": "higher_is_bad", "desc": "< 2.2"},
    "PPE": {"type": "threshold", "value": 0.2, "direction": "higher_is_bad", "desc": "< 0.2"},
}


os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "uploads"), exist_ok=True)

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
    return {"year": datetime.now(timezone.utc).year}


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    is_doctor = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

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
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    voice_score = db.Column(db.Float, nullable=True)
    spiral_score = db.Column(db.Float, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    voice_metrics = db.Column(db.Text, nullable=True)
    spiral_metrics = db.Column(db.Text, nullable=True)
    medication_state = db.Column(db.String(20), default="UNMEDICATED")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def ensure_database_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        columns = {column["name"] for column in inspector.get_columns("screening_sessions")}
        with db.engine.begin() as connection:
            if "voice_metrics" not in columns:
                connection.execute(text("ALTER TABLE screening_sessions ADD COLUMN voice_metrics TEXT"))
            if "spiral_metrics" not in columns:
                connection.execute(text("ALTER TABLE screening_sessions ADD COLUMN spiral_metrics TEXT"))
            if "medication_state" not in columns:
                connection.execute(
                    text("ALTER TABLE screening_sessions ADD COLUMN medication_state VARCHAR(20) DEFAULT 'UNMEDICATED'")
                )

        user_columns = {column["name"] for column in inspector.get_columns("users")}
        with db.engine.begin() as connection:
            if "age" not in user_columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN age INTEGER"))
            if "gender" not in user_columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN gender VARCHAR(10)"))
            if "is_doctor" not in user_columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN is_doctor BOOLEAN DEFAULT 0"))


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
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


_VOICE_MODEL = None
_VOICE_SCALER = None


def get_voice_dependencies():
    """Loads and caches the pruned XGBoost model and the corresponding StandardScaler."""
    global _VOICE_MODEL, _VOICE_SCALER
    if _VOICE_MODEL is None or _VOICE_SCALER is None:
        try:
            if os.path.exists(VOICE_MODEL_PATH):
                _VOICE_MODEL = joblib.load(VOICE_MODEL_PATH)
                print(f"Successfully loaded pruned voice model from {VOICE_MODEL_PATH}")
            else:
                print(f"Warning: Pruned voice model not found at {VOICE_MODEL_PATH}")
                _VOICE_MODEL = None

            if os.path.exists(VOICE_SCALER_PATH):
                _VOICE_SCALER = joblib.load(VOICE_SCALER_PATH)
                print(f"Successfully loaded pruned voice scaler from {VOICE_SCALER_PATH}")
            else:
                print(f"Warning: Pruned voice scaler not found at {VOICE_SCALER_PATH}")
                _VOICE_SCALER = None
        except Exception as e:
            print(f"Error loading voice dependencies: {e}")
            _VOICE_MODEL = None
            _VOICE_SCALER = None
    return _VOICE_MODEL, _VOICE_SCALER


_SPIRAL_MODEL = None


def get_spiral_model():
    global _SPIRAL_MODEL
    if _SPIRAL_MODEL is None:
        if os.path.exists(SPIRAL_MODEL_PATH):
            try:
                # Load model without compiling to avoid warnings on optimizer state
                # when only doing inference.
                _SPIRAL_MODEL = tf.keras.models.load_model(SPIRAL_MODEL_PATH, compile=False)
                print(f"Successfully loaded spiral model from {SPIRAL_MODEL_PATH}")
            except Exception as e:
                print(f"Error loading spiral model: {e}")
        else:
            print(f"Warning: Spiral model not found at {SPIRAL_MODEL_PATH}")
    return _SPIRAL_MODEL


def extract_praat_features(filepath):
    """
    Extracts all 22 acoustic features using Parselmouth (Praat) and Nolds to match
    the model's training data.
    """
    try:
        sound = parselmouth.Sound(filepath)
        pitch = sound.to_pitch()
        y, sr = librosa.load(filepath, sr=None)

        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        pulses = call([sound, pitch], "To PointProcess (cc)")
        jitter_pct = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap = call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = rap * 3

        shimmer_pct = call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3 = call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = call([sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11 = call([sound, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda = apq3 * 3

        harmonicity = sound.to_harmonicity_cc()
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 10.0 ** (-hnr / 10.0) if hnr and not np.isnan(hnr) else 0.0

        y_sub = y[:3000] if len(y) > 3000 else y
        dfa = nolds.dfa(y_sub)
        d2 = nolds.corr_dim(y_sub, emb_dim=2)
        rpde = nolds.sampen(y_sub)

        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 0:
            log_f0 = np.log(f0_voiced)
            spread1 = np.std(log_f0)
            spread2 = np.var(log_f0)
            hist, bin_edges = np.histogram(log_f0, bins="fd", density=True)
            p = hist * np.diff(bin_edges)
            p = p[p > 0]
            ppe = -np.sum(p * np.log2(p))
        else:
            spread1, spread2, ppe = 0.0, 0.0, 0.0

        return [
            mean_pitch,
            max_pitch,
            min_pitch,
            jitter_pct,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer_pct,
            shimmer_db,
            apq3,
            apq5,
            apq11,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe,
        ]

    except Exception as e:
        # Praat can fail on silent/corrupt files. Return NaNs for graceful failure.
        print(f"Error processing {filepath} with Parselmouth: {e}")
        return [np.nan] * 22


def update_final_score(session_record):
    scores = [score for score in [session_record.voice_score, session_record.spiral_score] if score is not None]
    session_record.final_score = round(sum(scores) / len(scores), 1) if scores else None


def generate_clinical_analysis(features_dict, user=None):
    analysis_points = []

    spread2_ref = ALL_BIOMARKER_REFERENCE.get("spread2", {}).get("value", 0.2)
    d2_ref = ALL_BIOMARKER_REFERENCE.get("D2", {}).get("value", 2.2)
    rpde_ref = ALL_BIOMARKER_REFERENCE.get("RPDE", {}).get("value", 0.5)

    if user:
        if user.gender and user.gender.lower() == "female":
            spread2_ref *= 1.1
            rpde_ref *= 1.05
        if user.age and user.age > 65:
            spread2_ref *= 1.15
            d2_ref *= 1.1

    spread2 = parse_float(features_dict.get("spread2"))
    if spread2 is not None:
        if spread2 > spread2_ref:
            analysis_points.append(
                f"Vocal Fold Spread (spread2) is elevated at {spread2:.4f} (Baseline: {spread2_ref:.4f}), suggesting potential vocal impairment."
            )
        else:
            analysis_points.append(
                f"Vocal Fold Spread (spread2) is within normal demographic parameters ({spread2:.4f} <= {spread2_ref:.4f})."
            )

    d2 = parse_float(features_dict.get("D2"))
    if d2 is not None:
        if d2 > d2_ref:
            analysis_points.append(
                f"Correlation Dimension (D2) indicates irregular vocal patterns ({d2:.4f} > {d2_ref:.4f})."
            )
        else:
            analysis_points.append(
                f"Correlation Dimension (D2) indicates stable phonation for demographics ({d2:.4f})."
            )

    rpde = parse_float(features_dict.get("RPDE"))
    if rpde is not None:
        if rpde > rpde_ref:
            analysis_points.append(
                f"RPDE score is {rpde:.4f}, showing increased vocal noise variance compared to baseline ({rpde_ref:.4f})."
            )
        else:
            analysis_points.append(f"RPDE score is controlled at {rpde:.4f}, indicating healthy vocal variance.")

    if not analysis_points:
        analysis_points.append("Insufficient data to generate specific vocal biomarker insights.")

    return analysis_points


def format_score(value):
    if value is None:
        return "Pending"
    return f"{value:.1f}%"


def format_timestamp(timestamp_value):
    if timestamp_value is None:
        return ""
    # Assume timestamp_value is a naive datetime in UTC from the database
    utc_dt = timestamp_value.replace(tzinfo=ZoneInfo("UTC"))
    # Convert to Indian Standard Time
    ist_dt = utc_dt.astimezone(ZoneInfo("Asia/Kolkata"))
    return ist_dt.strftime("%Y-%m-%d %I:%M %p")


def get_score_classes(score):
    """Return a dictionary of Tailwind CSS classes based on a score."""
    if score is None:
        return {
            "text": "text-slate-500",
            "bg": "bg-slate-100",
            "border": "border-slate-200",
            "name": "Pending",
        }
    elif score <= 40:
        return {
            "text": "text-emerald-600",
            "bg": "bg-emerald-50/50",
            "border": "border-emerald-100",
            "name": "Low Risk",
        }
    elif score <= 60:
        return {
            "text": "text-amber-600",
            "bg": "bg-amber-50/50",
            "border": "border-amber-100",
            "name": "Moderate Risk",
        }
    else:
        return {
            "text": "text-rose-600",
            "bg": "bg-rose-50/50",
            "border": "border-rose-100",
            "name": "High Risk",
        }


def serialize_session(session_record, session_number=None):
    if session_number is None:
        all_user_sessions = (
            ScreeningSession.query.filter_by(user_id=session_record.user_id)
            .order_by(ScreeningSession.timestamp.asc())
            .all()
        )
        try:
            session_number = all_user_sessions.index(session_record) + 1
        except ValueError:
            session_number = session_record.id

    voice_metrics = {}
    if session_record.voice_metrics:
        try:
            voice_metrics = json.loads(session_record.voice_metrics)
        except json.JSONDecodeError:
            voice_metrics = {}

    spiral_metrics = {}
    if session_record.spiral_metrics:
        try:
            spiral_metrics = json.loads(session_record.spiral_metrics)
        except json.JSONDecodeError:
            spiral_metrics = {}

    timestamp_text = format_timestamp(session_record.timestamp)
    voice_analysis = (
        generate_clinical_analysis(voice_metrics, getattr(session_record, "user", None)) if voice_metrics else []
    )
    return {
        "id": session_record.id,
        "medication_state": getattr(session_record, "medication_state", "UNMEDICATED"),
        "session_number": session_number,
        "date": (session_record.timestamp.strftime("%Y-%m-%d") if session_record.timestamp else ""),
        "time": (session_record.timestamp.strftime("%I:%M %p") if session_record.timestamp else ""),
        "full_timestamp": timestamp_text,
        "voice_score": session_record.voice_score,
        "spiral_score": session_record.spiral_score,
        "final_score": session_record.final_score,
        "voice_text": format_score(session_record.voice_score),
        "spiral_text": format_score(session_record.spiral_score),
        "final_text": format_score(session_record.final_score),
        "voice_metrics": voice_metrics,
        "voice_analysis": voice_analysis,
        "spiral_analysis": spiral_metrics.get("analysis", []),
        "voice_xai_image_url": voice_metrics.get("voice_xai_path"),
        "spiral_xai_image_url": spiral_metrics.get("xai_image_url"),
        "spiral_original_image_url": spiral_metrics.get("original_image_url"),
        "kinematic_variance": spiral_metrics.get("kinematic_variance"),
        "final_score_classes": get_score_classes(session_record.final_score),
        "voice_score_classes": get_score_classes(session_record.voice_score),
        "spiral_score_classes": get_score_classes(session_record.spiral_score),
    }


def resolve_session(session_id):
    if session_id is not None:
        session_record = ScreeningSession.query.filter_by(id=session_id, user_id=current_user.id).first()
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
    features_dict = session_data.get("voice_metrics", {})
    voice_analysis_points = session_data.get("voice_analysis", [])

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    page_width = pdf.w - 30

    # --- Header ---
    pdf.set_fill_color(13, 148, 136)  # Teal-600
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 18, " NeuroScreen Clinical Report", ln=1, align="L", fill=True)
    pdf.ln(6)

    # --- Patient Details Box ---
    pdf.set_draw_color(203, 213, 225)
    pdf.set_text_color(30, 41, 59)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 10)

    pdf.cell(page_width / 4, 10, f" Patient: {session_record.user.username}", border="L T B", fill=True, ln=0)
    pdf.cell(
        page_width / 4, 10, f" Session ID: #{session_data['session_number']}", border="T B", fill=True, align="C", ln=0
    )
    pdf.cell(
        page_width / 4,
        10,
        f" Meds: {session_data.get('medication_state', 'N/A')}",
        border="T B",
        fill=True,
        align="C",
        ln=0,
    )
    pdf.cell(page_width / 4, 10, f" Date: {session_data['full_timestamp']}", border="R T B", fill=True, align="R", ln=1)
    pdf.ln(10)

    # --- 1. Overall Neurological Risk Assessment ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "1. Overall Neurological Risk Assessment", ln=1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(40, 8, "Final Risk Score:", ln=0)

    pdf.set_font("Helvetica", "B", 20)
    score = session_data["final_score"]
    if score is None:
        pdf.set_text_color(148, 163, 184)  # slate-400
    elif score > 60:
        pdf.set_text_color(225, 29, 72)  # rose-600
    elif score > 40:
        pdf.set_text_color(217, 119, 6)  # amber-600
    else:
        pdf.set_text_color(5, 150, 105)  # emerald-600

    score_text = (
        f"{session_data['final_text']}  ({session_data['final_score_classes']['name']})"
        if score is not None
        else "Pending"
    )
    pdf.cell(0, 8, score_text, ln=1)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.set_x(15)
    pdf.multi_cell(
        0,
        6,
        f"Voice Assessment: {session_data['voice_text']}   |   Motor Kinematics Assessment: {session_data['spiral_text']}",
    )
    pdf.ln(6)

    # Divider
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # --- 2. Voice Acoustic Diagnostics ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "2. Voice Acoustic Diagnostics", ln=1)

    audio_duration = features_dict.get("audio_duration")
    if audio_duration:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(13, 148, 136)  # Teal-600
        pdf.cell(0, 6, f"[ Data Quality: High | Audio Length: {audio_duration:.1f} seconds ]", ln=1)
        pdf.ln(2)

    if voice_analysis_points:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        for point in voice_analysis_points:
            pdf.set_x(15)
            pdf.multi_cell(0, 5, f" - {point}")
        pdf.ln(4)

    # Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(241, 245, 249)  # slate-100
    pdf.set_text_color(15, 23, 42)
    pdf.set_draw_color(203, 213, 225)

    col_w = [40, 30, 110]
    pdf.cell(col_w[0], 8, " Biomarker", border=1, fill=True, ln=0)
    pdf.cell(col_w[1], 8, " Value", border=1, fill=True, align="C", ln=0)
    pdf.cell(col_w[2], 8, " Clinical Note", border=1, fill=True, ln=1)

    pdf.set_font("Helvetica", "", 9)
    fill = False

    for metric_name in PDF_VOICE_FEATURES:
        measured_value = features_dict.get(metric_name)
        ref = ALL_BIOMARKER_REFERENCE.get(metric_name)

        pdf.set_fill_color(248, 250, 252)
        pdf.cell(col_w[0], 8, f" {metric_name}", border=1, fill=fill, ln=0)

        if measured_value is not None:
            val_str = f"{measured_value:.4f}"
            pdf.set_text_color(15, 23, 42)
            if ref and ref.get("type") == "threshold":
                if ref.get("direction") == "higher_is_bad" and measured_value > ref.get("value"):
                    pdf.set_text_color(225, 29, 72)
                elif ref.get("direction") == "lower_is_bad" and measured_value < ref.get("value"):
                    pdf.set_text_color(225, 29, 72)
                else:
                    pdf.set_text_color(5, 150, 105)
            pdf.cell(col_w[1], 8, val_str, border=1, align="C", fill=fill, ln=0)
            pdf.set_text_color(15, 23, 42)
        else:
            pdf.set_text_color(148, 163, 184)
            pdf.cell(col_w[1], 8, "N/A", border=1, align="C", fill=fill, ln=0)
            pdf.set_text_color(15, 23, 42)

        note = ref.get("desc", "") if ref else ""
        pdf.cell(col_w[2], 8, f" {note}", border=1, ln=1, fill=fill)
        fill = not fill

    pdf.ln(6)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # --- 3. Spiral Assessment ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "3. Motor Kinematics (Spiral Assessment)", ln=1)

    spiral_analysis_points = session_data.get("spiral_analysis", [])
    kinematic_variance = session_data.get("kinematic_variance")

    if kinematic_variance is not None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(225, 29, 72) if kinematic_variance >= 40 else pdf.set_text_color(5, 150, 105)
        pdf.cell(0, 6, f"[ Kinematic Variance Score: {kinematic_variance:.1f}%  (Baseline: < 40%) ]", ln=1)
        pdf.ln(2)

    pdf.set_text_color(71, 85, 105)
    pdf.set_font("Helvetica", "", 10)
    if spiral_analysis_points:
        for point in spiral_analysis_points:
            pdf.set_x(15)
            pdf.multi_cell(0, 5, f" - {point}")
    else:
        pdf.set_x(15)
        pdf.multi_cell(0, 6, "N/A - Insufficient Data: Spiral test was not completed for this session.")

    pdf.ln(6)

    # Images
    original_image_url = session_data.get("spiral_original_image_url")
    xai_image_url = session_data.get("spiral_xai_image_url")

    img_w = 70
    if original_image_url and xai_image_url:
        orig_path = os.path.join(BASE_DIR, original_image_url.lstrip("/"))
        xai_path = os.path.join(BASE_DIR, xai_image_url.lstrip("/"))
        if os.path.exists(orig_path) and os.path.exists(xai_path):
            if pdf.get_y() + img_w > 260:
                pdf.add_page()
            y_img = pdf.get_y()

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(15, 23, 42)
            pdf.text(35, y_img + 3, "Original Drawing")
            pdf.text(115, y_img + 3, "AI Heatmap Overlay")

            y_img += 5
            pdf.image(orig_path, w=img_w, x=20, y=y_img)
            pdf.image(xai_path, w=img_w, x=100, y=y_img)
            pdf.set_y(y_img + img_w + 5)

    elif xai_image_url:
        xai_path = os.path.join(BASE_DIR, xai_image_url.lstrip("/"))
        if os.path.exists(xai_path):
            if pdf.get_y() + img_w > 260:
                pdf.add_page()
            y_img = pdf.get_y()
            pdf.set_font("Helvetica", "B", 9)
            pdf.text(pdf.w / 2 - 15, y_img + 3, "AI Heatmap Overlay")
            y_img += 5
            pdf.image(xai_path, w=img_w, x=(pdf.w - img_w) / 2, y=y_img)
            pdf.set_y(y_img + img_w + 5)

    voice_xai_url = session_data.get("voice_xai_image_url")
    if voice_xai_url:
        voice_xai_path = os.path.join(BASE_DIR, voice_xai_url.lstrip("/"))
        if os.path.exists(voice_xai_path):
            if pdf.get_y() + 60 > 260:
                pdf.add_page()
            pdf.ln(8)
            y_img = pdf.get_y()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(15, 23, 42)
            pdf.text(15, y_img + 3, "Acoustic Spectrogram (Voice Heatmap)")
            y_img += 5
            pdf.image(voice_xai_path, w=170, x=15, y=y_img)
            pdf.set_y(y_img + 60 + 5)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.set_x(15)
    pdf.multi_cell(
        0,
        4,
        "Disclaimer: This clinical report is a screening summary generated by an AI model and should be reviewed alongside formal medical evaluation, patient history, and neurological assessment by a certified professional. It is not intended as a substitute for professional medical advice, diagnosis, or treatment.",
    )

    buffer = BytesIO()
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin1")
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer


def generate_voice_spectrogram(audio_path, session_id, user_id):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=None)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=None, cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Acoustic Mel-Spectrogram (Voice Heatmap)")
    plt.tight_layout()

    user_uploads_dir = os.path.join(BASE_DIR, "static", "uploads", f"user_{user_id}")
    os.makedirs(user_uploads_dir, exist_ok=True)
    save_path = os.path.join(user_uploads_dir, f"session_{session_id}_voice_xai.jpg")
    plt.savefig(save_path, format="jpg", dpi=150)
    plt.close()
    return f"/static/uploads/user_{user_id}/session_{session_id}_voice_xai.jpg"


def generate_gradcam(
    img_array,
    base_img,
    model,
    session_id,
    user_id,
    last_conv_layer_name="out_relu",
):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_output = preds[:, 0]

    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # --- OpenCV steps for visualization ---
    heatmap_resized = cv2.resize(heatmap, (base_img.shape[1], base_img.shape[0]))

    # Threshold the heatmap to get the "hot" regions
    _, hot_region_thresh = cv2.threshold((heatmap_resized * 255).astype("uint8"), 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(hot_region_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heatmap_jet = cv2.applyColorMap((heatmap_resized * 255).astype("uint8"), cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(base_img, 0.6, heatmap_jet, 0.4, 0)

    kinematic_variance = 0.0
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

        roi = heatmap_resized[y : y + h, x : x + w]
        if roi.size > 0:
            max_intensity = np.max(roi)
            kinematic_variance = float(max_intensity * 100)

    # Save the annotated image
    user_uploads_dir = os.path.join(BASE_DIR, "static", "uploads", f"user_{user_id}")
    os.makedirs(user_uploads_dir, exist_ok=True)
    save_path = os.path.join(user_uploads_dir, f"session_{session_id}_xai.jpg")
    cv2.imwrite(save_path, superimposed_img)

    return f"/static/uploads/user_{user_id}/session_{session_id}_xai.jpg", kinematic_variance


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
        age_val = request.form.get("age", "")
        gender = request.form.get("gender", "")
        doctor_code = request.form.get("doctor_code", "")

        age = int(age_val) if age_val.isdigit() else None
        is_doctor = doctor_code == "NEURO2026"

        if not username or not password:
            flash("Name and password are required.", "danger")
            return render_template("register.html")

        # Case-insensitive query to prevent duplicate handles like 'Admin' vs 'admin'
        existing_user = User.query.filter(db.func.lower(User.username) == db.func.lower(username)).first()
        if existing_user is not None:
            flash("That account already exists.", "danger")
            return render_template("register.html")

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            age=age,
            gender=gender,
            is_doctor=is_doctor,
        )
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
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

        # Case-insensitive query prevents mobile auto-capitalization bugs from failing login
        user = User.query.filter(db.func.lower(User.username) == db.func.lower(username)).first()

        if user is None or not check_password_hash(user.password_hash, password):
            flash("Invalid name or password.", "danger")
            return render_template("login.html")

        login_user(user, remember=True)
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
        medication_state = request.form.get("medication_state", "UNMEDICATED")
        session_record = ScreeningSession(user_id=current_user.id, medication_state=medication_state)
        db.session.add(session_record)
        db.session.commit()
        flash("New screening session created.", "success")
        return redirect(url_for("session_hub", session_id=session_record.id))

    sessions = (
        ScreeningSession.query.filter_by(user_id=current_user.id).order_by(ScreeningSession.timestamp.asc()).all()
    )

    recent_session = sessions[-1] if sessions else None
    total_sessions = len(sessions)

    return render_template(
        "dashboard.html",
        recent_session=(serialize_session(recent_session, session_number=total_sessions) if recent_session else None),
        total_sessions=total_sessions,
    )


@app.route("/clinician")
@login_required
def clinician_dashboard():
    if not current_user.is_doctor:
        flash("Unauthorized access. Clinician portal only.", "danger")
        return redirect(url_for("dashboard"))

    patients = User.query.filter_by(is_doctor=False).all()
    patient_data = []

    for patient in patients:
        sessions = (
            ScreeningSession.query.filter_by(user_id=patient.id).order_by(ScreeningSession.timestamp.desc()).all()
        )
        if sessions:
            recent_session = sessions[0]
            patient_data.append(
                {
                    "patient_id": patient.id,
                    "username": patient.username,
                    "age": patient.age,
                    "gender": patient.gender,
                    "total_sessions": len(sessions),
                    "recent_session_id": recent_session.id,
                    "last_screening_date": (
                        format_timestamp(recent_session.timestamp) if recent_session.timestamp else "N/A"
                    ),
                    "risk_score": recent_session.final_score,
                    "score_classes": (
                        get_score_classes(recent_session.final_score)
                        if recent_session.final_score is not None
                        else None
                    ),
                }
            )

    # Sort by highest risk score first
    patient_data.sort(key=lambda x: (x["risk_score"] is None, -(x["risk_score"] or 0)))

    return render_template("clinician_dashboard.html", patients=patient_data)


@app.route("/model_info")
@login_required
def model_info():
    return render_template("model_info.html")


@app.route("/session/", defaults={"session_id": None})
@app.route("/session/<int:session_id>/")
@login_required
def session_hub(session_id):
    session_record = resolve_session(session_id)
    if session_record is None:
        flash("Start a new screening session first.", "warning")
        return redirect(url_for("dashboard"))

    serialized = serialize_session(session_record)
    metrics = serialized.get("voice_metrics", {})

    def get_status(val, threshold):
        if val is None:
            return "Pending"
        return "Elevated" if val > threshold else "Normal"

    top_voice_features = []
    for feat_key, config in TOP_XAI_FEATURES.items():
        value = metrics.get(feat_key)
        percent = 0
        if value is not None:
            # Calculate position on the 0-100 scale for the bullet chart
            feat_range = config["pd_avg"] - config["healthy_avg"]
            if feat_range > 0:
                percent = ((value - config["healthy_avg"]) / feat_range) * 100
                percent = max(0, min(100, percent))  # Clamp between 0 and 100

        top_voice_features.append({**config, "value": value, "percent": percent})

    kinematic_jitter = session_record.spiral_score
    xai_image_url = serialized.get("spiral_xai_image_url")
    original_image_url = serialized.get("spiral_original_image_url")
    audio_duration = metrics.get("audio_duration")
    kinematic_variance = serialized.get("kinematic_variance")

    clinical_interpretation = "No data available to form a clinical interpretation."
    if session_record.final_score is not None:
        if session_record.final_score > 60:
            clinical_interpretation = "Analysis detected localized high-frequency jitter and elevated vocal biomarkers, suggesting further clinical evaluation is recommended."
        elif session_record.final_score > 40:
            clinical_interpretation = (
                "Analysis detected moderate variations in fine motor or vocal control. Monitoring over time is advised."
            )
        else:
            clinical_interpretation = (
                "Analysis indicates stable fine motor and vocal control. No significant anomalies detected."
            )

    clinical_data = {
        "top_voice_features": top_voice_features,
        "kinematic_jitter": kinematic_jitter,
        "xai_image_url": xai_image_url,
        "original_image_url": original_image_url,
        "audio_duration": audio_duration,
        "kinematic_variance": kinematic_variance,
        "clinical_interpretation": clinical_interpretation,
        "biomarker_references": ALL_BIOMARKER_REFERENCE,
    }

    return render_template("session.html", session_record=serialized, clinical_data=clinical_data)


@app.route("/update_medication/<int:session_id>", methods=["POST"])
@login_required
def update_medication(session_id):
    session_record = resolve_session(session_id)
    if not session_record:
        flash("Session not found.", "danger")
        return redirect(url_for("dashboard"))

    medication_state = request.form.get("medication_state")
    if medication_state in ["UNMEDICATED", "ON", "OFF"]:
        session_record.medication_state = medication_state
        db.session.commit()
        flash("Medication phase tracked successfully.", "success")

    return redirect(url_for("session_hub", session_id=session_id))


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

        # --- DURATION VALIDATION ---
        # Use librosa to check audio length before processing. This prevents
        # downstream errors in feature extraction for very short clips.
        duration = librosa.get_duration(path=temp_path)
        if duration < 3.0:
            message = "Audio recording too short. Please upload a continuous vocal sample of at least 3 seconds for accurate clinical analysis."
            if wants_json_response():
                # Return a specific error format as requested for this validation step.
                return jsonify({"status": "error", "message": message}), 400
            # Fallback for non-JS form submissions.
            flash(message, "danger")
            return redirect(url_for("voice_test", session_id=session_record.id))

        # 1. Load model and scaler
        model, scaler = get_voice_dependencies()
        if model is None or scaler is None:
            message = "Voice analysis model or scaler is not available. Please check server logs."
            if wants_json_response():
                return jsonify({"success": False, "message": message}), 503
            flash(message, "error")
            return redirect(url_for("voice_test", session_id=session_record.id))

        # 2. Extract all 22 Praat features
        feature_values = extract_praat_features(temp_path)
        if any(np.isnan(val) for val in feature_values):
            message = "Could not extract valid audio features. The audio may be silent or corrupted."
            if wants_json_response():
                return jsonify({"success": False, "message": message}), 400
            flash(message, "danger")
            return redirect(url_for("voice_test", session_id=session_record.id))

        # 3. Convert to DataFrame, prune, scale, and predict
        features_df = pd.DataFrame([feature_values], columns=ALL_22_FEATURES)
        pruned_df = features_df.drop(columns=FEATURES_TO_DROP, errors="ignore")

        scaled_features = scaler.transform(pruned_df)
        probability = model.predict_proba(scaled_features)[0][1]

        prediction = "Parkinson's" if probability >= CUSTOM_THRESHOLD else "Healthy"
        voice_score = round(float(probability) * 100, 1)

        # 4. Store results
        # Store all 22 features for comprehensive reporting
        # CRITICAL FIX: Convert numpy types (e.g., float32) to standard Python
        # floats to ensure they are JSON serializable.
        features_dict = {key: float(value) for key, value in zip(ALL_22_FEATURES, feature_values)}
        features_dict["audio_duration"] = duration
        features_dict["voice_xai_path"] = generate_voice_spectrogram(temp_path, session_record.id, current_user.id)
        session_record.voice_score = voice_score
        session_record.voice_metrics = json.dumps(features_dict, sort_keys=True)
        update_final_score(session_record)
        db.session.commit()

        # 5. Return JSON response
        redirect_url = url_for("session_hub", session_id=session_record.id)
        if wants_json_response():
            return jsonify(
                {
                    "success": True,
                    "risk_probability": float(probability),
                    "custom_threshold_applied": CUSTOM_THRESHOLD,
                    "prediction": prediction,
                    "features_analyzed": len(pruned_df.columns),
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


def crop_to_spiral(img_array):
    """
    Aggressively crops the image tightly to all ink strokes.
    Converts to grayscale, isolates ink, finds all contours,
    calculates the absolute bounding box, and applies 10px padding.
    """
    # 2. Convert to grayscale, handling various input formats
    if len(img_array.shape) > 2 and img_array.shape[2] == 4:
        alpha_channel = img_array[:, :, 3]
        rgb_channels = img_array[:, :, :3]
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        blended_img = (rgb_channels * alpha_factor + white_background * (1 - alpha_factor)).astype(np.uint8)
        img_gray = cv2.cvtColor(blended_img, cv2.COLOR_BGR2GRAY)
        img_color = blended_img
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_color = img_array
    else:
        img_gray = img_array.copy()
        img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian Blur and Otsu's thresholding to isolate ink
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Locate all ink strokes
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_color, binary_img

    # 4. Calculate absolute bounding box encompassing ALL contours
    min_x = min(cv2.boundingRect(c)[0] for c in contours)
    min_y = min(cv2.boundingRect(c)[1] for c in contours)
    max_x = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    max_y = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

    # 5. Add a small, strict padding (10 pixels)
    pad = 10
    h, w = img_gray.shape
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(w, max_x + pad)
    max_y = min(h, max_y + pad)

    # 6. Crop the original images
    cropped_color_img = img_color[min_y:max_y, min_x:max_x]
    cropped_binary_img = binary_img[min_y:max_y, min_x:max_x]

    # 7. Return the cropped images
    return cropped_color_img, cropped_binary_img


def _preprocess_spiral_image(img_bytes):
    """
    Applies the strict preprocessing pipeline to an in-memory image.
    This matches the training script to prevent training-serving skew.
    Returns the processed tensor for the model and the base image for XAI.
    """
    # 1. Decode image from bytes.
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Use the aggressive cropping function
    cropped_color_img, cropped_binary_img = crop_to_spiral(img_raw)

    # Resize to (224, 224) for the model input.
    resized_img = cv2.resize(cropped_binary_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert grayscale binary image back to 3-channel for MobileNetV2.
    model_input_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

    # Expand dimensions and apply MobileNetV2 preprocessing.
    img_expanded = np.expand_dims(model_input_img, axis=0)
    processed_img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(img_expanded.astype(np.float32))

    # For Grad-CAM, we need a 3-channel BGR image.
    xai_base_img = cv2.cvtColor(cropped_binary_img, cv2.COLOR_GRAY2BGR)

    return processed_img_tensor, xai_base_img, cropped_color_img


@app.route("/upload_spiral/<int:session_id>", methods=["POST"])
@login_required
def upload_spiral(session_id):
    session_record = resolve_session(session_id)
    if session_record is None:
        message = "That screening session could not be found."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 404
        flash(message, "danger")
        return redirect(url_for("dashboard"))

    model = get_spiral_model()
    if model is None:
        message = "Spiral analysis model is not available. Please check server logs."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 503
        flash(message, "error")
        return redirect(url_for("session_hub", session_id=session_id))

    img_bytes = None
    if "canvas_data" in request.form and request.form.get("canvas_data"):
        base64_data = request.form.get("canvas_data")
        try:
            _, encoded = base64_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        except (ValueError, TypeError) as e:
            message = f"Invalid canvas data received: {e}"
            if wants_json_response():
                return jsonify({"success": False, "message": message}), 400
            flash(message, "error")
            return redirect(url_for("spiral_test", session_id=session_id))
    elif "image" in request.files:
        file = request.files["image"]
        if file and file.filename != "":
            img_bytes = file.read()

    if img_bytes is None:
        message = "No image data received. Please draw a spiral or upload a file."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 400
        flash(message, "error")
        return redirect(url_for("spiral_test", session_id=session_id))

    try:
        # 1. Preprocess the image using the strict, centralized pipeline.
        processed_img, xai_base_img, cropped_color_img = _preprocess_spiral_image(img_bytes)

        # Save the true original uploaded image for comparison
        user_uploads_dir = os.path.join(BASE_DIR, "static", "uploads", f"user_{current_user.id}")
        os.makedirs(user_uploads_dir, exist_ok=True)
        orig_save_path = os.path.join(user_uploads_dir, f"session_{session_id}_original.jpg")

        # Save the beautifully cropped color image so it matches the heatmap overlay perfectly
        cv2.imwrite(orig_save_path, cropped_color_img)

        original_image_url = f"/static/uploads/user_{current_user.id}/session_{session_id}_original.jpg"

        # 2. Perform prediction. The model output is a single probability (sigmoid).
        parkinsons_prob = model.predict(processed_img)[0][0]
        spiral_score = round(float(parkinsons_prob) * 100, 1)
        prediction = "High Risk" if parkinsons_prob >= 0.5 else "Low Risk"

        # 3. Generate Explainable AI (Grad-CAM) heatmap.
        xai_image_url, kinematic_variance = generate_gradcam(
            processed_img,
            xai_base_img,
            model,
            session_id,
            current_user.id,
            "out_relu",
        )

        # 4. Generate analysis text based on the risk score.
        spiral_analysis = []
        if spiral_score >= 50:
            spiral_analysis.append(
                "Analysis indicates potential motor impairments. Irregular kinematics and micro-tremors may be present in the highlighted region."
            )
            spiral_analysis.append(
                "The model detected deviations from a smooth, consistent spiral path, which contributed to the risk score."
            )
        else:
            spiral_analysis.append("The drawing shows stable motor control with smooth, consistent curvature.")

        # 5. Update database record.
        session_record.spiral_score = spiral_score
        session_record.spiral_metrics = json.dumps(
            {
                "analysis": spiral_analysis,
                "xai_image_url": xai_image_url,
                "original_image_url": original_image_url,
                "kinematic_variance": kinematic_variance,
            }
        )
        update_final_score(session_record)
        db.session.commit()

        # 6. Return appropriate response (JSON for AJAX, redirect for form post).
        redirect_url = url_for("session_hub", session_id=session_id)
        if wants_json_response():
            return jsonify(
                {
                    "success": True,
                    "risk_probability": float(parkinsons_prob),
                    "prediction": prediction,
                    "redirect_url": redirect_url,
                }
            )

        flash(
            f"Spiral drawing analyzed successfully! Score: {spiral_score:.1f}%",
            "success",
        )
        return redirect(redirect_url)

    except Exception as e:
        db.session.rollback()
        print(f"An error occurred during spiral analysis: {e}")
        message = "An error occurred during image processing. Please try again."
        if wants_json_response():
            return jsonify({"success": False, "message": message}), 500
        flash(message, "error")
        return redirect(url_for("spiral_test", session_id=session_id))


@app.route("/history")
@login_required
def history():
    sessions = (
        ScreeningSession.query.filter_by(user_id=current_user.id)
        .order_by(ScreeningSession.timestamp.desc(), ScreeningSession.id.desc())
        .all()
    )

    # Order is desc, so we calculate the numbers properly based on the total
    total = len(sessions)
    serialized_sessions = [
        serialize_session(session_record, session_number=total - i) for i, session_record in enumerate(sessions)
    ]

    return render_template(
        "history.html",
        sessions=serialized_sessions,
    )


@app.route("/delete_session/<int:session_id>", methods=["POST"])
@login_required
def delete_session(session_id):
    session_record = ScreeningSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if session_record:
        db.session.delete(session_record)
        db.session.commit()
        flash("Session successfully deleted.", "success")
    else:
        flash("Session not found or unauthorized.", "danger")
    return redirect(url_for("history"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        age_val = request.form.get("age", "")
        gender = request.form.get("gender", "")

        if not username:
            flash("Name cannot be empty.", "danger")
            return redirect(url_for("settings"))

        current_user.username = username
        current_user.age = int(age_val) if age_val.isdigit() else None
        current_user.gender = gender
        db.session.commit()
        flash("Profile settings updated successfully!", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html")


@app.route("/api/generate_report/", methods=["GET"])
@login_required
def generate_report():
    session_id = request.args.get("session_id", type=int)
    if session_id is None:
        abort(400)

    if current_user.is_doctor:
        session_record = ScreeningSession.query.filter_by(id=session_id).first()
    else:
        session_record = ScreeningSession.query.filter_by(id=session_id, user_id=current_user.id).first()

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


@app.route("/api/generate_history_report/", methods=["GET"])
@login_required
def generate_history_report():
    sessions = (
        ScreeningSession.query.filter_by(user_id=current_user.id).order_by(ScreeningSession.timestamp.asc()).all()
    )

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(13, 110, 253)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(
        0,
        16,
        f"NeuroScreen History - {current_user.username}",
        ln=True,
        align="C",
        fill=True,
    )
    pdf.ln(8)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(226, 232, 240)

    # Table Header
    pdf.cell(25, 10, "Session", border=1, fill=True, align="C")
    pdf.cell(50, 10, "Date", border=1, fill=True, align="C")
    pdf.cell(35, 10, "Voice Risk", border=1, fill=True, align="C")
    pdf.cell(35, 10, "Spiral Risk", border=1, fill=True, align="C")
    pdf.cell(35, 10, "Final Risk", border=1, fill=True, ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    for i, s in enumerate(sessions):
        ser = serialize_session(s, session_number=i + 1)
        pdf.cell(25, 10, f"#{ser['session_number']}", border=1, align="C")
        pdf.cell(50, 10, ser["full_timestamp"], border=1, align="C")
        pdf.cell(35, 10, ser["voice_text"], border=1, align="C")
        pdf.cell(35, 10, ser["spiral_text"], border=1, align="C")
        pdf.cell(35, 10, ser["final_text"], border=1, ln=True, align="C")

    buffer = BytesIO()
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin1")
    buffer.write(pdf_bytes)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"neuroscreen_history_{current_user.username}.pdf",
    )


ensure_database_schema()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
