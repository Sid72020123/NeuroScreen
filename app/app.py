import os
import sys
import tempfile
from datetime import datetime

import joblib
import librosa
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_user,
    login_required,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "voice_model.pkl")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DEFAULT_SQLITE_PATH = os.path.join(INSTANCE_DIR, "neuroscreen.db")

os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_object(Config)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "SQLITE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}"
)
# app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
#     "MYSQL_URL", "mysql+pymysql://user:password@localhost/neuroscreen"
# )
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    tests = db.relationship("PatientTest", backref="user", lazy=True)


class PatientTest(db.Model):
    __tablename__ = "patient_tests"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    risk_score = db.Column(db.Integer, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained voice model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


tables_created = False


@app.before_request
def create_database_tables():
    global tables_created
    if not tables_created:
        db.create_all()
        tables_created = True


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("register.html")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("That username is already taken.", "danger")
            return render_template("register.html")

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration complete. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user is None or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        login_user(user)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/upload_voice", methods=["POST"])
@login_required
def upload_voice():
    audio_file = request.files.get("file") or request.files.get("voice")

    if audio_file is None or audio_file.filename == "":
        return jsonify({"success": False, "message": "No .wav file was uploaded."}), 400

    filename = secure_filename(audio_file.filename)
    if not filename.lower().endswith(".wav"):
        return (
            jsonify({"success": False, "message": "Only .wav files are supported."}),
            400,
        )

    def extract_voice_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)

            if y is None or len(y) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=50,
                fmax=300,
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

            if f0 is None or voiced_flag is None:
                f0_voiced = np.array([])
                rms_voiced = np.array([])
            else:
                min_len = min(len(f0), len(rms), len(voiced_flag))
                f0 = f0[:min_len]
                rms = rms[:min_len]
                voiced_flag = voiced_flag[:min_len]

                f0_voiced = f0[voiced_flag]
                rms_voiced = rms[voiced_flag]

                f0_voiced = f0_voiced[np.isfinite(f0_voiced)]
                rms_voiced = rms_voiced[np.isfinite(rms_voiced)]

            if len(f0_voiced) > 1:
                periods = np.divide(
                    1.0, f0_voiced, out=np.zeros_like(f0_voiced), where=f0_voiced != 0
                )
                mean_period = np.mean(periods) if len(periods) > 0 else 0.0
                jitter = (
                    np.mean(np.abs(np.diff(periods))) / mean_period
                    if mean_period and mean_period > 0
                    else 0.0
                )
            else:
                jitter = 0.0

            if len(rms_voiced) > 1:
                mean_rms = np.mean(rms_voiced)
                shimmer = (
                    np.mean(np.abs(np.diff(rms_voiced))) / mean_rms
                    if mean_rms and mean_rms > 0
                    else 0.0
                )
            else:
                shimmer = 0.0

            pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            zcr = (
                float(np.mean(librosa.feature.zero_crossing_rate(y)))
                if len(y) > 0
                else 0.0
            )

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
            mfcc_means = np.mean(mfccs, axis=1) if mfccs.size else np.array([])

            mfcc1 = (
                float(mfcc_means[0])
                if len(mfcc_means) > 0 and np.isfinite(mfcc_means[0])
                else 0.0
            )
            mfcc2 = (
                float(mfcc_means[1])
                if len(mfcc_means) > 1 and np.isfinite(mfcc_means[1])
                else 0.0
            )
            mfcc3 = (
                float(mfcc_means[2])
                if len(mfcc_means) > 2 and np.isfinite(mfcc_means[2])
                else 0.0
            )

            return (
                float(jitter),
                float(shimmer),
                float(pitch_std),
                float(zcr),
                mfcc1,
                mfcc2,
                mfcc3,
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            tmp_path = temp_file.name

        features = extract_voice_features(tmp_path)
        feature_columns = [
            "Jitter",
            "Shimmer",
            "Pitch_STD",
            "ZCR",
            "MFCC_1",
            "MFCC_2",
            "MFCC_3",
        ]
        feature_frame = pd.DataFrame([features], columns=feature_columns)

        positive_class_index = (
            1 if hasattr(model, "classes_") and 1 in list(model.classes_) else -1
        )
        positive_probability = float(
            model.predict_proba(feature_frame)[0][positive_class_index]
        )
        risk_score = int(round(positive_probability * 100))
        risk_score = max(0, min(100, risk_score))

        test_record = PatientTest(risk_score=risk_score)
        test_record.user_id = current_user.id
        db.session.add(test_record)
        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Voice test processed successfully.",
                    "risk_score": risk_score,
                }
            ),
            200,
        )
    except Exception as exc:
        db.session.rollback()
        return jsonify({"success": False, "message": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/history")
@login_required
def api_history():
    records = (
        PatientTest.query.filter_by(user_id=current_user.id)
        .order_by(PatientTest.timestamp.asc())
        .all()
    )

    return jsonify(
        {
            "success": True,
            "history": [
                {
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "risk_score": record.risk_score,
                }
                for record in records
            ],
        }
    )


@app.route("/")
@login_required
def home():
    return render_template("index.html", username=current_user.username)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
