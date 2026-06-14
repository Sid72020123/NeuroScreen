import os
import sys
import tempfile
from datetime import datetime
from io import BytesIO

import joblib
import librosa
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
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
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "voice_model.pkl")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DEFAULT_SQLITE_PATH = os.path.join(INSTANCE_DIR, "neuroscreen.db")

os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_mapping(SECRET_KEY=os.getenv("SECRET_KEY", "default-key"))
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "SQLITE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
setattr(login_manager, "login_view", "login")
setattr(login_manager, "login_message_category", "warning")


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
    test_type = db.Column(db.String(20), nullable=False, default="voice")
    risk_score = db.Column(db.Integer, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained voice model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

database_initialized = False


def ensure_database_schema():
    global database_initialized

    if database_initialized:
        return

    db.create_all()
    inspector = inspect(db.engine)

    if inspector.has_table("patient_tests"):
        columns = {column["name"] for column in inspector.get_columns("patient_tests")}
        if "test_type" not in columns:
            with db.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE patient_tests ADD COLUMN test_type VARCHAR(20) NOT NULL DEFAULT 'voice'"
                    )
                )
        else:
            with db.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE patient_tests SET test_type = 'voice' WHERE test_type IS NULL OR TRIM(test_type) = ''"
                    )
                )

    database_initialized = True


@app.before_request
def initialize_database_on_first_request():
    ensure_database_schema()


def format_timestamp(timestamp_value):
    return {
        "date": timestamp_value.strftime("%Y-%m-%d"),
        "time": timestamp_value.strftime("%I:%M %p"),
        "full": timestamp_value.strftime("%Y-%m-%d %I:%M %p"),
    }


def format_test_type(test_type):
    if test_type == "spiral":
        return "Spiral Drawing Test"
    return "Voice Analysis Test"


def classify_risk(score):
    if score < 34:
        return {
            "label": "Low Risk",
            "tone": "text-sky-700",
            "background": "bg-sky-50",
            "description": "The current screening score is low. Continue routine monitoring and review any symptoms with a clinician if needed.",
        }

    if score < 67:
        return {
            "label": "Moderate Risk",
            "tone": "text-amber-700",
            "background": "bg-amber-50",
            "description": "The score sits in the mid-range. Repeating the test or discussing the result with a healthcare professional may be helpful.",
        }

    return {
        "label": "High Risk",
        "tone": "text-rose-700",
        "background": "bg-rose-50",
        "description": "The screening score is elevated. A professional medical assessment is recommended for proper interpretation.",
    }


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
            float(np.mean(librosa.feature.zero_crossing_rate(y))) if len(y) > 0 else 0.0
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


def calculate_risk_score(features):
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

    if not hasattr(model, "predict_proba"):
        predicted_value = model.predict(feature_frame)[0]
        return int(max(0, min(100, round(float(predicted_value) * 100))))

    probabilities = model.predict_proba(feature_frame)[0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_class_index = classes.index(1)
    else:
        positive_class_index = len(probabilities) - 1

    positive_probability = float(probabilities[positive_class_index])
    risk_score = int(round(positive_probability * 100))
    return max(0, min(100, risk_score))


def serialize_test_record(record):
    formatted_timestamp = format_timestamp(record.timestamp)
    return {
        "id": record.id,
        "date": formatted_timestamp["date"],
        "time": formatted_timestamp["time"],
        "full_timestamp": formatted_timestamp["full"],
        "test_type": record.test_type,
        "test_type_label": format_test_type(record.test_type),
        "risk_score": record.risk_score,
    }


def get_user_tests(user_id, test_type=None):
    query = PatientTest.query.filter_by(user_id=user_id)
    if test_type:
        query = query.filter_by(test_type=test_type)
    return query.order_by(PatientTest.timestamp.asc(), PatientTest.id.asc()).all()


class ClinicalReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, "NeuroScreen Clinical Report")
        self.ln(8)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(71, 85, 105)
        self.cell(0, 5, "Premium neurological screening summary")
        self.ln(5)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def build_clinical_report_pdf(user, record):
    formatted_timestamp = format_timestamp(record.timestamp)
    report_type = format_test_type(record.test_type)
    risk_profile = classify_risk(record.risk_score)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %I:%M %p")

    pdf = ClinicalReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "NeuroScreen Clinical Report")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(
        0,
        6,
        "A structured summary of the screening result, generated for clinical review and record keeping.",
    )
    pdf.ln(2)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "Patient Information", fill=True)
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 8, "Patient Name", border=1)
    pdf.cell(0, 8, str(user.username), border=1)
    pdf.ln(8)
    pdf.cell(60, 8, "Record ID", border=1)
    pdf.cell(0, 8, str(record.id), border=1)
    pdf.ln(8)
    pdf.cell(60, 8, "Generated At", border=1)
    pdf.cell(0, 8, generated_at, border=1)
    pdf.ln(8)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "Test Details", fill=True)
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 8, "Test Type", border=1)
    pdf.cell(0, 8, report_type, border=1)
    pdf.ln(8)
    pdf.cell(60, 8, "Date", border=1)
    pdf.cell(0, 8, formatted_timestamp["date"], border=1)
    pdf.ln(8)
    pdf.cell(60, 8, "Exact Time", border=1)
    pdf.cell(0, 8, formatted_timestamp["time"], border=1)
    pdf.ln(8)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "Result Summary", fill=True)
    pdf.ln(9)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 14, f"Risk Score: {record.risk_score}/100")
    pdf.ln(14)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 8, risk_profile["label"])
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, risk_profile["description"])
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "Clinical Notes", fill=True)
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        "This report is generated from the NeuroScreen screening workflow and should be interpreted in the context of the patient's complete medical picture.",
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf_path = temp_pdf.name

    try:
        pdf.output(temp_pdf_path)
        with open(temp_pdf_path, "rb") as file_handle:
            return BytesIO(file_handle.read())
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

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

        user = User()
        user.username = username
        user.password_hash = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration complete. Please log in.", "success")
        return redirect(url_for("login"))

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
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        login_user(user)
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    total_tests = PatientTest.query.filter_by(user_id=current_user.id).count()
    voice_tests = PatientTest.query.filter_by(
        user_id=current_user.id, test_type="voice"
    ).count()
    spiral_tests = PatientTest.query.filter_by(
        user_id=current_user.id, test_type="spiral"
    ).count()
    latest_test = (
        PatientTest.query.filter_by(user_id=current_user.id)
        .order_by(PatientTest.timestamp.desc(), PatientTest.id.desc())
        .first()
    )
    latest_payload = serialize_test_record(latest_test) if latest_test else None

    return render_template(
        "dashboard.html",
        total_tests=total_tests,
        voice_tests=voice_tests,
        spiral_tests=spiral_tests,
        latest_test=latest_payload,
    )


@app.route("/test/voice", methods=["GET", "POST"])
@app.route("/upload_voice", methods=["GET", "POST"])
@login_required
def voice_test():
    result = None
    if request.method == "POST":
        audio_file = request.files.get("file") or request.files.get("voice")

        if audio_file is None or audio_file.filename == "":
            flash("No .wav file was uploaded.", "danger")
        else:
            filename = secure_filename(audio_file.filename or "")
            if not filename.lower().endswith(".wav"):
                flash("Only .wav files are supported.", "danger")
            else:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_file:
                        audio_file.save(temp_file.name)
                        tmp_path = temp_file.name

                    features = extract_voice_features(tmp_path)
                    risk_score = calculate_risk_score(features)

                    test_record = PatientTest()
                    test_record.user_id = current_user.id
                    test_record.risk_score = risk_score
                    test_record.test_type = "voice"
                    db.session.add(test_record)
                    db.session.commit()

                    result = {
                        "record": serialize_test_record(test_record),
                        "risk_profile": classify_risk(risk_score),
                        "feature_count": 7,
                    }
                    flash("Voice test processed successfully.", "success")
                except Exception as exc:
                    db.session.rollback()
                    flash(f"Unable to process the voice test: {exc}", "danger")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

    latest_voice_test = (
        PatientTest.query.filter_by(user_id=current_user.id, test_type="voice")
        .order_by(PatientTest.timestamp.desc(), PatientTest.id.desc())
        .first()
    )

    return render_template(
        "voice_test.html",
        result=result,
        latest_test=(
            serialize_test_record(latest_voice_test) if latest_voice_test else None
        ),
    )


@app.route("/test/spiral", methods=["GET", "POST"])
@login_required
def spiral_test():
    if request.method == "POST":
        flash("Spiral drawing analysis is not enabled yet.", "warning")

    return render_template("spiral_test.html")


@app.route("/history")
@login_required
def history():
    records = get_user_tests(current_user.id)
    serialized_records = [serialize_test_record(record) for record in records]
    chart_records = serialized_records
    latest_record = serialized_records[-1] if serialized_records else None

    return render_template(
        "history.html",
        records=list(reversed(serialized_records)),
        chart_records=chart_records,
        latest_test=latest_record,
    )


@app.route("/api/history")
@login_required
def api_history():
    records = get_user_tests(current_user.id)
    return jsonify(
        {
            "success": True,
            "history": [serialize_test_record(record) for record in records],
        }
    )


@app.route("/api/generate_report/", methods=["GET"])
@login_required
def generate_report():
    test_id = request.args.get("test_id", type=int)
    if test_id is None:
        abort(400, description="test_id is required.")

    test_record = PatientTest.query.filter_by(
        id=test_id, user_id=current_user.id
    ).first()
    if test_record is None:
        abort(404)

    pdf_buffer = build_clinical_report_pdf(current_user, test_record)
    filename = f"neuroscreen_clinical_report_{test_record.id}.pdf"
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
