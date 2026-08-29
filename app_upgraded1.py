import base64
import io
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Forensic Glass Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATHS / CONSTANTS
# ============================================================
APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "glass_prediction_model.pkl"
SCALER_PATH = APP_DIR / "glass_scaler.pkl"
DATASET_PATH = APP_DIR / "glass.csv"
BACKGROUND_PATH = APP_DIR / "background.png"

FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

FEATURE_INFO = {
    "RI": ("Refractive Index", "Physical property measuring how strongly light bends through glass."),
    "Na": ("Sodium", "Chemical composition feature used by the trained model."),
    "Mg": ("Magnesium", "Chemical composition feature used by the trained model."),
    "Al": ("Aluminum", "Chemical composition feature used by the trained model."),
    "Si": ("Silicon", "Primary glass-forming composition feature."),
    "K": ("Potassium", "Chemical composition feature used by the trained model."),
    "Ca": ("Calcium", "Chemical composition feature used by the trained model."),
    "Ba": ("Barium", "Chemical composition feature used by the trained model."),
    "Fe": ("Iron", "Trace composition feature that can influence glass colour."),
}

GLASS_TYPES = {
    1: ("Building Windows", "Float Processed"),
    2: ("Building Windows", "Non-Float Processed"),
    3: ("Vehicle Windows", "Float Processed"),
    4: ("Vehicle Windows", "Non-Float Processed"),
    5: ("Containers", "Bottles / Jars"),
    6: ("Tableware", "Drinking Glasses / Dishes"),
    7: ("Headlamps", "Automotive / Heavy-Duty Lamps"),
}

# Results recorded in the supplied training notebook.
BENCHMARK_RESULTS = {
    "Random Forest": 83.72,
    "SVM": 72.09,
    "KNN": 69.77,
    "Gradient Boosting": 86.05,
    "Extra Trees": 81.40,
    "Logistic Regression": 72.09,
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def image_to_base64(path: Path):
    if not path.exists():
        return None
    try:
        suffix = path.suffix.lower().replace(".", "")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


def glass_type_name(glass_type):
    try:
        glass_type = int(glass_type)
    except Exception:
        return f"Type {glass_type}"

    if glass_type in GLASS_TYPES:
        family, process = GLASS_TYPES[glass_type]
        return f"Type {glass_type} · {family} ({process})"
    return f"Type {glass_type}"


def confidence_label(confidence):
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


@st.cache_resource(show_spinner=False)
def load_ml_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.name}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH.name}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH.name}")

    df = pd.read_csv(DATASET_PATH)

    rename_map = {}
    for col in df.columns:
        clean = str(col).strip()
        if clean.lower() == "id":
            rename_map[col] = "Id"
        elif clean.lower() in {"type of glass", "type"}:
            rename_map[col] = "Type of glass"
        else:
            rename_map[col] = clean

    return df.rename(columns=rename_map)


@st.cache_data(show_spinner=False)
def evaluate_saved_model():
    """
    Recreates the notebook's 80/20 split and evaluates the saved model.
    The supplied notebook scaled the full feature matrix before splitting,
    so this intentionally mirrors that workflow for comparability.
    """
    df = load_dataset()

    missing = [c for c in FEATURES + ["Type of glass"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    model, scaler = load_ml_assets()

    X = df[FEATURES].copy()
    y = df["Type of glass"].copy()

    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
    )

    y_pred = model.predict(X_test)

    labels = sorted(pd.Series(y_test).unique().tolist())

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return metrics, report, cm, labels, y_test, y_pred


def make_pdf_report(input_data, prediction, probabilities, model_name):
    """
    Generate a compact PDF report. Requires reportlab.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return None

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("FORENSIC GLASS ANALYSIS", styles["Title"]))
    story.append(Paragraph("Machine-Learning Classification Report", styles["Heading2"]))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}<br/>"
            f"<b>Model:</b> {model_name}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    family, process = GLASS_TYPES.get(
        int(prediction),
        ("Unknown category", "Dataset-specific label"),
    )

    story.append(Paragraph("Prediction", styles["Heading2"]))
    story.append(
        Paragraph(
            f"<b>Type {prediction}</b> — {family} ({process})",
            styles["BodyText"],
        )
    )

    if probabilities is not None:
        confidence = float(np.max(probabilities))
        story.append(
            Paragraph(
                f"<b>Highest model probability:</b> {confidence * 100:.2f}%",
                styles["BodyText"],
            )
        )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Sample Properties", styles["Heading2"]))

    table_data = [["Feature", "Value"]]
    for feature in FEATURES:
        table_data.append([feature, f"{float(input_data[feature].iloc[0]):.5f}"])

    table = Table(table_data, colWidths=[45 * mm, 45 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Interpretation & Limitation", styles["Heading2"]))
    story.append(
        Paragraph(
            "This report presents a machine-learning prediction based on the "
            "provided input values. It is an analytical aid for an educational "
            "forensic application and must not be treated as a standalone forensic "
            "conclusion or a substitute for validated laboratory examination and "
            "expert interpretation.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 4. CSS
# ============================================================
def inject_css():
    bg = image_to_base64(BACKGROUND_PATH)

    if bg:
        background_css = f"""
        .hero {{
            position: relative;
            border-radius: 26px;
            overflow: hidden;
            min-height: 475px;
            padding: 58px;
            display: flex;
            align-items: center;
            background:
                linear-gradient(90deg,
                    rgba(2,9,20,.98) 0%,
                    rgba(2,9,20,.90) 38%,
                    rgba(2,9,20,.50) 72%,
                    rgba(2,9,20,.18) 100%),
                url("{bg}") center / cover no-repeat;
            border: 1px solid rgba(255,255,255,.10);
            box-shadow: 0 25px 70px rgba(0,0,0,.20);
        }}
        """
    else:
        background_css = """
        .hero {
            position: relative;
            border-radius: 26px;
            min-height: 475px;
            padding: 58px;
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, #020914, #10253b);
            border: 1px solid rgba(255,255,255,.10);
        }
        """

    st.markdown(
        f"""
        <style>
        .stApp {{ background:#f6f8fc; }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(246,248,252,.92);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg,#06111f 0%,#0b1a2c 100%);
            border-right:1px solid rgba(255,255,255,.07);
        }}

        section[data-testid="stSidebar"] * {{ color:#e8eef7; }}

        .sidebar-brand {{
            padding: 8px 8px 20px 8px;
        }}

        .brand-icon {{
            width:44px;height:44px;
            display:inline-flex;
            align-items:center;justify-content:center;
            border-radius:14px;
            background:linear-gradient(135deg,#6d63ff,#8b5cf6);
            font-size:23px;
            box-shadow:0 10px 28px rgba(108,99,255,.28);
        }}

        .brand-title {{
            font-size:19px;
            font-weight:800;
            margin-top:11px;
        }}

        .brand-subtitle {{
            font-size:12px;
            color:#9fb0c8 !important;
            margin-top:2px;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:6px; }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            border-radius:11px;
            padding:9px 11px;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background:rgba(124,116,255,.13);
        }}

        .topbar {{
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:18px;
            padding:20px 2px;
        }}

        .topbar-title {{
            font-weight:800;
            font-size:15px;
            color:#172033;
        }}

        .topbar-sub {{
            font-size:12px;
            color:#7b8799;
            margin-top:2px;
        }}

        .status-dot {{
            width:7px;height:7px;
            border-radius:50%;
            background:#22c55e;
        }}

        {background_css}

        .hero-copy {{ max-width:700px; }}

        .hero-eyebrow {{
            display:inline-block;
            padding:7px 12px;
            border-radius:999px;
            color:#dbe7ff;
            background:rgba(99,102,241,.18);
            border:1px solid rgba(129,140,248,.30);
            font-size:11px;
            font-weight:800;
            letter-spacing:.8px;
        }}

        .hero-title {{
            color:white;
            font-size:clamp(38px,5vw,68px);
            line-height:.98;
            font-weight:900;
            letter-spacing:-2.2px;
            margin-top:19px;
        }}

        .hero-title span {{ color:#9b8cff; }}

        .hero-description {{
            color:#c6d1e1;
            font-size:16px;
            line-height:1.7;
            max-width:650px;
            margin-top:20px;
        }}

        .hero-note {{
            color:#8fa3bd;
            font-size:12px;
            margin-top:22px;
        }}

        .section-title {{
            font-size:25px;
            font-weight:850;
            color:#111827;
            margin:30px 0 13px;
        }}

        .metric-card {{
            background:white;
            border:1px solid #e5eaf1;
            border-radius:18px;
            padding:20px;
            min-height:145px;
            box-shadow:0 8px 30px rgba(15,23,42,.045);
        }}

        .metric-icon {{ font-size:23px; margin-bottom:8px; }}
        .metric-value {{ font-size:23px;font-weight:850;color:#111827; }}
        .metric-label {{ font-size:13px;color:#64748b;line-height:1.45;margin-top:4px; }}

        .info-card {{
            background:white;
            border:1px solid #e5eaf1;
            border-radius:18px;
            padding:20px;
            box-shadow:0 7px 25px rgba(15,23,42,.035);
        }}

        .small-note {{
            color:#64748b;
            font-size:12px;
            line-height:1.55;
        }}

        .result-card {{
            background:linear-gradient(135deg,#ecfdf5,#f0fdf4);
            border:1px solid #bbf7d0;
            border-radius:22px;
            padding:25px;
        }}

        .result-label {{
            color:#15803d;
            font-size:11px;
            font-weight:800;
            letter-spacing:1.2px;
            text-transform:uppercase;
        }}

        .result-type {{
            color:#166534;
            font-size:40px;
            font-weight:900;
            margin:5px 0;
        }}

        .result-description {{ color:#3f6212;font-size:14px; }}

        .confidence-card {{
            background:white;
            border:1px solid #e5eaf1;
            border-radius:18px;
            padding:20px;
        }}

        .confidence-value {{
            font-size:32px;
            font-weight:900;
            color:#111827;
        }}

        .confidence-label {{
            font-size:12px;
            color:#64748b;
            margin-top:2px;
        }}

        .page-kicker {{
            color:#6d63ff;
            text-transform:uppercase;
            letter-spacing:1.8px;
            font-size:11px;
            font-weight:850;
            margin-bottom:4px;
        }}

        .muted {{ color:#64748b; }}

        .stButton > button {{
            border-radius:11px;
            min-height:43px;
            font-weight:750;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:18px; }}

        div[data-testid="stDataFrame"] {{
            border-radius:14px;
            overflow:hidden;
        }}

        .footer {{
            text-align:center;
            color:#94a3b8;
            font-size:11px;
            padding-top:30px;
        }}

        @media print {{
            section[data-testid="stSidebar"],
            header[data-testid="stHeader"],
            .stButton,
            [data-testid="stDownloadButton"] {{
                display:none !important;
            }}

            .block-container {{
                padding-top:0 !important;
                max-width:100% !important;
            }}

            .hero {{
                min-height:360px !important;
                -webkit-print-color-adjust:exact;
                print-color-adjust:exact;
            }}

            .metric-card,.info-card,.result-card,.confidence-card {{
                box-shadow:none !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. SIDEBAR
# ============================================================
def sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">🔬</div>
                <div class="brand-title">Forensic Glass</div>
                <div class="brand-subtitle">Analysis Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🔬 Analyze Sample",
                "📊 Dataset Explorer",
                "🤖 Model Performance",
                "📚 Glass Reference",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

    return page


# ============================================================
# 6. TOP HEADER
# ============================================================
def top_header():
    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="topbar-title">Forensic Glass Analysis</div>
                <div class="topbar-sub">Physical & chemical evidence classification</div>
            </div>
            
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. DASHBOARD
# ============================================================
def dashboard_page():
    try:
        df = load_dataset()
        row_count = len(df)
        type_count = df["Type of glass"].nunique()
    except Exception:
        df = None
        row_count = 214
        type_count = 6

    st.markdown(
        """
        <div class="hero">
            <div class="hero-copy">
                <div class="hero-eyebrow">FORENSIC · MACHINE LEARNING · GLASS ANALYSIS</div>
                <div class="hero-title">
                    Forensic Glass<br><span>Analysis</span>
                </div>
                <div class="hero-description">
                    Classify a glass sample using its refractive index and
                    chemical composition through a trained machine-learning model.
                </div>
                <div class="hero-note">
                    Academic forensic analytics dashboard · Not a standalone forensic conclusion.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("🧪", "9", "Input Features", "RI + 8 chemical properties"),
        ("📊", str(row_count), "Dataset Samples", "Forensic glass observations"),
        ("🏷️", str(type_count), "Observed Classes", "Classes present in dataset"),
        ("⚡", "Real-time", "Prediction", "Fast model inference"),
    ]

    for col, item in zip([c1, c2, c3, c4], cards):
        with col:
            icon, value, label, desc = item
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label"><b>{label}</b><br>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">How the analysis works</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("01", "Measure", "Obtain RI and chemical composition values."),
        ("02", "Enter", "Enter the laboratory/sample measurements."),
        ("03", "Predict", "Saved scaler + ML model generate a class prediction."),
        ("04", "Interpret", "Review probabilities and model diagnostics."),
    ]

    for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="min-height:155px;">
                    <div style="font-size:12px;font-weight:900;color:#6d63ff;">{num}</div>
                    <div style="font-size:18px;font-weight:850;margin-top:7px;">{title}</div>
                    <div class="small-note" style="margin-top:7px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Project snapshot</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            <div class="info-card">
                <b>Input variables</b>
                <div class="small-note" style="margin-top:8px;">
                    Refractive Index (RI), Sodium (Na), Magnesium (Mg),
                    Aluminum (Al), Silicon (Si), Potassium (K), Calcium (Ca),
                    Barium (Ba) and Iron (Fe).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="info-card">
                <b>Selected model</b>
                <div style="font-size:20px;font-weight:850;margin-top:7px;">
                    Gradient Boosting
                </div>
                <div class="small-note" style="margin-top:5px;">
                    Notebook benchmark accuracy: <b>{BENCHMARK_RESULTS["Gradient Boosting"]:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Quick start</div>', unsafe_allow_html=True)

    if st.button("🔬 Start Glass Analysis", type="primary", use_container_width=True):
        st.session_state["navigate_to"] = "🔬 Analyze Sample"
        st.rerun()


# ============================================================
# 8. ANALYZE SAMPLE
# ============================================================
def analyze_page():
    st.markdown('<div class="page-kicker">Machine Learning</div>', unsafe_allow_html=True)
    st.title("Analyze Glass Sample")
    st.markdown(
        '<span class="muted">Enter the measured properties of a glass sample and generate a model-based classification.</span>',
        unsafe_allow_html=True,
    )

    try:
        model, scaler = load_ml_assets()
    except Exception as exc:
        st.error(f"ML assets could not be loaded: {exc}")
        st.info("Keep glass_prediction_model.pkl and glass_scaler.pkl beside app.py.")
        return

    with st.container(border=True):
        st.markdown("### Glass Sample Properties")
        st.caption("Chemical values are entered as weight percentages.")

        c1, c2, c3 = st.columns(3)

        with c1:
            ri = st.number_input(
                "Refractive Index (RI)",
                min_value=1.0,
                max_value=2.0,
                value=1.51800,
                step=0.00001,
                format="%.5f",
                help=FEATURE_INFO["RI"][1],
            )
            na = st.number_input(
                "Sodium (Na %)",
                min_value=0.0,
                value=13.30,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Na"][1],
            )
            mg = st.number_input(
                "Magnesium (Mg %)",
                min_value=0.0,
                value=3.40,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Mg"][1],
            )

        with c2:
            al = st.number_input(
                "Aluminum (Al %)",
                min_value=0.0,
                value=1.30,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Al"][1],
            )
            si = st.number_input(
                "Silicon (Si %)",
                min_value=0.0,
                value=72.70,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Si"][1],
            )
            k = st.number_input(
                "Potassium (K %)",
                min_value=0.0,
                value=0.50,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["K"][1],
            )

        with c3:
            ca = st.number_input(
                "Calcium (Ca %)",
                min_value=0.0,
                value=8.90,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Ca"][1],
            )
            ba = st.number_input(
                "Barium (Ba %)",
                min_value=0.0,
                value=0.00,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Ba"][1],
            )
            fe = st.number_input(
                "Iron (Fe %)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                format="%.2f",
                help=FEATURE_INFO["Fe"][1],
            )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        predict_btn = st.button(
            "🔬 Analyze Sample",
            type="primary",
            use_container_width=True,
        )

    if not predict_btn:
        st.markdown(
            """
            <div class="small-note" style="margin-top:15px;">
                Tip: use laboratory-measured values for meaningful model output.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    input_data = pd.DataFrame(
        [[ri, na, mg, al, si, k, ca, ba, fe]],
        columns=FEATURES,
    )

    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    family, process = GLASS_TYPES.get(
        int(prediction),
        ("Unknown category", "Dataset-specific label"),
    )

    confidence = float(np.max(probabilities)) if probabilities is not None else None

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("### · Analysis Result")

    left, right = st.columns([1.35, 0.65])

    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Most Probable Classification</div>
                <div class="result-type">Type {prediction}</div>
                <div class="result-description">
                    <b>{family}</b> · {process}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if confidence is not None:
            st.markdown(
                f"""
                <div class="confidence-card">
                    <div class="small-note">Highest model probability</div>
                    <div class="confidence-value">{confidence * 100:.2f}%</div>
                    <div class="confidence-label">
                        {confidence_label(confidence)} confidence indicator
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="confidence-card">
                    <div class="confidence-value">N/A</div>
                    <div class="confidence-label">
                        This saved model does not expose predict_proba().
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    history_item = {
        "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Prediction": f"Type {prediction}",
        "Class": family,
        "Probability": f"{confidence * 100:.2f}%" if confidence is not None else "N/A",
    }

    # Avoid adding the same result repeatedly during a single rerun.
    if not st.session_state.prediction_history or st.session_state.prediction_history[-1] != history_item:
        st.session_state.prediction_history.append(history_item)

    if probabilities is not None:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        p1, p2 = st.columns([1.25, 1])

        with p1:
            st.markdown("#### Prediction Probabilities")

            classes = getattr(
                model,
                "classes_",
                np.arange(1, len(probabilities) + 1),
            )

            prob_df = pd.DataFrame(
                {
                    "Glass Type": [f"Type {c}" for c in classes],
                    "Probability": probabilities,
                }
            )
            prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)

            chart_df = prob_df[["Glass Type", "Probability"]].set_index("Glass Type")
            st.bar_chart(chart_df, y="Probability", height=330)

        with p2:
            st.markdown("#### Input Summary")
            summary = input_data.T.reset_index()
            summary.columns = ["Feature", "Value"]
            summary["Value"] = summary["Value"].round(5)
            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
                height=330,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("#### Interpretation")
    st.info(
        f"The model classified this sample as Type {prediction}: "
        f"{family} ({process}). "
        "The probability shown above is the model's class probability, not a "
        "measure of forensic certainty."
    )

    pdf_data = make_pdf_report(
        input_data,
        prediction,
        probabilities,
        type(model).__name__,
    )

    if pdf_data:
        st.download_button(
            "📄 Download Analysis Report (PDF)",
            data=pdf_data,
            file_name="forensic_glass_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.caption(
        "Important: this machine-learning prediction is an analytical aid and "
        "should not be treated as a standalone forensic conclusion."
    )


# ============================================================
# 9. DATASET EXPLORER
# ============================================================
def dataset_page():
    st.markdown('<div class="page-kicker">Data Exploration</div>', unsafe_allow_html=True)
    st.title("Dataset Explorer")
    st.markdown(
        '<span class="muted">Inspect, filter and understand the glass dataset.</span>',
        unsafe_allow_html=True,
    )

    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")
        return

    display_df = df.copy()

    if "Id" in display_df.columns:
        display_df = display_df.drop(columns=["Id"])

    target_col = "Type of glass"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", display_df.shape[0])
    c2.metric("Columns", display_df.shape[1])
    c3.metric(
        "Observed Classes",
        display_df[target_col].nunique() if target_col in display_df else "—",
    )
    c4.metric(
        "ML Features",
        display_df.shape[1] - 1 if target_col in display_df else display_df.shape[1],
    )

    st.markdown("---")
    st.markdown("### Filters")

    f1, f2, f3 = st.columns([1, 1, 1])

    with f1:
        if target_col in display_df.columns:
            type_options = ["All Types"] + sorted(
                display_df[target_col].dropna().unique().tolist()
            )
            selected_type = st.selectbox("Glass Type", type_options)
        else:
            selected_type = "All Types"

    with f2:
        search_text = st.text_input(
            "Search values",
            placeholder="e.g. 72.72",
        )

    with f3:
        columns = display_df.columns.tolist()
        selected_columns = st.multiselect(
            "Columns to display",
            columns,
            default=columns,
        )

    filtered = display_df.copy()

    if selected_type != "All Types" and target_col in filtered.columns:
        filtered = filtered[filtered[target_col] == selected_type]

    if search_text.strip():
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search_text.strip(),
                case=False,
                na=False,
            )
        ).any(axis=1)
        filtered = filtered[mask]

    if not selected_columns:
        st.warning("Select at least one column.")
        return

    filtered = filtered[selected_columns]

    st.markdown(f"**Showing {len(filtered)} of {len(display_df)} rows**")
    st.dataframe(filtered, use_container_width=True, height=410)

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="filtered_glass_dataset.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">Dataset Analytics</div>', unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown("#### Class Distribution")
        if target_col in df.columns:
            distribution = (
                df[target_col]
                .value_counts()
                .sort_index()
                .rename_axis("Glass Type")
                .to_frame("Samples")
            )
            st.bar_chart(distribution, y="Samples", height=350)

    with b:
        st.markdown("#### Descriptive Statistics")
        numeric_cols = [c for c in FEATURES if c in df.columns]
        st.dataframe(
            df[numeric_cols].describe().T.round(3),
            use_container_width=True,
            height=350,
        )

    st.markdown("#### Feature Correlation")

    numeric_cols = [c for c in FEATURES if c in df.columns]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        st.dataframe(corr, use_container_width=True)

        feature_choice = st.selectbox(
            "Select feature for distribution",
            numeric_cols,
        )
        st.line_chart(
            df[feature_choice].sort_values().reset_index(drop=True),
            height=280,
        )


# ============================================================
# 10. MODEL PERFORMANCE
# ============================================================
def performance_page():
    st.markdown('<div class="page-kicker">Model Diagnostics</div>', unsafe_allow_html=True)
    st.title("Model Performance")
    st.markdown(
        '<span class="muted">Performance and diagnostics for the saved Gradient Boosting model.</span>',
        unsafe_allow_html=True,
    )

    try:
        model, scaler = load_ml_assets()
        metrics, report, cm, labels, y_test, y_pred = evaluate_saved_model()
    except Exception as exc:
        st.error(f"Model evaluation could not be completed: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    m4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")

    st.caption(
        "Evaluation recreates the 80/20 split with random_state=42 used in the "
        "supplied training notebook. The notebook scaled the complete feature "
        "matrix before splitting; this page mirrors that workflow for comparability."
    )

    st.markdown("---")

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### Confusion Matrix")

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_title("Saved Model — Test Set")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("### Classification Report")

        report_rows = []
        for label in labels:
            key = str(label)
            if key in report:
                row = report[key]
                report_rows.append(
                    {
                        "Class": f"Type {label}",
                        "Precision": round(row["precision"] * 100, 2),
                        "Recall": round(row["recall"] * 100, 2),
                        "F1": round(row["f1-score"] * 100, 2),
                        "Support": int(row["support"]),
                    }
                )

        st.dataframe(
            pd.DataFrame(report_rows),
            hide_index=True,
            use_container_width=True,
            height=300,
        )

        st.markdown("### Model Information")
        st.markdown(
            f"""
            <div class="info-card">
                <b>Loaded model</b>
                <div class="small-note" style="margin-top:7px;">
                    {type(model).__name__}
                </div>
                <br>
                <b>Scaler</b>
                <div class="small-note" style="margin-top:7px;">
                    {type(scaler).__name__}
                </div>
                <br>
                <b>Notebook benchmark</b>
                <div class="small-note" style="margin-top:7px;">
                    Gradient Boosting: {BENCHMARK_RESULTS["Gradient Boosting"]:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model Benchmark</div>', unsafe_allow_html=True)

    benchmark_df = (
        pd.DataFrame(
            [
                {"Model": model_name, "Accuracy (%)": accuracy}
                for model_name, accuracy in BENCHMARK_RESULTS.items()
            ]
        )
        .sort_values("Accuracy (%)", ascending=False)
        .set_index("Model")
    )

    st.bar_chart(benchmark_df, y="Accuracy (%)", height=360)

    st.caption(
        "Benchmark values are taken from the supplied training notebook. "
        "They are presented as the recorded experiment results, not as a newly "
        "retrained comparison in this application."
    )

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURES[: len(importance)],
                "Importance": importance,
            }
        ).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.set_index("Feature"),
            y="Importance",
            height=360,
        )
    else:
        st.info("This model does not expose feature_importances_.")


# ============================================================
# 11. GLASS REFERENCE
# ============================================================
def reference_page():
    st.markdown('<div class="page-kicker">Reference</div>', unsafe_allow_html=True)
    st.title("Glass Type Reference")
    st.markdown(
        '<span class="muted">Reference labels used by the application.</span>',
        unsafe_allow_html=True,
    )

    st.info(
        "The dataset contains observed classes 1, 2, 3, 5, 6 and 7. "
        "Type 4 is retained in the reference dictionary for label completeness "
        "but may be absent from the original dataset."
    )

    cols = st.columns(3)

    icons = {
        1: "🪟",
        2: "🪟",
        3: "🚗",
        4: "🚗",
        5: "🫙",
        6: "🥛",
        7: "💡",
    }

    for i, (type_id, (family, process)) in enumerate(GLASS_TYPES.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="info-card" style="margin-bottom:14px;min-height:170px;">
                    <div style="font-size:25px;">{icons[type_id]}</div>
                    <div style="color:#6d63ff;font-size:11px;font-weight:850;margin-top:9px;">
                        GLASS TYPE {type_id}
                    </div>
                    <div style="font-size:18px;font-weight:850;margin-top:6px;">
                        {family}
                    </div>
                    <div class="small-note" style="margin-top:6px;">
                        {process}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 12. ABOUT
# ============================================================
def about_page():
    st.markdown('<div class="page-kicker">Project Information</div>', unsafe_allow_html=True)
    st.title("About the Project")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
            ### Forensic Glass Analysis

            This application demonstrates machine-learning classification of
            forensic glass samples using refractive index and elemental
            composition features.

            **Workflow**

            Measure → Enter → Scale → Predict → Interpret

            The application loads a saved scaler and saved machine-learning model,
            then applies the same feature order during prediction.
            """
        )

        st.markdown("### Technology Stack")

        tech = pd.DataFrame(
            {
                "Layer": [
                    "Frontend / UI",
                    "Data Processing",
                    "Machine Learning",
                    "Model Storage",
                    "Visualization",
                    "Report",
                ],
                "Technology": [
                    "Streamlit",
                    "Pandas + NumPy",
                    "Scikit-learn compatible model",
                    "Joblib",
                    "Streamlit + Matplotlib",
                    "ReportLab",
                ],
            }
        )

        st.dataframe(tech, hide_index=True, use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="info-card">
                <div style="font-size:30px;">🔬</div>
                <h3>Forensic Context</h3>
                <div class="small-note">
                    Glass fragments can contain physical and chemical
                    characteristics that may be useful for comparison.
                </div>
            </div>

            <div style="height:14px;"></div>

            <div class="info-card">
                <div style="font-size:30px;">⚠️</div>
                <h3>Important Limitation</h3>
                <div class="small-note">
                    This is an educational machine-learning application.
                    Predictions should not replace laboratory examination,
                    validated forensic methods or expert interpretation.
                </div>
            </div>

            <div style="height:14px;"></div>

            <div class="info-card">
                <div style="font-size:30px;">📌</div>
                <h3>Model</h3>
                <div class="small-note">
                    The supplied project saves a Gradient Boosting model and
                    StandardScaler using Joblib.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 13. HISTORY
# ============================================================
def history_page_section():
    if "prediction_history" not in st.session_state:
        return

    if not st.session_state.prediction_history:
        return

    st.markdown('<div class="section-title">Recent Analyses</div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.prediction_history[::-1])

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True,
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()


# ============================================================
# 14. MAIN
# ============================================================
inject_css()

if "navigate_to" in st.session_state:
    default_page = st.session_state.pop("navigate_to")
else:
    default_page = None

page = sidebar()

# Dashboard button navigation is handled by selecting the matching page.
if default_page and default_page != page:
    page = default_page

top_header()

if page == "🏠 Dashboard":
    dashboard_page()

elif page == "🔬 Analyze Sample":
    analyze_page()
    history_page_section()

elif page == "📊 Dataset Explorer":
    dataset_page()

elif page == "🤖 Model Performance":
    performance_page()

elif page == "📚 Glass Reference":
    reference_page()

elif page == "ℹ️ About":
    about_page()

