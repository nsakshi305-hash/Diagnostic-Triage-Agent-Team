
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from agents.triage_agent import TriageAgent
from segmentation_agent_trained import SegmentationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reporting_agent import ReportingAgent
from agents.medical_classifier_agent import MedicalClassifierAgent


st.set_page_config(
    page_title="Diagnostic Triage Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
    }
    section[data-testid="stSidebar"] {
        background: #0b1830;
    }
    section[data-testid="stSidebar"] * {
        color: #eef4ff !important;
    }
    .brand {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .brand-sub {
        color: #aebbd0 !important;
        font-size: 13px;
        margin-bottom: 28px;
    }
    .nav-links {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin: 8px 0 18px 0;
    }
    .nav-links a {
        display: block;
        padding: 10px 12px;
        border-radius: 9px;
        color: #eef4ff !important;
        text-decoration: none !important;
        font-size: 15px;
        font-weight: 600;
        transition: background .15s ease, transform .15s ease;
    }
    .nav-links a:hover {
        background: rgba(255,255,255,.10);
        transform: translateX(2px);
    }
    .hero {
        background: linear-gradient(135deg, #102447 0%, #183763 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 32px;
    }
    .hero p {
        margin: 0;
        color: #c9d6ea;
        font-size: 15px;
    }
    .status {
        display: inline-block;
        background: #dff7e8;
        color: #187341;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 14px;
    }
    .section-title {
        color: #12223f;
        font-size: 21px;
        font-weight: 800;
        margin: 24px 0 12px 0;
    }
    .finding-card {
        background: white;
        border: 1px solid #e1e8f1;
        border-radius: 16px;
        padding: 22px;
        min-height: 210px;
    }
    .finding-label {
        color: #718096;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }
    .finding-name {
        color: #102447;
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0;
    }
    .explain {
        color: #4a5568;
        line-height: 1.55;
        font-size: 14px;
    }
    .disclaimer {
        background: #fff8e6;
        border-left: 4px solid #e3a72f;
        padding: 12px 14px;
        border-radius: 8px;
        color: #654f1d;
        font-size: 13px;
        margin-top: 12px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e1e8f1;
        border-radius: 14px;
        padding: 17px;
        text-align: center;
    }
    .metric-title {
        color: #718096;
        font-size: 12px;
        font-weight: 700;
    }
    .metric-value {
        color: #102447;
        font-size: 25px;
        font-weight: 800;
        margin-top: 4px;
    }
    .report-card {
        background: #ffffff !important;
        border: 1px solid #e1e8f1;
        border-radius: 16px;
        padding: 22px;
        color: #12223f !important;
    }
    .report-card * {
        color: #12223f !important;
    }
    .report-card .metric-title {
        color: #718096 !important;
    }
    .report-card .metric-value {
        color: #102447 !important;
    }
    .report-heading {
        color: #12223f !important;
        font-size: 22px;
        font-weight: 800;
    }
    .small-muted {
        color: #718096;
        font-size: 12px;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="brand">🏥 Diagnostic Triage Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">AI-assisted gastrointestinal endoscopy analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown("### Navigation")
    st.markdown("""
    <div class="nav-links">
        <a href="#input-image">🖼️&nbsp;&nbsp;Input Image</a>
        <a href="#ai-finding">🧠&nbsp;&nbsp;AI Finding</a>
        <a href="#segmentation">🎯&nbsp;&nbsp;Segmentation</a>
        <a href="#evaluation">📊&nbsp;&nbsp;Evaluation</a>
        <a href="#reporting">📋&nbsp;&nbsp;Reporting</a>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**System status**")
    st.success("Online")
    st.caption("Research / educational prototype")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>Diagnostic Triage Agent</h1>
    <p>AI-assisted analysis of gastrointestinal endoscopy images</p>
    <span class="status">● SYSTEM ONLINE</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a Kvasir-SEG image",
    type=["jpg", "jpeg", "png"],
    help="For segmentation evaluation, upload an image whose matching Kvasir-SEG mask exists."
)

if uploaded_file is None:
    st.markdown("""
    <div class="finding-card">
        <div class="finding-label">Getting started</div>
        <div class="finding-name">Upload an endoscopy image</div>
        <div class="explain">
            The system will perform image triage, AI finding classification,
            segmentation, objective evaluation, and automated reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------
# Load image
# -----------------------------
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -----------------------------
# Triage
# -----------------------------
triage_agent = TriageAgent()
triage_result = triage_agent.assess(image)

if triage_result["status"] == "rejected":
    st.error(triage_result["reason"])
    st.stop()

height, width = image.shape[:2]

st.markdown('<div id="input-image"></div><div class="section-title">1. Input Image & Triage</div>', unsafe_allow_html=True)

image_col, info_col = st.columns([1.45, 1])

with image_col:
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Endoscopy Image",
        width="stretch"
    )

with info_col:
    st.markdown("""
    <div class="finding-card">
        <div class="finding-label">Triage Status</div>
        <div class="finding-name" style="font-size:24px;">Accepted</div>
        <div class="explain">
            The image passed the basic input-quality checks and can proceed
            through the analysis pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Width", f"{width}px")
    with m2:
        st.metric("Height", f"{height}px")

# -----------------------------
# Temporary image
# -----------------------------
temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
cv2.imwrite(temp_image.name, image)
temp_image.close()

# -----------------------------
# Medical classifier
# -----------------------------
st.markdown('<div id="ai-finding"></div><div class="section-title">2. Medical AI Finding</div>', unsafe_allow_html=True)

@st.cache_resource
def load_medical_classifier():
    return MedicalClassifierAgent()

predictions = None
classifier_error = None

with st.spinner("Analyzing image with the medical AI classifier..."):
    try:
        classifier = load_medical_classifier()
        predictions = classifier.predict(temp_image.name)
    except Exception as exc:
        classifier_error = str(exc)

if predictions:
    top = predictions[0]
    raw_label = str(top["label"])
    confidence = float(top["confidence"]) * 100

    simple_names = {
        "polyps": "Polyp",
        "dyed-lifted-polyps": "Dyed / Lifted Polyp",
        "dyed-resection-margins": "Dyed Resection Margin",
        "esophagitis": "Esophagitis",
        "ulcerative-colitis": "Ulcerative Colitis",
        "normal-cecum": "Normal Cecum",
        "normal-pylorus": "Normal Stomach Outlet",
        "normal-z-line": "Normal Z-Line",
    }

    explanations = {
        "polyps": "A polyp is a growth that can develop on the inner lining of the digestive tract. Its nature cannot be determined from this AI result alone.",
        "dyed-lifted-polyps": "The image pattern is consistent with a dyed or lifted polyp appearance.",
        "dyed-resection-margins": "The image pattern is consistent with a dyed resection-margin appearance.",
        "esophagitis": "The image pattern is associated with inflammation of the esophagus.",
        "ulcerative-colitis": "The image pattern is associated with ulcerative-colitis-related inflammation.",
        "normal-cecum": "The model classified the image as a normal-appearing cecum pattern.",
        "normal-pylorus": "The model classified the image as a normal-appearing stomach-outlet pattern.",
        "normal-z-line": "The model classified the image as a normal-appearing Z-line pattern.",
    }

    display_name = simple_names.get(raw_label, raw_label.replace("-", " ").title())
    explanation = explanations.get(
        raw_label,
        "The AI identified an image pattern associated with this model class."
    )

    view = st.radio(
        "Result view",
        ["👤 Patient-Friendly View", "🔬 Scientific / Doctor View"],
        horizontal=True,
    )

    if view == "👤 Patient-Friendly View":
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-label">AI-Predicted Finding</div>
            <div class="finding-name">{display_name}</div>
            <div class="explain"><b>What does this mean?</b><br>{explanation}</div>
            <div class="disclaimer">
                This is an AI-predicted image finding, not a medical diagnosis.
                A qualified healthcare professional should review the image and
                determine the appropriate clinical interpretation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-label">Scientific / Doctor View</div>
            <div class="finding-name">{display_name}</div>
            <div class="explain">
                <b>Model class:</b> {raw_label}<br>
                <b>Top prediction confidence:</b> {confidence:.2f}%
            </div>
            <div class="disclaimer">
                Model output is intended for research and educational use and
                should not be interpreted as a clinical diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Top model predictions")
    for p in predictions:
        label = simple_names.get(
            str(p["label"]),
            str(p["label"]).replace("-", " ").title()
        )
        pct = float(p["confidence"]) * 100
        st.progress(min(max(float(p["confidence"]), 0.0), 1.0), text=f"{label} — {pct:.2f}%")

else:
    st.warning(f"Medical AI classification could not be generated: {classifier_error}")

# -----------------------------
# Segmentation
# -----------------------------
st.markdown('<div id="segmentation"></div><div class="section-title">3. Segmentation Agent</div>', unsafe_allow_html=True)

image_name = uploaded_file.name
mask_path = os.path.join("Kvasir-SEG", "Kvasir-SEG", "masks", image_name)

if not os.path.exists(mask_path):
    st.warning("A matching Kvasir-SEG ground-truth mask was not found for this image.")
    st.info("Upload an image directly from the Kvasir-SEG images folder for segmentation evaluation.")
    os.unlink(temp_image.name)
    st.stop()

segmentation_agent = SegmentationAgent()
result = segmentation_agent.segment(temp_image.name, mask_path)
predicted_mask = result["mask"]

colored_mask = np.zeros_like(image)
colored_mask[:, :, 2] = predicted_mask

overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

seg1, seg2 = st.columns(2)

with seg1:
    st.image(predicted_mask, caption="Predicted Mask", width="stretch")

with seg2:
    st.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="Segmentation Overlay",
        width="stretch"
    )

# -----------------------------
# Evaluation
# -----------------------------
st.markdown('<div id="evaluation"></div><div class="section-title">4. Evaluation Agent</div>', unsafe_allow_html=True)

ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
evaluation_agent = EvaluationAgent()

dice = evaluation_agent.dice_score(predicted_mask, ground_truth)
iou = evaluation_agent.iou_score(predicted_mask, ground_truth)

ec1, ec2, ec3 = st.columns(3)

with ec1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">DICE SCORE</div><div class="metric-value">{dice:.4f}</div></div>', unsafe_allow_html=True)

with ec2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">IoU SCORE</div><div class="metric-value">{iou:.4f}</div></div>', unsafe_allow_html=True)

with ec3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">IMAGE SIZE</div><div class="metric-value">{width} × {height}</div></div>', unsafe_allow_html=True)

st.caption(
    "Dice and IoU measure overlap between the predicted segmentation mask and the available ground-truth mask."
)

# -----------------------------
# Reporting
# -----------------------------
st.markdown('<div id="reporting"></div><div class="section-title">5. Reporting Agent</div>', unsafe_allow_html=True)

reporting_agent = ReportingAgent()
report = reporting_agent.generate_report(triage_result, dice, iou)
triage_status = report.get("triage_status", triage_result["status"])

confidence_text = f"{confidence:.2f}%" if predictions else "Unavailable"
finding_text = display_name if predictions else "Unavailable"

st.markdown(f"""
<div class="report-card">
    <div class="report-heading">Image Analysis Summary</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px;">
        <div>
            <div class="metric-title">TRIAGE STATUS</div>
            <div class="metric-value" style="font-size:20px;">{str(triage_status).title()}</div>
        </div>
        <div>
            <div class="metric-title">IMAGE SIZE</div>
            <div class="metric-value" style="font-size:20px;">{width} × {height}</div>
        </div>
        <div>
            <div class="metric-title">DICE SCORE</div>
            <div class="metric-value" style="font-size:20px;">{dice:.4f}</div>
        </div>
        <div>
            <div class="metric-title">IOU SCORE</div>
            <div class="metric-value" style="font-size:20px;">{iou:.4f}</div>
        </div>
    </div>
    <div style="margin-top:20px; padding-top:18px; border-top:1px solid #e1e8f1; color:#12223f; font-size:15px; line-height:1.8;">
        <b>AI-predicted finding:</b> {finding_text}<br>
        <b>Classifier confidence:</b> {confidence_text}
    </div>
    <div style="margin-top:18px; padding:14px 16px; border-radius:12px; background:#f8fafc; color:#334155; font-size:13px; line-height:1.7;">
        <b>Research &amp; educational use only.</b><br>
        This prototype provides AI-assisted image analysis and is not a medical diagnosis or a substitute for professional clinical evaluation.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Cleanup
# -----------------------------
try:
    os.unlink(temp_image.name)
except OSError:
    pass

