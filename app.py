import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from agents.triage_agent import TriageAgent
from agents.segmentation_agent import SegmentationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reporting_agent import ReportingAgent


st.set_page_config(
    page_title="Diagnostic Triage Agent Team",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diagnostic Triage Agent Team")
st.write(
    "AI-assisted medical image segmentation and evaluation "
    "for research and educational purposes."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a Kvasir-SEG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.subheader("1. Triage Agent")

    triage_agent = TriageAgent()
    triage_result = triage_agent.assess(image)

    if triage_result["status"] == "rejected":
        st.error(triage_result["reason"])
        st.stop()

    st.success(
        f"Image accepted — {triage_result['image_size']}"
    )

    # Display original
    st.subheader("2. Input Image")

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Medical Image",
        use_container_width=True
    )

    # Save temporary image
    temp_image = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    cv2.imwrite(
        temp_image.name,
        image
    )

    temp_image.close()

    st.subheader("3. Segmentation Agent")

    segmentation_agent = SegmentationAgent()

    # For current pipeline we use the dataset mask
    image_name = uploaded_file.name

    mask_path = os.path.join(
        "Kvasir-SEG",
        "Kvasir-SEG",
        "masks",
        image_name
    )

    if not os.path.exists(mask_path):
        st.warning(
            "Corresponding ground-truth mask was not found "
            "in the Kvasir-SEG dataset."
        )
        st.info(
            "Please upload an image directly from the "
            "Kvasir-SEG images folder."
        )
        os.unlink(temp_image.name)
        st.stop()

    result = segmentation_agent.segment(
        temp_image.name,
        mask_path
    )

    predicted_mask = result["mask"]

    # Create overlay
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 2] = predicted_mask

    overlay = cv2.addWeighted(
        image,
        0.7,
        colored_mask,
        0.3,
        0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            predicted_mask,
            caption="Segmentation Mask",
            use_container_width=True
        )

    with col2:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="Segmentation Overlay",
            use_container_width=True
        )

    st.subheader("4. Evaluation Agent")

    ground_truth = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    evaluation_agent = EvaluationAgent()

    dice = evaluation_agent.dice_score(
        predicted_mask,
        ground_truth
    )

    iou = evaluation_agent.iou_score(
        predicted_mask,
        ground_truth
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dice Score",
            f"{dice:.4f}"
        )

    with col2:
        st.metric(
            "IoU Score",
            f"{iou:.4f}"
        )

    st.subheader("5. Reporting Agent")

    reporting_agent = ReportingAgent()

    report = reporting_agent.generate_report(
        triage_result,
        dice,
        iou
    )

    for key, value in report.items():

        if key != "note":
            st.write(
                f"**{key.replace('_', ' ').title()}:** {value}"
            )

    st.info(report["note"])

    # Cleanup
    os.unlink(temp_image.name)