import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import re
import io
import pytesseract
import os
import urllib.request


# ============================================================
# SYSTEM DEPENDENCIES & TESSERACT CONFIG
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Border Control & Document Integrity System",
    page_icon="🛂",
    layout="wide"
)

st.title("🛂 AI Border Control & Document Integrity System")
st.caption(
    "Automated OCR, Biometric & Forensic Document Verification Engine"
)


# ============================================================
# FACE MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

YUNET_PATH = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)

SFACE_PATH = os.path.join(
    MODEL_DIR,
    "face_recognition_sface_2021dec.onnx"
)

YUNET_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/"
    "resolve/main/face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/"
    "resolve/main/face_recognition_sface_2021dec.onnx"
)


def download_model(url, path):

    if os.path.exists(path):

        if os.path.getsize(path) > 10000:
            return True

    try:

        urllib.request.urlretrieve(
            url,
            path
        )

        return (
            os.path.exists(path)
            and os.path.getsize(path) > 10000
        )

    except Exception:

        return False


@st.cache_resource
def load_face_engine():

    try:

        if not hasattr(
            cv2,
            "FaceDetectorYN"
        ):
            return (
                None,
                None,
                "YuNet is not available in this OpenCV installation."
            )

        if not hasattr(
            cv2,
            "FaceRecognizerSF"
        ):
            return (
                None,
                None,
                "SFace is not available in this OpenCV installation."
            )

        # Download YuNet
        if not download_model(
            YUNET_URL,
            YUNET_PATH
        ):

            return (
                None,
                None,
                "Unable to download YuNet face detection model."
            )

        # Download SFace
        if not download_model(
            SFACE_URL,
            SFACE_PATH
        ):

            return (
                None,
                None,
                "Unable to download SFace recognition model."
            )

        # Create YuNet detector
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320),
            0.60,
            0.30,
            5000
        )

        # Create SFace recognizer
        recognizer = cv2.FaceRecognizerSF.create(
            SFACE_PATH,
            ""
        )

        return (
            detector,
            recognizer,
            None
        )

    except Exception as e:

        return (
            None,
            None,
            str(e)
        )


face_detector, face_recognizer, face_engine_error = (
    load_face_engine()
)


# ============================================================
# IMAGE CONVERSION HELPERS
# ============================================================

def pil_to_cv(image):

    return cv2.cvtColor(
        np.array(
            image.convert("RGB")
        ),
        cv2.COLOR_RGB2BGR
    )


def cv_to_pil(image):

    return Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )


# ============================================================
# OCR ENGINE
# ============================================================

@st.cache_data
def perform_real_ocr(image):

    try:

        text = pytesseract.image_to_string(
            image,
            config="--oem 3 --psm 6"
        )

        if text.strip():

            return text.strip()

        return (
            "No readable text detected "
            "in document image."
        )

    except Exception as e:

        return (
            f"OCR Engine Offline: {str(e)}"
        )


# ============================================================
# MRZ OCR
# ============================================================

def perform_mrz_ocr(image):

    try:

        img = pil_to_cv(image)

        height, width = img.shape[:2]

        # MRZ is normally located at the bottom
        mrz = img[
            int(height * 0.55):height,
            0:width
        ]

        gray = cv2.cvtColor(
            mrz,
            cv2.COLOR_BGR2GRAY
        )

        # Enlarge MRZ
        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        # Improve contrast
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

        text = pytesseract.image_to_string(
            binary,
            config=(
                "--oem 3 --psm 6 "
                "-c "
                "tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
            )
        )

        return text.strip()

    except Exception as e:

        return (
            f"MRZ OCR Error: {str(e)}"
        )


# ============================================================
# MRZ PARSER
# ============================================================

def parse_mrz_fallback(text_string):

    lines = [
        line.strip().upper()
        for line in text_string.split("\n")
        if len(line.strip()) > 10
    ]

    clean_lines = []

    for line in lines:

        clean = re.sub(
            r"[^A-Z0-9<]",
            "",
            line
        )

        if len(clean) >= 30:

            clean_lines.append(
                clean
            )

    # Prefer TD3 passport MRZ
    long_lines = [
        line
        for line in clean_lines
        if len(line) >= 40
    ]

    if len(long_lines) >= 2:

        line1 = (
            long_lines[-2][:44]
            .ljust(44, "<")
        )

        line2 = (
            long_lines[-1][:44]
            .ljust(44, "<")
        )

        valid = (
            line1.startswith("P<")
            and len(line1) == 44
            and len(line2) == 44
        )

        return (
            [line1, line2],
            valid
        )

    if len(clean_lines) >= 2:

        return (
            clean_lines[:2],
            False
        )

    return (
        clean_lines,
        False
    )


# ============================================================
# MRZ FIELD EXTRACTION
# ============================================================

def extract_mrz_fields(mrz_lines):

    if len(mrz_lines) != 2:

        return {}

    line1 = mrz_lines[0].ljust(
        44,
        "<"
    )

    line2 = mrz_lines[1].ljust(
        44,
        "<"
    )

    try:

        name_part = (
            line1[5:44]
            .replace("<", " ")
            .strip()
        )

        names = name_part.split()

        surname = (
            names[0]
            if names
            else ""
        )

        given_names = (
            " ".join(names[1:])
        )

        return {

            "Surname": surname,

            "Given Names": given_names,

            "Passport Number":
                line2[0:9].replace(
                    "<",
                    ""
                ),

            "Nationality":
                line2[10:13],

            "Date of Birth":
                line2[13:19],

            "Sex":
                line2[20],

            "Expiry Date":
                line2[21:27]
        }

    except Exception:

        return {}


# ============================================================
# ELA FORENSICS
# ============================================================

def perform_ela(
    image,
    quality=90
):

    temp_buffer = io.BytesIO()

    image.convert("RGB").save(
        temp_buffer,
        format="JPEG",
        quality=quality
    )

    temp_buffer.seek(0)

    resaved_img = Image.open(
        temp_buffer
    ).convert("RGB")

    ela_img = ImageChops.difference(
        image.convert("RGB"),
        resaved_img
    )

    extrema = ela_img.getextrema()

    max_diff = max(
        ex[1]
        for ex in extrema
    )

    if max_diff == 0:

        max_diff = 1

    scale = 255.0 / max_diff

    ela_img = ImageEnhance.Brightness(
        ela_img
    ).enhance(
        scale * 2.0
    )

    ela_np = np.array(
        ela_img
    )

    mean_diff = np.mean(
        ela_np
    )

    tamper_score = min(
        100.0,
        (mean_diff / 255.0) * 500
    )

    return (
        ela_img,
        round(
            tamper_score,
            2
        )
    )


# ============================================================
# YUNET FACE DETECTION
# ============================================================

def detect_faces(image):

    if face_detector is None:

        return []

    try:

        img = pil_to_cv(
            image
        )

        h, w = img.shape[:2]

        face_detector.setInputSize(
            (w, h)
        )

        _, faces = face_detector.detect(
            img
        )

        if faces is None:

            return []

        return list(faces)

    except Exception:

        return []


# ============================================================
# SELECT LARGEST FACE
# ============================================================

def get_largest_face(image):

    faces = detect_faces(
        image
    )

    if not faces:

        return None, []

    largest = max(
        faces,
        key=lambda face:
        float(face[2] * face[3])
    )

    return (
        largest,
        faces
    )


# ============================================================
# CROP DETECTED FACE
# ============================================================

def crop_face(
    image,
    face
):

    if face is None:

        return None

    img = pil_to_cv(
        image
    )

    x, y, w, h = (
        face[:4]
        .astype(int)
    )

    x = max(
        0,
        x
    )

    y = max(
        0,
        y
    )

    w = min(
        w,
        img.shape[1] - x
    )

    h = min(
        h,
        img.shape[0] - y
    )

    margin_x = int(
        w * 0.15
    )

    margin_y = int(
        h * 0.20
    )

    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        img.shape[1],
        x + w + margin_x
    )

    y2 = min(
        img.shape[0],
        y + h + margin_y
    )

    face_crop = img[
        y1:y2,
        x1:x2
    ]

    return cv_to_pil(
        face_crop
    )


# ============================================================
# REAL FACE COMPARISON
# ============================================================

def extract_and_compare_faces(
    doc_img,
    passenger_img
):

    try:

        if face_detector is None:

            return (
                None,
                None,
                None,
                "Face detector is unavailable."
            )

        if face_recognizer is None:

            return (
                None,
                None,
                None,
                "SFace recognizer is unavailable."
            )

        # ----------------------------------------
        # Detect passport face
        # ----------------------------------------

        doc_face, doc_faces = (
            get_largest_face(
                doc_img
            )
        )

        # ----------------------------------------
        # Detect live face
        # ----------------------------------------

        passenger_face, passenger_faces = (
            get_largest_face(
                passenger_img
            )
        )

        if doc_face is None:

            return (
                None,
                None,
                None,
                "No face detected in passport image."
            )

        if passenger_face is None:

            return (
                None,
                None,
                None,
                "No face detected in live photo."
            )

        # ----------------------------------------
        # Convert to OpenCV
        # ----------------------------------------

        doc_cv = pil_to_cv(
            doc_img
        )

        passenger_cv = pil_to_cv(
            passenger_img
        )

        # ----------------------------------------
        # Align faces
        # ----------------------------------------

        aligned_doc = (
            face_recognizer.alignCrop(
                doc_cv,
                doc_face
            )
        )

        aligned_passenger = (
            face_recognizer.alignCrop(
                passenger_cv,
                passenger_face
            )
        )

        # ----------------------------------------
        # Extract SFace features
        # ----------------------------------------

        feature_doc = (
            face_recognizer.feature(
                aligned_doc
            )
        )

        feature_passenger = (
            face_recognizer.feature(
                aligned_passenger
            )
        )

        # ----------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------

        cosine_similarity = float(

            face_recognizer.match(

                feature_doc,

                feature_passenger,

                cv2.FaceRecognizerSF_FR_COSINE
            )
        )

        # ----------------------------------------
        # SFace reference threshold
        # ----------------------------------------

        threshold = 0.363

        matched = (
            cosine_similarity >= threshold
        )

        # ----------------------------------------
        # Display score
        # ----------------------------------------

        display_score = (

            (cosine_similarity - 0.20)
            / 0.40
            * 100
        )

        display_score = max(
            0.0,
            min(
                100.0,
                display_score
            )
        )

        result = {

            "cosine":
                cosine_similarity,

            "threshold":
                threshold,

            "matched":
                matched,

            "doc_faces":
                len(doc_faces),

            "passenger_faces":
                len(passenger_faces)
        }

        return (

            round(
                display_score,
                2
            ),

            crop_face(
                doc_img,
                doc_face
            ),

            crop_face(
                passenger_img,
                passenger_face
            ),

            result
        )

    except Exception as e:

        return (

            None,

            None,

            None,

            f"Biometric Error: {str(e)}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🕹️ Control Panel"
)

uploaded_doc = (
    st.sidebar.file_uploader(
        "Upload Passport / Document",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )
)

live_photo = (
    st.sidebar.camera_input(
        "Capture Live Passenger Photo"
    )
)

st.sidebar.divider()


# OCR status

if os.path.exists(
    TESSERACT_PATH
):

    st.sidebar.success(
        "🟢 Tesseract OCR Ready"
    )

else:

    st.sidebar.error(
        "🔴 Tesseract OCR Not Found"
    )


# Face status

if face_recognizer is not None:

    st.sidebar.success(
        "🟢 YuNet + SFace Ready"
    )

else:

    st.sidebar.error(
        "🔴 Face Engine Not Ready"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_doc is None:

    st.info(
        "👈 Upload a passport image "
        "from the sidebar to begin."
    )

    st.stop()


doc_image = Image.open(
    uploaded_doc
).convert("RGB")


# ============================================================
# TABS
# ============================================================

t1, t2, t3, t4, t5 = st.tabs(

    [

        "📄 OCR Text",

        "📏 Validation",

        "🔍 ELA Forensics",

        "👤 Biometric Match",

        "📊 Risk Matrix"
    ]
)


# ============================================================
# TAB 1 — OCR
# ============================================================

with t1:

    st.header(
        "📄 Document Data Extraction"
    )

    col_img, col_txt = (
        st.columns([1, 1])
    )

    with col_img:

        st.image(
            doc_image,
            caption="Uploaded Document",
            use_container_width=True
        )

    with col_txt:

        st.subheader(
            "Live OCR & MRZ"
        )

        if not os.path.exists(
            TESSERACT_PATH
        ):

            st.error(
                "Tesseract executable was "
                "not found."
            )

            st.code(
                TESSERACT_PATH
            )

        else:

            with st.spinner(
                "Running OCR..."
            ):

                extracted_text = (
                    perform_real_ocr(
                        doc_image
                    )
                )

                mrz_text = (
                    perform_mrz_ocr(
                        doc_image
                    )
                )

            raw_text = (
                st.text_area(
                    "Live OCR Stream Output",
                    value=extracted_text,
                    height=200
                )
            )

            st.subheader(
                "🔐 MRZ OCR Output"
            )

            st.code(
                mrz_text
                if mrz_text
                else
                "No MRZ detected."
            )

            mrz_lines, is_mrz_valid = (
                parse_mrz_fallback(
                    mrz_text
                )
            )

            if is_mrz_valid:

                st.success(
                    "✅ Standard Passport MRZ Detected"
                )

                st.code(
                    "\n".join(
                        mrz_lines
                    )
                )

                fields = (
                    extract_mrz_fields(
                        mrz_lines
                    )
                )

                if fields:

                    st.subheader(
                        "📋 Extracted MRZ Fields"
                    )

                    for key, value in (
                        fields.items()
                    ):

                        st.write(
                            f"**{key}:** {value}"
                        )

            else:

                st.warning(
                    "⚠️ Two valid passport MRZ "
                    "lines were not confirmed."
                )


# ============================================================
# TAB 2 — VALIDATION
# ============================================================

with t2:

    st.header(
        "📏 Document Integrity & "
        "Standard Checks"
    )

    width, height = (
        doc_image.size
    )

    is_valid_res = (
        width >= 600
        and height >= 400
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Width",
        f"{width} px"
    )

    c2.metric(
        "Height",
        f"{height} px"
    )

    c3.metric(
        "Resolution",
        "PASS"
        if is_valid_res
        else
        "FAIL"
    )

    st.divider()

    st.checkbox(
        "Aspect Ratio Standard",
        value=width > height
    )

    st.checkbox(
        "Minimum Pixel Density",
        value=is_valid_res
    )

    st.checkbox(
        "Passport MRZ Structure",
        value=is_mrz_valid
        if "is_mrz_valid"
        in locals()
        else False
    )


# ============================================================
# TAB 3 — ELA
# ============================================================

with t3:

    st.header(
        "🔍 Digital Tampering Analysis"
    )

    st.write(
        "ELA highlights compression "
        "inconsistencies that may indicate "
        "digital editing."
    )

    ela_img, tamper_ratio = (
        perform_ela(
            doc_image
        )
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        st.image(
            doc_image,
            caption="Original Document",
            use_container_width=True
        )

    with c2:

        st.image(
            ela_img,
            caption="ELA Heatmap",
            use_container_width=True
        )

    st.divider()

    st.metric(
        "Detected Tamper Variance",
        f"{tamper_ratio}%"
    )

    if tamper_ratio < 15:

        st.success(
            "🟢 LOW: Consistent compression"
        )

    elif tamper_ratio < 35:

        st.warning(
            "🟡 MEDIUM: Compression anomalies"
        )

    else:

        st.error(
            "🔴 HIGH: Significant anomalies"
        )

    st.caption(
        "ELA is an indicator only and does "
        "not by itself prove document forgery."
    )


# ============================================================
# TAB 4 — BIOMETRIC
# ============================================================

with t4:

    st.header(
        "👤 1:1 Biometric Face Verification"
    )

    if face_engine_error:

        st.error(
            f"Face Engine: {face_engine_error}"
        )

    if live_photo is None:

        st.info(
            "📷 Capture a live passenger "
            "photo using the sidebar."
        )

    else:

        passenger_img = (
            Image.open(
                live_photo
            ).convert("RGB")
        )

        (
            sim_score,
            face_crop1,
            face_crop2,
            biometric_result
        ) = extract_and_compare_faces(

            doc_image,

            passenger_img
        )

        b1, b2 = (
            st.columns(2)
        )

        with b1:

            st.image(
                doc_image,
                caption="Passport",
                use_container_width=True
            )

            if face_crop1 is not None:

                st.image(
                    face_crop1,
                    caption="Detected Passport Face",
                    width=180
                )

        with b2:

            st.image(
                passenger_img,
                caption="Live Passenger",
                use_container_width=True
            )

            if face_crop2 is not None:

                st.image(
                    face_crop2,
                    caption="Detected Live Face",
                    width=180
                )

        st.divider()

        if isinstance(
            biometric_result,
            dict
        ):

            cosine = (
                biometric_result[
                    "cosine"
                ]
            )

            threshold = (
                biometric_result[
                    "threshold"
                ]
            )

            matched = (
                biometric_result[
                    "matched"
                ]
            )

            c1, c2, c3 = (
                st.columns(3)
            )

            c1.metric(
                "SFace Similarity",
                f"{cosine:.4f}"
            )

            c2.metric(
                "Reference Threshold",
                f"{threshold:.3f}"
            )

            c3.metric(
                "Display Score",
                f"{sim_score}%"
            )

            st.write(
                f"**Faces detected:** "
                f"Passport = "
                f"{biometric_result['doc_faces']}, "
                f"Live = "
                f"{biometric_result['passenger_faces']}"
            )

            if matched:

                st.success(
                    "🟢 BIOMETRIC MATCH CONFIRMED"
                )

                st.write(
                    "The SFace similarity is "
                    "above the configured threshold."
                )

            else:

                st.error(
                    "🔴 BIOMETRIC MISMATCH DETECTED"
                )

                st.write(
                    "The SFace similarity is "
                    "below the configured threshold."
                )

            st.info(
                "The decision uses SFace cosine "
                "similarity. The displayed percentage "
                "is only a UI representation and is "
                "not a probability."
            )

        else:

            st.error(
                str(
                    biometric_result
                )
            )


# ============================================================
# TAB 5 — RISK MATRIX
# ============================================================

with t5:

    st.header(
        "📊 Composite Risk Matrix"
    )

    width, height = (
        doc_image.size
    )

    is_valid_res = (
        width >= 600
        and height >= 400
    )

    res_score = (
        100
        if is_valid_res
        else 40
    )

    # MRZ score

    if "is_mrz_valid" in locals():

        mrz_score = (
            100
            if is_mrz_valid
            else 30
        )

    else:

        mrz_score = 30

    # ELA score

    tamper_score = max(
        0,
        int(
            100
            - (
                tamper_ratio
                * 3
            )
        )
    )

    # Biometric score

    bio_score = 50.0

    if live_photo is not None:

        passenger_img = (
            Image.open(
                live_photo
            ).convert("RGB")
        )

        (
            score,
            _,
            _,
            biometric_result
        ) = extract_and_compare_faces(

            doc_image,

            passenger_img
        )

        if isinstance(
            biometric_result,
            dict
        ):

            if biometric_result[
                "matched"
            ]:

                bio_score = 100.0

            else:

                bio_score = max(
                    0.0,
                    score
                )

    # Composite score

    composite_index = round(

        (res_score * 0.15)

        + (mrz_score * 0.25)

        + (tamper_score * 0.30)

        + (bio_score * 0.30),

        1
    )

    st.metric(
        "Total Compliance & Trust Index",
        f"{composite_index} / 100"
    )

    st.progress(
        int(
            composite_index
        )
    )

    st.divider()

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Resolution",
        f"{res_score}/100"
    )

    c2.metric(
        "MRZ",
        f"{mrz_score}/100"
    )

    c3.metric(
        "ELA",
        f"{tamper_score}/100"
    )

    c4.metric(
        "Biometric",
        f"{bio_score:.1f}/100"
    )

    st.divider()

    if composite_index >= 65:

        st.success(
            "✅ DECISION: "
            "STANDARD CLEARANCE / LOW RISK"
        )

    elif composite_index >= 45:

        st.warning(
            "⚠️ DECISION: "
            "MANUAL DOCUMENT INSPECTION REQUIRED"
        )

    else:

        st.error(
            "🚨 DECISION: "
            "HIGH RISK / SECONDARY INSPECTION"
        )

    st.caption(
        "Prototype decision-support system. "
        "Biometric and forensic results are "
        "indicators and should not be treated "
        "as definitive proof of identity or "
        "document authenticity."
    )