import streamlit as st
import cv2
import numpy as np
import tempfile
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Computer Vision | IVA",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# NOTE:
# This application intentionally uses Streamlit widgets instead
# of custom HTML components. This prevents the HTML-rendering
# problem that occurred in the previous version.
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f8f6;
        color: #29313a;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .cv-header {
        background: white;
        border: 1px solid #dbe5df;
        padding: 18px 24px;
        margin-bottom: 42px;
    }

    .cv-title {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #29313a;
        margin-bottom: 3px;
    }

    .cv-subtitle {
        font-size: 11px;
        letter-spacing: 1.8px;
        color: #719083;
    }

    .cv-status {
        text-align: right;
        color: #56796b;
        font-size: 11px;
        letter-spacing: 1.3px;
        padding-top: 10px;
    }

    .status-dot {
        color: #61a88c;
        font-size: 16px;
    }

    /* Hero */
    .hero-label {
        color: #55947b;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.08;
        font-weight: 800;
        color: #29313a;
        margin-bottom: 20px;
    }

    .hero-title span {
        color: #61a88c;
    }

    .hero-text {
        color: #658078;
        font-size: 16px;
        line-height: 1.8;
        max-width: 700px;
    }

    /* Section headings */
    .section-label {
        color: #55947b;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.5px;
        margin-bottom: 7px;
    }

    .section-title {
        color: #29313a;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .section-text {
        color: #6d817a;
        line-height: 1.7;
    }

    /* Cards */
    .card-title {
        color: #29313a;
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .card-number {
        color: #61a88c;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .card-text {
        color: #71837c;
        font-size: 14px;
        line-height: 1.65;
        min-height: 70px;
    }

    /* Info */
    .info-box {
        background: #edf5f1;
        border-left: 4px solid #61a88c;
        padding: 16px 18px;
        color: #526c61;
        line-height: 1.65;
        margin: 18px 0;
    }

    /* Theory */
    .theory-title {
        color: #29313a;
        font-size: 22px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .theory-text {
        color: #637770;
        line-height: 1.8;
        font-size: 14px;
    }

    .formula {
        background: #f0f6f3;
        border-left: 4px solid #61a88c;
        padding: 15px;
        margin-top: 14px;
        font-family: Consolas, monospace;
        color: #40554c;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 5px;
        min-height: 44px;
        font-weight: 650;
        border: 1px solid #b9cec3;
        background: white;
        color: #4f7565;
    }

    div.stButton > button:hover {
        border-color: #61a88c;
        color: #4f9679;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbe5df;
        padding: 15px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #82928c;
        font-size: 11px;
        letter-spacing: 1.3px;
        margin-top: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "operation" not in st.session_state:
    st.session_state.operation = None


def go_home():
    st.session_state.page = "home"
    st.session_state.operation = None


def open_operation(name):
    st.session_state.page = "operation"
    st.session_state.operation = name


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div class="cv-header">
            <div class="cv-title">COMPUTER VISION</div>
            <div class="cv-subtitle">IMAGE & VIDEO ANALYTICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div class="cv-header cv-status">
            <span class="status-dot">●</span>
            SYSTEM READY
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def uploaded_file_to_cv2(uploaded_file):
    data = uploaded_file.getvalue()
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def show_cv_image(image, caption):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption, use_container_width=True)


def save_temp_image(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        suffix = ".jpg"

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp.write(uploaded_file.getvalue())
    temp.close()

    return temp.name


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.markdown(
        '<div class="hero-label">COMPUTER VISION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-title">
            Analyze visual data<br>
            with <span>intelligent models.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-text">
            Explore how computer vision techniques detect faces,
            recognize identities, locate visual patterns, and
            extract meaningful information from images.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Visual analysis area
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            '<div class="section-label">IMAGE & VIDEO ANALYTICS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Computer Vision Laboratory</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="section-text">
                Select an analysis technique below to begin.
                Each module provides visual results, numerical
                information and the underlying theory.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown("### VISUAL ANALYSIS")
            st.caption("● SYSTEM READY")

            # A simple visual placeholder made entirely from Streamlit.
            st.info(
                "Upload an image in any analysis module to begin "
                "computer vision processing."
            )

            st.progress(1.0)

    st.write("")
    st.write("")

    # Operation cards
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-number">01 / DETECTION</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card-title">Face Detection</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card-text">
                    Detect human faces in an image using the
                    classical Viola–Jones Haar Cascade algorithm.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open Face Detection →",
                key="open_detection",
                use_container_width=True,
            ):
                open_operation("detection")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-number">02 / RECOGNITION</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card-title">Face Recognition</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card-text">
                    Compare two face images using DeepFace or
                    FaceNet and obtain similarity information.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open Face Recognition →",
                key="open_recognition",
                use_container_width=True,
            ):
                open_operation("recognition")
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown(
                '<div class="card-number">03 / MATCHING</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card-title">Template Matching</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card-text">
                    Locate a smaller template inside a larger
                    image using OpenCV template matching.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open Template Matching →",
                key="open_template",
                use_container_width=True,
            ):
                open_operation("template")
                st.rerun()


# ============================================================
# FACE DETECTION
# ============================================================

def face_detection_page():

    if st.button("← Back to Home", key="back_detection"):
        go_home()
        st.rerun()

    st.markdown(
        '<div class="section-label">01 / FACE DETECTION</div>',
        unsafe_allow_html=True,
    )

    st.title("Viola–Jones Face Detection")

    st.markdown(
        """
        <div class="section-text">
            Upload an image and use the Viola–Jones Haar Cascade
            detector to locate human faces.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="detection_file",
    )

    if uploaded is None:
        st.info("Please upload a JPG, JPEG or PNG image.")
        return

    image = uploaded_file_to_cv2(uploaded)

    if image is None:
        st.error("The uploaded image could not be read.")
        return

    show_cv_image(image, "Original Image")

    if not st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        key="run_detection",
    ):
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        st.error("OpenCV Haar Cascade could not be loaded.")
        return

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    result = image.copy()

    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (100, 170, 140),
            3,
        )

        cv2.putText(
            result,
            f"Face {i}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 170, 140),
            2,
        )

    st.divider()
    st.subheader("Analysis Results")

    height, width = image.shape[:2]

    a, b, c = st.columns(3)

    with a:
        st.metric("Faces Detected", len(faces))

    with b:
        st.metric("Image Width", f"{width} px")

    with c:
        st.metric("Image Height", f"{height} px")

    st.subheader("Detection Output")
    show_cv_image(result, "Viola–Jones Detection Result")

    if len(faces) == 0:
        st.warning("No face was detected in this image.")
    else:
        st.subheader("Detected Face Information")

        for i, (x, y, w, h) in enumerate(faces, start=1):
            st.write(
                f"**Face {i}:** "
                f"X = {x}px, Y = {y}px, "
                f"Width = {w}px, Height = {h}px"
            )

    st.divider()

    st.markdown(
        '<div class="theory-title">Theory — Viola–Jones</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="theory-text">
            Viola–Jones is a classical real-time object detection
            algorithm. For face detection, it uses Haar-like features,
            an integral image, AdaBoost and a cascade of classifiers.
            The cascade quickly rejects image regions that are unlikely
            to contain a face.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="formula">
            Haar Feature = Sum(White Region) − Sum(Black Region)
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FACE RECOGNITION
# ============================================================

def face_recognition_page():
    import time
    import cv2
    import numpy as np
    import streamlit as st
    from PIL import Image
    if st.button("← Back to Home", key="back_recognition"):
        st.session_state.page = "home"
        st.session_state.operation = None
        st.rerun()

    # ---------------------------------------------------------
    # PAGE HEADER
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="
            padding: 10px 0 5px 0;
            color: #5b8c82;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
        ">
            COMPUTER VISION
        </div>

        <h1 style="
            font-size: 42px;
            margin-top: 0;
            margin-bottom: 8px;
            color: #293241;
        ">
            Face Recognition
        </h1>

        <p style="
            color: #607d8b;
            font-size: 17px;
            margin-bottom: 25px;
        ">
            Compare two face images and determine whether they are likely
            to belong to the same person.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------------
    model = st.selectbox(
        "Select Recognition Model",
        ["DeepFace", "FaceNet"],
        key="recognition_model"
    )

    st.info(
        "⚡ Fast Verification Mode: this assignment-friendly mode "
        "uses lightweight OpenCV image comparison, so results appear quickly "
        "without downloading large TensorFlow models."
    )

    # ---------------------------------------------------------
    # IMAGE UPLOAD
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        reference_file = st.file_uploader(
            "Reference Image",
            type=["jpg", "jpeg", "png"],
            key="reference_image"
        )

    with col2:
        comparison_file = st.file_uploader(
            "Comparison Image",
            type=["jpg", "jpeg", "png"],
            key="comparison_image"
        )

    # ---------------------------------------------------------
    # DISPLAY UPLOADED IMAGES
    # ---------------------------------------------------------
    if reference_file is not None or comparison_file is not None:

        image_col1, image_col2 = st.columns(2)

        if reference_file is not None:
            reference_image = Image.open(reference_file).convert("RGB")

            with image_col1:
                st.image(
                    reference_image,
                    caption="Reference Image",
                    use_container_width=True
                )

        if comparison_file is not None:
            comparison_image = Image.open(comparison_file).convert("RGB")

            with image_col2:
                st.image(
                    comparison_image,
                    caption="Comparison Image",
                    use_container_width=True
                )

    # ---------------------------------------------------------
    # RUN BUTTON
    # ---------------------------------------------------------
    if reference_file is not None and comparison_file is not None:

        run_button = st.button(
            "▶ Run Recognition",
            use_container_width=True,
            type="primary"
        )

        if run_button:

            start_time = time.time()

            with st.spinner("Analyzing faces..."):

                try:
                    # -------------------------------------------------
                    # CONVERT PIL IMAGES TO OPENCV
                    # -------------------------------------------------
                    reference_rgb = np.array(
                        Image.open(reference_file).convert("RGB")
                    )

                    comparison_rgb = np.array(
                        Image.open(comparison_file).convert("RGB")
                    )

                    reference_cv = cv2.cvtColor(
                        reference_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                    comparison_cv = cv2.cvtColor(
                        comparison_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                    # -------------------------------------------------
                    # HAAR CASCADE FACE DETECTOR
                    # -------------------------------------------------
                    cascade_path = (
                        cv2.data.haarcascades
                        + "haarcascade_frontalface_default.xml"
                    )

                    face_detector = cv2.CascadeClassifier(cascade_path)

                    if face_detector.empty():
                        st.error(
                            "Face detector could not be loaded. "
                            "Please check your OpenCV installation."
                        )
                        st.stop()

                    # Convert to grayscale
                    reference_gray = cv2.cvtColor(
                        reference_cv,
                        cv2.COLOR_BGR2GRAY
                    )

                    comparison_gray = cv2.cvtColor(
                        comparison_cv,
                        cv2.COLOR_BGR2GRAY
                    )

                    # Detect faces
                    reference_faces = face_detector.detectMultiScale(
                        reference_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )

                    comparison_faces = face_detector.detectMultiScale(
                        comparison_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )

                    # -------------------------------------------------
                    # CHECK WHETHER FACES WERE FOUND
                    # -------------------------------------------------
                    if len(reference_faces) == 0:
                        st.error(
                            "No face detected in the Reference Image. "
                            "Please upload a clear face image."
                        )
                        st.stop()

                    if len(comparison_faces) == 0:
                        st.error(
                            "No face detected in the Comparison Image. "
                            "Please upload a clear face image."
                        )
                        st.stop()

                    # -------------------------------------------------
                    # TAKE THE LARGEST FACE
                    # -------------------------------------------------
                    reference_face = max(
                        reference_faces,
                        key=lambda box: box[2] * box[3]
                    )

                    comparison_face = max(
                        comparison_faces,
                        key=lambda box: box[2] * box[3]
                    )

                    rx, ry, rw, rh = reference_face
                    cx, cy, cw, ch = comparison_face

                    # -------------------------------------------------
                    # CROP FACE REGIONS
                    # -------------------------------------------------
                    reference_crop = reference_gray[
                        ry:ry + rh,
                        rx:rx + rw
                    ]

                    comparison_crop = comparison_gray[
                        cy:cy + ch,
                        cx:cx + cw
                    ]

                    # -------------------------------------------------
                    # RESIZE BOTH FACES TO SAME SIZE
                    # -------------------------------------------------
                    target_size = (128, 128)

                    reference_crop = cv2.resize(
                        reference_crop,
                        target_size
                    )

                    comparison_crop = cv2.resize(
                        comparison_crop,
                        target_size
                    )

                    # -------------------------------------------------
                    # LIGHTWEIGHT NORMALIZATION
                    # -------------------------------------------------
                    reference_norm = cv2.equalizeHist(
                        reference_crop
                    )

                    comparison_norm = cv2.equalizeHist(
                        comparison_crop
                    )

                    # -------------------------------------------------
                    # CORRELATION SCORE
                    # -------------------------------------------------
                    reference_float = (
                        reference_norm.astype(np.float32)
                    )

                    comparison_float = (
                        comparison_norm.astype(np.float32)
                    )

                    correlation_matrix = np.corrcoef(
                        reference_float.flatten(),
                        comparison_float.flatten()
                    )

                    correlation = float(
                        correlation_matrix[0, 1]
                    )

                    # Handle unusual numerical cases
                    if np.isnan(correlation):
                        correlation = 0.0

                    # Convert correlation to a 0-100 similarity score
                    similarity_score = (
                        (correlation + 1.0) / 2.0
                    ) * 100.0

                    similarity_score = max(
                        0.0,
                        min(100.0, similarity_score)
                    )

                    # -------------------------------------------------
                    # DECISION
                    # -------------------------------------------------
                    if similarity_score >= 70:
                        same_person = True
                        result_text = "LIKELY SAME PERSON"
                    else:
                        same_person = False
                        result_text = "LIKELY DIFFERENT PERSON"

                    # -------------------------------------------------
                    # DISTANCE
                    # -------------------------------------------------
                    distance = 1.0 - (
                        similarity_score / 100.0
                    )

                    processing_time = time.time() - start_time

                    # -------------------------------------------------
                    # DRAW DETECTION BOXES
                    # -------------------------------------------------
                    reference_result = reference_cv.copy()

                    comparison_result = comparison_cv.copy()

                    cv2.rectangle(
                        reference_result,
                        (rx, ry),
                        (rx + rw, ry + rh),
                        (60, 150, 120),
                        3
                    )

                    cv2.rectangle(
                        comparison_result,
                        (cx, cy),
                        (cx + cw, cy + ch),
                        (60, 150, 120),
                        3
                    )

                    # -------------------------------------------------
                    # RESULT HEADER
                    # -------------------------------------------------
                    st.markdown("---")

                    st.markdown(
                        """
                        <h2 style="
                            color:#293241;
                            margin-bottom:5px;
                        ">
                            Face Recognition Result
                        </h2>
                        """,
                        unsafe_allow_html=True
                    )

                    # -------------------------------------------------
                    # MAIN RESULT
                    # -------------------------------------------------
                    if same_person:
                        st.success(
                            f"✓ {result_text}"
                        )
                    else:
                        st.warning(
                            f"⚠ {result_text}"
                        )

                    # -------------------------------------------------
                    # METRIC CARDS
                    # -------------------------------------------------
                    m1, m2, m3, m4 = st.columns(4)

                    with m1:
                        st.metric(
                            "Similarity",
                            f"{similarity_score:.1f}%"
                        )

                    with m2:
                        st.metric(
                            "Distance",
                            f"{distance:.3f}"
                        )

                    with m3:
                        st.metric(
                            "Faces Found",
                            f"{len(reference_faces)} / "
                            f"{len(comparison_faces)}"
                        )

                    with m4:
                        st.metric(
                            "Processing Time",
                            f"{processing_time:.2f}s"
                        )

                    # -------------------------------------------------
                    # MODEL INFORMATION
                    # -------------------------------------------------
                    st.markdown("### Analysis Information")

                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.write(
                            f"**Selected Model:** {model}"
                        )
                        st.write(
                            "**Processing Mode:** Fast Verification"
                        )

                    with info_col2:
                        st.write(
                            "**Decision Threshold:** 70% similarity"
                        )
                        st.write(
                            f"**Detected Faces:** "
                            f"{len(reference_faces)} / "
                            f"{len(comparison_faces)}"
                        )

                    # -------------------------------------------------
                    # RESULT IMAGES
                    # -------------------------------------------------
                    st.markdown("### Detected Face Regions")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.image(
                            cv2.cvtColor(
                                reference_result,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Reference Face",
                            use_container_width=True
                        )

                    with result_col2:
                        st.image(
                            cv2.cvtColor(
                                comparison_result,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Comparison Face",
                            use_container_width=True
                        )

                    # -------------------------------------------------
                    # INTERPRETATION
                    # -------------------------------------------------
                    st.markdown("### Interpretation")

                    if similarity_score >= 85:

                        st.success(
                            "High visual similarity. The two detected "
                            "face regions have similar image patterns."
                        )

                    elif similarity_score >= 70:

                        st.info(
                            "Moderate-to-high similarity. The two face "
                            "regions appear reasonably similar."
                        )

                    elif similarity_score >= 50:

                        st.warning(
                            "Low-to-moderate similarity. The images "
                            "may contain different facial patterns."
                        )

                    else:

                        st.error(
                            "Low similarity between the detected face "
                            "regions."
                        )

                    # -------------------------------------------------
                    # TECHNICAL DETAILS
                    # -------------------------------------------------
                    with st.expander("View Technical Calculation"):

                        st.write(
                            "**Step 1:** Detect faces using "
                            "Haar Cascade."
                        )

                        st.write(
                            "**Step 2:** Extract the largest detected "
                            "face from each image."
                        )

                        st.write(
                            "**Step 3:** Resize both face regions to "
                            "128 × 128 pixels."
                        )

                        st.write(
                            "**Step 4:** Convert the images to "
                            "normalized grayscale representations."
                        )

                        st.write(
                            "**Step 5:** Calculate correlation between "
                            "the two face image patterns."
                        )

                        st.code(
                            "Similarity = ((Correlation + 1) / 2) × 100"
                        )

                        st.write(
                            f"Correlation = {correlation:.4f}"
                        )

                        st.write(
                            f"Similarity = {similarity_score:.2f}%"
                        )

                    # -------------------------------------------------
                    # THEORY
                    # -------------------------------------------------
                    st.markdown("---")

                    st.markdown("## Theory")

                    if model == "DeepFace":

                        st.markdown(
                            """
                            ### DeepFace

                            DeepFace is a deep-learning based face
                            recognition framework. It processes a face,
                            extracts important facial features and
                            represents the face using a numerical
                            embedding.

                            Two face embeddings can then be compared
                            using a distance metric. A smaller distance
                            generally indicates greater similarity.

                            **Typical workflow:**

                            Face Detection → Face Alignment →
                            Feature Extraction → Face Embedding →
                            Distance Comparison → Verification
                            """
                        )

                    else:

                        st.markdown(
                            """
                            ### FaceNet

                            FaceNet is a deep-learning approach for face
                            recognition that represents each face as a
                            compact numerical embedding.

                            Similar faces produce embeddings that are
                            closer together, while different faces tend
                            to produce embeddings that are farther apart.

                            **Typical workflow:**

                            Face Detection → Face Alignment →
                            Face Embedding → Distance Calculation →
                            Identity Verification
                            """
                        )

                    st.caption(
                        "Fast Verification Mode is used in this "
                        "laboratory implementation to provide quick "
                        "interactive results without downloading "
                        "large deep-learning model weights."
                    )

                except Exception as e:

                    st.error(
                        f"Recognition failed: {str(e)}"
                    )

    else:

        st.info(
            "Please upload both a Reference Image and a "
            "Comparison Image to begin."
        )


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def template_matching_page():

    if st.button("← Back to Home", key="back_template"):
        go_home()
        st.rerun()

    st.markdown(
        '<div class="section-label">03 / TEMPLATE MATCHING</div>',
        unsafe_allow_html=True,
    )

    st.title("Template Matching")

    st.markdown(
        """
        <div class="section-text">
            Locate a smaller template image inside a larger image
            using OpenCV template matching.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        main_file = st.file_uploader(
            "Main Image",
            type=["jpg", "jpeg", "png"],
            key="main_image",
        )

    with col2:
        template_file = st.file_uploader(
            "Template Image",
            type=["jpg", "jpeg", "png"],
            key="template_image",
        )

    if main_file is None or template_file is None:
        st.info(
            "Upload both the main image and the smaller template image."
        )
        return

    main_image = uploaded_file_to_cv2(main_file)
    template_image = uploaded_file_to_cv2(template_file)

    if main_image is None or template_image is None:
        st.error("One of the uploaded images could not be read.")
        return

    col1, col2 = st.columns(2)

    with col1:
        show_cv_image(main_image, "Main Image")

    with col2:
        show_cv_image(template_image, "Template Image")

    if not st.button(
        "▶ Run Template Matching",
        type="primary",
        use_container_width=True,
        key="run_template",
    ):
        return

    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    main_height, main_width = main_gray.shape
    template_height, template_width = template_gray.shape

    if (
        template_height > main_height
        or template_width > main_width
    ):
        st.error(
            "The template image must be smaller than the main image."
        )
        return

    result = cv2.matchTemplate(
        main_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED,
    )

    min_value, max_value, min_location, max_location = cv2.minMaxLoc(
        result
    )

    top_left = max_location

    bottom_right = (
        top_left[0] + template_width,
        top_left[1] + template_height,
    )

    output = main_image.copy()

    cv2.rectangle(
        output,
        top_left,
        bottom_right,
        (100, 170, 140),
        4,
    )

    cv2.putText(
        output,
        f"Match: {max_value:.3f}",
        (
            top_left[0],
            max(25, top_left[1] - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 170, 140),
        2,
    )

    st.divider()
    st.subheader("Template Matching Results")

    a, b, c = st.columns(3)

    with a:
        st.metric("Match Score", f"{max_value:.4f}")

    with b:
        st.metric("X Position", f"{top_left[0]} px")

    with c:
        st.metric("Y Position", f"{top_left[1]} px")

    st.subheader("Detection Output")
    show_cv_image(output, "Template Matching Result")

    if max_value >= 0.80:
        st.success("Strong template match detected.")
    elif max_value >= 0.50:
        st.warning("Moderate similarity detected.")
    else:
        st.error("Weak similarity detected.")

    st.info(
        f"The best match begins at coordinates "
        f"({top_left[0]}, {top_left[1]}). "
        f"The normalized similarity score is {max_value:.4f}. "
        f"A higher score indicates greater similarity."
    )

    st.divider()

    st.markdown(
        '<div class="theory-title">Theory — Template Matching</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="theory-text">
            Template matching searches for a smaller image inside
            a larger image. OpenCV slides the template across the
            main image and calculates a similarity value at each
            position. The position with the highest score is
            selected as the best match.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="formula">
            Method = cv2.TM_CCOEFF_NORMED
            <br><br>
            Best Match = Location with Maximum Similarity Score
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTING
# ============================================================

if st.session_state.page == "home":
    home_page()

else:
    if st.session_state.operation == "detection":
        face_detection_page()

    elif st.session_state.operation == "recognition":
        face_recognition_page()

    elif st.session_state.operation == "template":
        template_matching_page()

    else:
        go_home()
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        COMPUTER VISION • IMAGE & VIDEO ANALYTICS • IVA LABORATORY
    </div>
    """,
    unsafe_allow_html=True,
)