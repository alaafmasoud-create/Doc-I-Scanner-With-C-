import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

try:
    import docscanner_cpp
    CPP_IMPORT_ERROR = None
except Exception as e:
    docscanner_cpp = None
    CPP_IMPORT_ERROR = e


# -----------------------------
# Python-side helpers (UI / I/O)
# -----------------------------
def make_preview_for_clicks(image, max_width=1000, max_height=1400):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb, scale


def draw_points_on_preview(preview_rgb, points_preview, radius=8):
    canvas = preview_rgb.copy()

    for idx, (x, y) in enumerate(points_preview):
        cv2.circle(canvas, (int(x), int(y)), radius, (255, 0, 0), -1)
        cv2.putText(
            canvas,
            str(idx + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def decode_uploaded_image(file_bytes):
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen subida.")
    return img


def image_to_download_bytes(image_bgr):
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return None
    return buffer.tobytes()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Escáner de Documentos A4", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-box {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(59,130,246,0.16);
        border-radius: 22px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: .35rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 0;
    }
    .section-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(59,130,246,0.14);
        border-radius: 20px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 12px 28px rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: .2rem;
    }
    .section-note {
        font-size: .95rem;
        color: #475569;
        margin-bottom: .65rem;
    }
    .stButton > button {
        border-radius: 14px !important;
        border: none !important;
        min-height: 3rem;
        font-weight: 800 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        box-shadow: 0 10px 24px rgba(37,99,235,0.28) !important;
        transition: all .2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(37,99,235,0.34) !important;
        filter: brightness(1.03);
    }
    .stButton > button:focus,
    .stDownloadButton > button:focus,
    div[data-baseweb="select"] *:focus {
        outline: 3px solid rgba(59,130,246,0.25) !important;
        outline-offset: 2px !important;
    }
    .stDownloadButton > button {
        border-radius: 14px !important;
        border: none !important;
        min-height: 3rem;
        font-weight: 800 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        box-shadow: 0 10px 24px rgba(16,185,129,0.28) !important;
        transition: all .2s ease;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(16,185,129,0.34) !important;
        filter: brightness(1.03);
    }
    .stFileUploader, .stRadio {
        background: rgba(255,255,255,0.82);
        border-radius: 16px;
        padding: .4rem .55rem;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.75rem;
        padding-top: 0.2rem;
    }
    .stRadio [role="radiogroup"] label {
        background: linear-gradient(135deg, #ede9fe 0%, #dbeafe 100%);
        border: 1px solid #a5b4fc;
        border-radius: 14px;
        padding: 0.45rem 0.95rem;
        box-shadow: 0 6px 14px rgba(79,70,229,0.12);
    }
    .stRadio [role="radiogroup"] label p {
        color: #312e81;
        font-weight: 800;
    }
    div[data-testid="stSelectbox"] {
        background: linear-gradient(135deg, rgba(255,247,237,0.98) 0%, rgba(254,242,242,0.98) 100%);
        border: 1px solid #fdba74;
        border-radius: 18px;
        padding: 0.6rem 0.7rem 0.75rem 0.7rem;
        box-shadow: 0 10px 22px rgba(249,115,22,0.12);
        margin-bottom: 0.9rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #9a3412 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
    }
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] {
        background: rgba(255,255,255,0.92) !important;
        border: 2px solid #fb923c !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(249,115,22,0.10);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .footer-signature {
        text-align: center;
        margin-top: 2.2rem;
        padding-top: 0.8rem;
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .stFileUploader {
        border: 2px dashed #38bdf8;
        background: linear-gradient(135deg, rgba(224,242,254,0.9) 0%, rgba(240,249,255,0.96) 100%);
        box-shadow: 0 8px 22px rgba(14,165,233,0.10);
    }
    .stFileUploader label {
        color: #0f172a !important;
        font-weight: 800;
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 22px rgba(37,99,235,0.22);
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37,99,235,0.28);
    }
    div[data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    }
    .result-label {
        display: inline-block;
        padding: .36rem .75rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1d4ed8;
        font-size: .84rem;
        font-weight: 800;
        margin-bottom: .55rem;
        border: 1px solid #93c5fd;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">📄 Escáner de Documentos A4</div>
    <p class="hero-subtitle">
        Sube una imagen de un documento A4. Usa el modo automático predeterminado o ajusta manualmente las esquinas para un recorte más preciso.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if CPP_IMPORT_ERROR is not None:
    st.error("No se pudo cargar el módulo C++ (docscanner_cpp).")
    st.code(str(CPP_IMPORT_ERROR))
    st.info(
        "Comprueba que el despliegue haya instalado requirements.txt y packages.txt, "
        "y que el módulo haya compilado correctamente durante el build."
    )
    st.stop()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Configuración del escáner</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Elige el modo y sube una o varias imágenes.</div>', unsafe_allow_html=True)

mode = st.radio("Modo", ["Automático", "Manual"], horizontal=True)

uploaded_files = st.file_uploader(
    "Subir imagen",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

st.markdown("</div>", unsafe_allow_html=True)

if "manual_points_preview" not in st.session_state:
    st.session_state.manual_points_preview = []

if "manual_points_original" not in st.session_state:
    st.session_state.manual_points_original = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "last_uploaded_key" not in st.session_state:
    st.session_state.last_uploaded_key = None

if uploaded_files:
    if mode == "Automático":
        for file_index, uploaded_file in enumerate(uploaded_files):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">RESULTADO AUTOMÁTICO</div>', unsafe_allow_html=True)
            st.subheader(f"{uploaded_file.name}")

            file_bytes = uploaded_file.getvalue()
            original = decode_uploaded_image(file_bytes)

            try:
                result = docscanner_cpp.detect_document_auto(original)
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Resultado final",
                    use_container_width=True,
                )

                download_bytes = image_to_download_bytes(result)
                if download_bytes is not None:
                    file_base = uploaded_file.name.rsplit(".", 1)[0]
                    st.download_button(
                        label="Descargar resultado final",
                        data=download_bytes,
                        file_name=f"{file_base}_final_result.png",
                        mime="image/png",
                        key=f"download_auto_{file_index}",
                    )

            except Exception as e:
                st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">MODO MANUAL</div>', unsafe_allow_html=True)

        selected_file_name = st.selectbox(
            "Selecciona una imagen para el modo manual",
            [file.name for file in uploaded_files],
        )

        uploaded_file = next(file for file in uploaded_files if file.name == selected_file_name)
        upload_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.last_uploaded_key != upload_key:
            st.session_state.manual_points_preview = []
            st.session_state.manual_points_original = []
            st.session_state.last_click = None
            st.session_state.last_uploaded_key = upload_key

        file_bytes = uploaded_file.getvalue()
        original = decode_uploaded_image(file_bytes)

        preview_rgb, preview_scale = make_preview_for_clicks(
            original,
            max_width=1000,
            max_height=1400,
        )

        preview_with_points = draw_points_on_preview(
            preview_rgb,
            st.session_state.manual_points_preview,
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Restablecer puntos"):
                st.session_state.manual_points_preview = []
                st.session_state.manual_points_original = []
                st.session_state.last_click = None
                st.rerun()

        with col_btn2:
            if st.button("Deshacer el último punto"):
                if st.session_state.manual_points_preview:
                    st.session_state.manual_points_preview.pop()
                if st.session_state.manual_points_original:
                    st.session_state.manual_points_original.pop()
                st.session_state.last_click = None
                st.rerun()

        st.subheader("Haz clic en las 4 esquinas")

        clicked = streamlit_image_coordinates(preview_with_points, key="manual_click_image")

        if clicked is not None:
            current_click = (clicked["x"], clicked["y"])

            if st.session_state.last_click != current_click:
                if len(st.session_state.manual_points_preview) < 4:
                    st.session_state.manual_points_preview.append(current_click)

                    ox = int(round(clicked["x"] / preview_scale))
                    oy = int(round(clicked["y"] / preview_scale))

                    ox = max(0, min(ox, original.shape[1] - 1))
                    oy = max(0, min(oy, original.shape[0] - 1))

                    st.session_state.manual_points_original.append((ox, oy))

                st.session_state.last_click = current_click
                st.rerun()

        if len(st.session_state.manual_points_original) == 4:
            try:
                points = np.array(st.session_state.manual_points_original, dtype=np.float32)
                result = docscanner_cpp.detect_document_manual(original, points)
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Resultado manual",
                    use_container_width=True,
                )

                download_bytes = image_to_download_bytes(result)
                if download_bytes is not None:
                    file_base = uploaded_file.name.rsplit(".", 1)[0]
                    st.download_button(
                        label="Descargar resultado final",
                        data=download_bytes,
                        file_name=f"{file_base}_final_result.png",
                        mime="image/png",
                    )

            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer-signature">By Alan Masoud</div>', unsafe_allow_html=True)
