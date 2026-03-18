import math
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import whisper

st.set_page_config(
    page_title="Transcriptor de Audio",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SUPPORTED_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "webm", "flac", "mp4", "mpeg", "mpga"]

# Opciones con nombres amigables
MODEL_OPTIONS = {
    "Rápido (menos preciso)": "tiny",
    "Equilibrado ✅ Recomendado": "base",
    "Preciso": "small",
    "Muy preciso (más lento)": "medium",
    "Máxima precisión (muy lento)": "large",
}

LANGUAGE_OPTIONS = {
    "Detectar automáticamente": "auto",
    "Español": "es",
    "Inglés": "en",
}


@st.cache_resource
def load_model(model_name: str):
    return whisper.load_model(model_name)


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def get_audio_duration(file_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def transcribe_streaming(file_path: str, model_name: str, language: str):
    """Generator: yields (text_acumulado, segments_acumulados, chunk_actual, total_chunks) por cada trozo."""
    model = load_model(model_name)
    options = {"task": "transcribe", "fp16": False}
    if language != "auto":
        options["language"] = language

    duration = get_audio_duration(file_path)
    chunk_size = 30  # segundos
    total_chunks = math.ceil(duration / chunk_size)

    all_text = ""
    all_segments = []
    detected_language = None

    for idx in range(total_chunks):
        start = idx * chunk_size
        suffix = Path(file_path).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            chunk_path = tmp.name

        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(chunk_size),
             "-i", file_path, chunk_path, "-loglevel", "error"],
            check=True,
        )

        result = model.transcribe(chunk_path, **options)
        os.remove(chunk_path)

        if detected_language is None:
            detected_language = result.get("language", "desconocido")

        chunk_text = result.get("text", "").strip()
        for seg in result.get("segments", []):
            adjusted = seg.copy()
            adjusted["start"] += start
            adjusted["end"] += start
            all_segments.append(adjusted)

        all_text = (all_text + " " + chunk_text).strip()
        yield all_text, all_segments, idx + 1, total_chunks, detected_language


def to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def build_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


# ── Encabezado ──────────────────────────────────────────────────────────────
st.title("🎙️ Transcriptor de Audio")
st.markdown("Convierte audios y vídeos a texto de forma automática, sin enviar nada a internet.")

st.divider()

# ── Paso 1: Subir o grabar ───────────────────────────────────────────────────
st.subheader("Paso 1 — Elige tu audio")

tab_file, tab_mic = st.tabs(["📁 Subir archivo", "🎤 Grabar con el micrófono"])

with tab_file:
    uploaded_file = st.file_uploader(
        "Arrastra tu archivo aquí o haz clic para buscarlo",
        type=SUPPORTED_EXTENSIONS,
        label_visibility="collapsed",
    )
    st.caption("Formatos aceptados: MP3, WAV, M4A, MP4, OGG, FLAC, WEBM…")

with tab_mic:
    mic_audio = st.audio_input("Pulsa el botón rojo para grabar")

source_file = mic_audio if mic_audio is not None else uploaded_file

if source_file is not None:
    st.audio(source_file)

st.divider()

# ── Paso 2: Opciones ─────────────────────────────────────────────────────────
st.subheader("Paso 2 — Opciones (opcional)")

col1, col2 = st.columns(2)

with col1:
    selected_language_label = st.selectbox(
        "Idioma del audio",
        list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Si no sabes el idioma, deja 'Detectar automáticamente'.",
    )
    selected_language = LANGUAGE_OPTIONS[selected_language_label]

with col2:
    selected_model_label = st.selectbox(
        "Velocidad / precisión",
        list(MODEL_OPTIONS.keys()),
        index=1,
        help="'Equilibrado' funciona bien para la mayoría de los casos.",
    )
    selected_model = MODEL_OPTIONS[selected_model_label]

st.divider()

# ── Paso 3: Transcribir ───────────────────────────────────────────────────────
st.subheader("Paso 3 — Transcribir")

if source_file is None:
    st.info("⬆️ Primero sube un archivo o graba un audio en el Paso 1.")
    st.stop()

if st.button("▶️ Iniciar transcripción", type="primary", use_container_width=True):
    temp_path = None
    try:
        temp_path = save_uploaded_file(source_file)

        st.subheader("Resultado")
        progress_bar = st.progress(0, text="Iniciando…")
        live_text = st.empty()

        transcript = ""
        segments = []
        detected_language = "desconocido"

        for text, segs, current, total, lang in transcribe_streaming(
            file_path=temp_path,
            model_name=selected_model,
            language=selected_language,
        ):
            transcript = text
            segments = segs
            detected_language = lang
            progress = current / total
            progress_bar.progress(progress, text=f"Trozo {current} / {total}…")
            live_text.text_area(
                "Texto transcrito",
                transcript,
                height=280,
                label_visibility="collapsed",
                key=f"live_{current}",
            )

        progress_bar.empty()
        st.success("¡Transcripción completada!")

        lang_names = {v: k for k, v in LANGUAGE_OPTIONS.items()}
        lang_display = lang_names.get(detected_language, detected_language)
        st.caption(f"Idioma detectado: {lang_display}")

        st.download_button(
            "⬇️ Descargar como TXT",
            data=transcript.encode("utf-8"),
            file_name="transcripcion.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except FileNotFoundError:
        st.error(
            "No se pudo procesar el audio porque falta un componente del sistema (ffmpeg). "
            "Pide ayuda a la persona que instaló esta aplicación."
        )
    except Exception as e:
        st.error(
            f"Algo salió mal durante la transcripción. "
            f"Intenta con un archivo más corto o un modelo diferente.\n\nDetalle técnico: {e}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
