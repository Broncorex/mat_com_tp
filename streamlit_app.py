from typing import Tuple
import streamlit as st
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procesamiento Matricial B&W",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── ESTILOS GLOBALES ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,210,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p { color: #8b949e; font-size: 0.9rem; margin: 0; }

/* ── GUÍA DE PASOS (stepper) ── */
.stepper {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0 2rem;
    overflow-x: auto;
    padding-bottom: 4px;
}
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}
.step-bubble {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    border: 2px solid #30363d;
    background: #161b22;
    color: #8b949e;
    transition: all 0.3s;
}
.step-bubble.done   { background: #00d2ff22; border-color: #00d2ff; color: #00d2ff; }
.step-bubble.active { background: linear-gradient(135deg, #00d2ff33, #7b2ff733); border-color: #7b2ff7; color: #e6edf3; box-shadow: 0 0 12px rgba(123,47,247,0.4); }
.step-label { font-size: 0.72rem; color: #8b949e; white-space: nowrap; }
.step-label.active { color: #e6edf3; font-weight: 600; }
.step-label.done   { color: #00d2ff; }
.step-connector {
    width: 32px; height: 2px;
    background: #21262d;
    flex-shrink: 0;
    margin: 0 4px;
}
.step-connector.done { background: linear-gradient(90deg, #00d2ff, #30363d); }

/* ── BANNER INFO (pre-procesamiento) — VERSIÓN REFORZADA ── */
.info-banner {
    background: linear-gradient(135deg, #0a1a2e 0%, #071628 100%);
    border: 3px solid #00d2ff;
    border-radius: 20px;
    padding: 2.8rem 2.4rem 2.4rem;
    margin: 1.8rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 6px rgba(0,210,255,0.08),
        0 0 60px rgba(0, 210, 255, 0.30),
        inset 0 0 80px rgba(0,210,255,0.06);
    animation: glow-pulse-cyan 2.2s ease-in-out infinite;
}
@keyframes glow-pulse-cyan {
    0%, 100% {
        box-shadow:
            0 0 0 6px rgba(0,210,255,0.08),
            0 0 40px rgba(0,210,255,0.20),
            inset 0 0 80px rgba(0,210,255,0.04);
    }
    50% {
        box-shadow:
            0 0 0 8px rgba(0,210,255,0.14),
            0 0 80px rgba(0,210,255,0.45),
            inset 0 0 100px rgba(0,210,255,0.10);
    }
}
.info-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,210,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.info-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(123,47,247,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.info-banner .ib-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 0.8rem;
    animation: bounce-icon-cyan 1.4s ease-in-out infinite;
    filter: drop-shadow(0 0 16px rgba(0,210,255,0.7));
}
@keyframes bounce-icon-cyan {
    0%, 100% { transform: translateY(0) scale(1); }
    50%       { transform: translateY(-8px) scale(1.08); }
}
.info-banner .ib-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #00d2ff;
    display: block;
    margin-bottom: 0.7rem;
    letter-spacing: 0.04em;
    text-shadow: 0 0 20px rgba(0,210,255,0.6), 0 0 40px rgba(0,210,255,0.3);
}
.info-banner .ib-sub {
    font-size: 1.05rem;
    color: #b8d8f0;
    display: block;
    line-height: 1.75;
    max-width: 560px;
    margin: 0 auto;
}
.info-banner .ib-sub strong {
    color: #7b2ff7;
    font-weight: 700;
}
.info-banner .ib-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    border-radius: 4px;
    margin: 1.2rem auto 1rem;
}
.info-banner .ib-arrow {
    display: block;
    font-size: 2.4rem;
    color: #00d2ff;
    margin-top: 1rem;
    animation: arrow-bounce-cyan 1s ease-in-out infinite;
    text-shadow: 0 0 12px rgba(0,210,255,0.6);
}
@keyframes arrow-bounce-cyan {
    0%, 100% { transform: translateY(0); opacity: 1; }
    50%       { transform: translateY(8px); opacity: 0.55; }
}

/* ── BANNERS DE ACCIÓN ── */
.action-banner {
    background: linear-gradient(135deg, #161b22, #1a1f2e);
    border: 1px solid #00d2ff44;
    border-left: 4px solid #00d2ff;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.action-banner .icon { font-size: 1.5rem; }
.action-banner .text { }
.action-banner .text strong { color: #00d2ff; display: block; margin-bottom: 2px; }
.action-banner .text span { font-size: 0.82rem; color: #8b949e; }


/* ── BANNER DE ÉXITO (post-procesamiento) — VERSIÓN REFORZADA ── */
.success-banner {
    background: linear-gradient(135deg, #0a2e1a 0%, #071628 100%);
    border: 3px solid #00ff88;
    border-radius: 20px;
    padding: 2.8rem 2.4rem 2.4rem;
    margin: 1.8rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 6px rgba(0,255,136,0.08),
        0 0 60px rgba(0, 255, 136, 0.30),
        inset 0 0 80px rgba(0,255,136,0.06);
    animation: glow-pulse 2.2s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%, 100% {
        box-shadow:
            0 0 0 6px rgba(0,255,136,0.08),
            0 0 40px rgba(0,255,136,0.20),
            inset 0 0 80px rgba(0,255,136,0.04);
    }
    50% {
        box-shadow:
            0 0 0 8px rgba(0,255,136,0.14),
            0 0 80px rgba(0,255,136,0.45),
            inset 0 0 100px rgba(0,255,136,0.10);
    }
}
.success-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,255,136,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.success-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,210,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.success-banner .sb-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 0.8rem;
    animation: bounce-icon 1.4s ease-in-out infinite;
    filter: drop-shadow(0 0 16px rgba(0,255,136,0.7));
}
@keyframes bounce-icon {
    0%, 100% { transform: translateY(0) scale(1); }
    50%       { transform: translateY(-8px) scale(1.08); }
}
.success-banner .sb-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #00ff88;
    display: block;
    margin-bottom: 0.7rem;
    letter-spacing: 0.04em;
    text-shadow: 0 0 20px rgba(0,255,136,0.6), 0 0 40px rgba(0,255,136,0.3);
}
.success-banner .sb-sub {
    font-size: 1.05rem;
    color: #b8f0d8;
    display: block;
    line-height: 1.75;
    max-width: 560px;
    margin: 0 auto;
}
.success-banner .sb-sub strong {
    color: #00d2ff;
    font-weight: 700;
}
.success-banner .sb-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #00ff88, #00d2ff);
    border-radius: 4px;
    margin: 1.2rem auto 1rem;
}
.success-banner .sb-arrow {
    display: block;
    font-size: 2.4rem;
    color: #00ff88;
    margin-top: 1rem;
    animation: arrow-bounce 1s ease-in-out infinite;
    text-shadow: 0 0 12px rgba(0,255,136,0.6);
}
@keyframes arrow-bounce {
    0%, 100% { transform: translateY(0); opacity: 1; }
    50%       { transform: translateY(8px); opacity: 0.55; }
}

/* ── SECTION HEADERS ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #00d2ff;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 1.2rem;
}

/* ── CARDS ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-accent {
    background: linear-gradient(135deg, #161b22, #1a1f2e);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem;
    border-left: 3px solid #00d2ff;
    margin-bottom: 1rem;
}

/* ── STAT BADGES ── */
.stats-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.8rem 0; }
.stat-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.stat-badge .label { color: #8b949e; font-size: 0.65rem; display: block; }
.stat-badge .value { color: #00d2ff; font-weight: 700; }

/* ── FORMULA BOX ── */
.formula-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}

hr { border-color: #21262d !important; }

[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #00d2ff; }

.stRadio > div { gap: 12px; }
.stRadio label {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    cursor: pointer;
    transition: all 0.2s;
}
.stRadio label:hover { border-color: #00d2ff; color: #00d2ff; }

.stButton > button {
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.7em 1.5em;
    font-size: 0.9rem;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
img { image-rendering: pixelated; border-radius: 8px; }
.stWarning { border-radius: 10px; }
[data-testid="stSlider"] > div > div > div > div { background: #00d2ff !important; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB TEMA OSCURO ────────────────────────────────────────────────────
def dark_fig(figsize=(6, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    return fig, ax


def plot_hist(data, color='#00d2ff', label='', figsize=(6, 3)):
    fig, ax = dark_fig(figsize)
    ax.hist(data.ravel(), bins=256, range=[0, 256], color=color, alpha=0.85, label=label)
    ax.set_xlabel("Nivel de Intensidad")
    ax.set_ylabel("Frecuencia")
    if label:
        ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)
    fig.tight_layout()
    return fig


# ── FUNCIONES DE CÁLCULO ──────────────────────────────────────────────────────

def verificar_es_bw(img_array: NDArray[np.uint8]) -> bool:
    if len(img_array.shape) == 2:
        return True
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    return bool(np.array_equal(r, g) and np.array_equal(g, b))


def convertir_a_gris(img_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    gris = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    return np.round(gris).astype(np.uint8)


def expansion_manual(matrix, out_min, out_max):
    i_min, i_max = int(np.min(matrix)), int(np.max(matrix))
    if i_max == i_min:
        niveles = np.arange(256)
        orig_hist, _ = np.histogram(matrix, bins=256, range=[0, 256])
        df_proc = pd.DataFrame({
            "Intensidad Original": niveles,
            "Cantidad Original": orig_hist,
            "Cantidad Resultado": orig_hist,
            f"Mapeo y=1.0000x+0.0000": niveles,
        })
        return matrix.copy(), 1.0, 0.0, i_min, i_max, df_proc
    m = (out_max - out_min) / (i_max - i_min)
    b = out_min - m * i_min
    niveles = np.arange(256)
    lut = np.clip(np.round(m * niveles + b), 0, 255).astype(np.uint8)
    res = lut[matrix.astype(np.uint8)]
    orig_hist, _ = np.histogram(matrix, bins=256, range=[0, 256])
    res_hist, _ = np.histogram(res, bins=256, range=[0, 256])
    df_proc = pd.DataFrame({
        "Intensidad Original": niveles,
        "Cantidad Original": orig_hist,
        "Cantidad Resultado": res_hist,
        f"Mapeo y={m:.4f}x+{b:.4f}": lut,
    })
    return res, m, b, i_min, i_max, df_proc


def ecualizacion_manual(matrix):
    total_n = matrix.size
    hist, _ = np.histogram(matrix, bins=256, range=[0, 256])
    prob = hist / total_n
    cdf = prob.cumsum()
    nuevo_tono = np.round(255 * cdf).astype(np.uint8)
    df_proc = pd.DataFrame({
        'rk (Original)': range(256),
        'nk (Frecuencia)': hist,
        'Pr(rk) = nk/n': prob,
        'CDF(rk)': cdf,
        'sk = (L-1)·CDF': nuevo_tono,
    })
    return nuevo_tono[matrix.astype(np.uint8)], df_proc


def stat_badges(arr):
    return f"""
    <div class="stats-row">
      <div class="stat-badge"><span class="label">MIN</span><span class="value">{int(arr.min())}</span></div>
      <div class="stat-badge"><span class="label">MAX</span><span class="value">{int(arr.max())}</span></div>
      <div class="stat-badge"><span class="label">MEDIA</span><span class="value">{arr.mean():.1f}</span></div>
      <div class="stat-badge"><span class="label">STD</span><span class="value">{arr.std():.1f}</span></div>
      <div class="stat-badge"><span class="label">PÍXELES</span><span class="value">{arr.size:,}</span></div>
    </div>
    """


def stepper_html(step: int) -> str:
    steps = [
        ("1", "Cargar imagen"),
        ("2", "Analizar"),
        ("3", "Procesar"),
        ("4", "Comparar"),
    ]
    parts = []
    for i, (num, label) in enumerate(steps, start=1):
        if i < step:
            bubble_cls = "done"
            label_cls  = "done"
            icon = "✓"
        elif i == step:
            bubble_cls = "active"
            label_cls  = "active"
            icon = num
        else:
            bubble_cls = ""
            label_cls  = ""
            icon = num

        parts.append(f"""
        <div class="step">
          <div class="step-bubble {bubble_cls}">{icon}</div>
          <span class="step-label {label_cls}">{label}</span>
        </div>""")

        if i < len(steps):
            conn_cls = "done" if i < step else ""
            parts.append(f'<div class="step-connector {conn_cls}"></div>')

    return f'<div class="stepper">{"".join(parts)}</div>'


# ── ESTADO DE SESIÓN — reset COMPLETO al cambiar archivo ─────────────────────
def _full_reset():
    st.session_state.img_gris  = None
    st.session_state.img_res   = None
    st.session_state.df_proc   = None
    st.session_state.params    = {}

KEYS_DEFAULT = {
    "img_gris": None, "img_res": None,
    "df_proc": None,  "params": {},
}
for k, v in KEYS_DEFAULT.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'uploader_counter' not in st.session_state:
    st.session_state.uploader_counter = 0

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⬛ Procesamiento Matricial B&amp;W</h1>
  <p>Expansión lineal · Ecualización de histograma · Análisis de intensidades</p>
</div>
""", unsafe_allow_html=True)

# ── CARGA DE IMAGEN ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Paso 1 — Sube tu imagen para comenzar (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    key=f"file_uploader_{st.session_state.uploader_counter}",
    on_change=_full_reset,
)

# ── FLUJO PRINCIPAL ───────────────────────────────────────────────────────────
if not uploaded_file:
    st.markdown(stepper_html(1), unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem;">
      <div style="font-size:3rem;">⬛</div>
      <p style="color:#8b949e; margin-top:1rem;">
        Sube una imagen PNG o JPG usando el botón de arriba para comenzar el análisis.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

img_raw = np.array(Image.open(uploaded_file).convert("RGB"))

# PASO 1 — VERIFICACIÓN (solo B&W)
if st.session_state.img_gris is None:
    if not verificar_es_bw(img_raw):
        st.markdown(stepper_html(1), unsafe_allow_html=True)
        st.markdown("""
        <div class="info-banner" style="border-color: #ff4757; animation: glow-pulse-red 2.2s ease-in-out infinite;">
          <span class="ib-icon" style="filter: drop-shadow(0 0 16px rgba(255,71,87,0.7));">🚫</span>
          <span class="ib-title" style="color: #ff4757; text-shadow: 0 0 20px rgba(255,71,87,0.6), 0 0 40px rgba(255,71,87,0.3);">Imagen a color no permitida</span>
          <div class="ib-divider" style="background: linear-gradient(90deg, #ff4757, #ff6b81);"></div>
          <span class="ib-sub" style="color: #f0b8c0;">
            Esta herramienta solo acepta imágenes en <strong style="color:#ff6b81;">blanco y negro (escala de grises)</strong>.<br>
            Por favor, sube una imagen B&W para continuar.
          </span>
        </div>
        <style>
        @keyframes glow-pulse-red {
            0%, 100% {
                box-shadow:
                    0 0 0 6px rgba(255,71,87,0.08),
                    0 0 40px rgba(255,71,87,0.20),
                    inset 0 0 80px rgba(255,71,87,0.04);
            }
            50% {
                box-shadow:
                    0 0 0 8px rgba(255,71,87,0.14),
                    0 0 80px rgba(255,71,87,0.45),
                    inset 0 0 100px rgba(255,71,87,0.10);
            }
        }
        </style>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.session_state.img_gris = img_raw[:, :, 0].copy() if len(img_raw.shape) == 3 else img_raw.copy()

img_act = st.session_state.img_gris

# ── STEPPER: paso actual ──────────────────────────────────────────────────────
current_step = 4 if st.session_state.img_res is not None else 2
st.markdown(stepper_html(current_step), unsafe_allow_html=True)

# ── SECCIÓN 1: ANÁLISIS DE ENTRADA ───────────────────────────────────────────
st.markdown('<div class="section-label">Paso 1</div><div class="section-title">Análisis de Imagen de Entrada</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("📥 Imagen Original")
    st.image(img_act, caption=f"Resolución: {img_act.shape[1]}×{img_act.shape[0]} px", width='stretch')
    st.markdown(stat_badges(img_act), unsafe_allow_html=True)
    with st.expander("Ver matriz parcial (32×32)", expanded=True):
        st.dataframe(pd.DataFrame(img_act[:32, :32]), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("📈 Histograma de Intensidades")
    _fig = plot_hist(img_act, color='#00d2ff')
    st.pyplot(_fig)
    plt.close(_fig)
    h_d, _ = np.histogram(img_act.ravel(), bins=256, range=[0, 256])
    df_h = pd.DataFrame({'Tono': range(256), 'Frecuencia': h_d})
    with st.expander("Ver tabla de frecuencias", expanded=True):
        st.dataframe(df_h[df_h['Frecuencia'] > 0], width='stretch', height=200)
    st.markdown('</div>', unsafe_allow_html=True)

# ── HINT: bajar para continuar ────────────────────────────────────────────────
st.markdown("""
<div class="info-banner">
  <span class="ib-icon">👇</span>
  <span class="ib-title">¡Continúa aquí abajo!</span>
  <div class="ib-divider"></div>
  <span class="ib-sub">
    Selecciona el algoritmo y presiona <strong>Procesar</strong> para ver los resultados.
  </span>
  <span class="ib-arrow">⬇</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── SECCIÓN 2: PROCESAMIENTO ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Paso 2</div><div class="section-title">Configuración y Procesamiento</div>', unsafe_allow_html=True)

st.markdown("""
<div class="action-banner">
  <div class="icon">⚙️</div>
  <div class="text">
    <strong>Elige el algoritmo que quieras aplicar</strong>
    <span>Expansión lineal ajusta el rango de brillo. Ecualización redistribuye los tonos para mayor contraste.</span>
  </div>
</div>
""", unsafe_allow_html=True)

def _clear_results():
    st.session_state.img_res  = None
    st.session_state.df_proc  = None
    st.session_state.params   = {}

metodo = st.radio(
    "Seleccione algoritmo:",
    ["Expansión Lineal (y=mx+b)", "Ecualización Histograma"],
    horizontal=True,
    on_change=_clear_results,
)

if "Expansión" in metodo:
    col_sl1, col_sl2 = st.columns([3, 1])
    with col_sl1:
        min_d, max_d = st.slider(
            "Rango de salida deseado (valores mínimo y máximo de brillo en la imagen resultante):",
            0, 255, (0, 255)
        )
    with col_sl2:
        st.metric("Amplitud salida", f"{max_d - min_d + 1} niveles")
    if st.button("▶ PROCESAR — EXPANSIÓN LINEAL"):
        res, m, b, imin, imax, df = expansion_manual(img_act, min_d, max_d)
        st.session_state.img_res = res
        st.session_state.df_proc = df
        st.session_state.params  = {"t": "exp", "m": m, "b": b,
                                    "imin": imin, "imax": imax,
                                    "omin": min_d, "omax": max_d}
        st.rerun()
else:
    if st.button("▶ PROCESAR — ECUALIZACIÓN DE HISTOGRAMA"):
        res, df = ecualizacion_manual(img_act)
        st.session_state.img_res = res
        st.session_state.df_proc = df
        st.session_state.params  = {"t": "equ"}
        st.rerun()

# ── RESULTADOS ────────────────────────────────────────────────────────────────
if st.session_state.img_res is not None:
    img_res  = st.session_state.img_res
    df_proc  = st.session_state.df_proc
    p        = st.session_state.params

    # ── BANNER DE ÉXITO REFORZADO ─────────────────────────────────────────────
    st.markdown("""
    <div class="success-banner">
      <span class="sb-icon">✅</span>
      <span class="sb-title">¡Procesamiento completado!</span>
      <div class="sb-divider"></div>
      <span class="sb-sub">
        👇 &nbsp; <strong>Baja para ver:</strong> &nbsp;
        el <strong>desglose matemático</strong>,
        la <strong>imagen resultado</strong>
        y la <strong>comparación final</strong>
      </span>
      <span class="sb-arrow">⬇</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">Paso 3</div><div class="section-title">Desglose Matemático</div>', unsafe_allow_html=True)

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    if p["t"] == "exp":
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            st.markdown("**Fórmulas aplicadas**")
            st.latex(r"m = \frac{y_{max} - y_{min}}{x_{max} - x_{min}}")
            st.latex(r"b = y_{min} - m \cdot x_{min}")
            st.latex(rf"y = {p['m']:.4f} \cdot x + ({p['b']:.4f})")
        with fc2:
            st.markdown("**Valores calculados**")
            st.markdown(f"""
            <div class="stats-row">
              <div class="stat-badge"><span class="label">PENDIENTE m</span><span class="value">{p['m']:.6f}</span></div>
              <div class="stat-badge"><span class="label">INTERCEPTO b</span><span class="value">{p['b']:.6f}</span></div>
              <div class="stat-badge"><span class="label">ENTRADA [{p['imin']}, {p['imax']}]</span><span class="value">→</span></div>
              <div class="stat-badge"><span class="label">SALIDA [{p['omin']}, {p['omax']}]</span><span class="value">✓</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("**Ecualización de Histograma**")
        st.latex(r"s_k = (L-1) \sum_{j=0}^{k} P_r(r_j) = (L-1) \sum_{j=0}^{k} \frac{n_j}{n}")
        st.caption("rk: nivel original · nk: frecuencia · n: total píxeles · sk: nivel transformado")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 Tabla de mapeo completa (solo tonos presentes)", expanded=True):
        st.dataframe(df_proc[df_proc.iloc[:, 1] > 0], width='stretch', height=280)

    # ── ANÁLISIS DE IMAGEN OUTPUT ─────────────────────────────────────────────
    st.markdown('<div class="card-accent" style="margin-top:1.2rem;">', unsafe_allow_html=True)
    st.caption("📤 Análisis de imagen resultante (output)")
    out_col1, out_col2, out_col3 = st.columns([1, 1.4, 1], gap="medium")

    with out_col1:
        st.image(img_res, caption="Imagen output", width='stretch')
        st.markdown(stat_badges(img_res), unsafe_allow_html=True)

    with out_col2:
        _fig = plot_hist(img_res, color='#7b2ff7', label='Distribución output')
        st.pyplot(_fig)
        plt.close(_fig)

    with out_col3:
        h_out, _ = np.histogram(img_res.ravel(), bins=256, range=[0, 256])
        df_out   = pd.DataFrame({'Tono': range(256), 'n': h_out})
        st.caption("Tabla de frecuencias output")
        st.dataframe(df_out[df_out['n'] > 0].reset_index(drop=True),
                     width='stretch', height=250)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── COMPARACIÓN FINAL ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Paso 4</div><div class="section-title">Comparación de Resultados</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2, gap="medium")

    with col_r1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("🔵 ORIGINAL")
        st.image(img_act, width='stretch')
        st.markdown(stat_badges(img_act), unsafe_allow_html=True)
        _fig = plot_hist(img_act, color='#00d2ff', label='Original')
        st.pyplot(_fig)
        plt.close(_fig)
        with st.expander("Matriz 32×32", expanded=True):
            st.dataframe(pd.DataFrame(img_act[:32, :32]), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("🟣 RESULTADO")
        st.image(img_res, width='stretch')
        st.markdown(stat_badges(img_res), unsafe_allow_html=True)
        _fig = plot_hist(img_res, color='#7b2ff7', label='Resultado')
        st.pyplot(_fig)
        plt.close(_fig)
        with st.expander("Matriz 32×32", expanded=True):
            st.dataframe(pd.DataFrame(img_res[:32, :32]), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SUPERPOSICIÓN ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">Paso 5</div><div class="section-title">Superposición de Histogramas</div>', unsafe_allow_html=True)

    fig_sup, ax_sup = dark_fig(figsize=(12, 4))
    ax_sup.hist(img_act.ravel(), bins=256, range=[0, 256],
                color='#00d2ff', alpha=0.55, label='Original')
    ax_sup.hist(img_res.ravel(), bins=256, range=[0, 256],
                color='#7b2ff7', alpha=0.55, label='Resultado')
    ax_sup.set_title("Cambio en la Distribución de Frecuencias",
                      color='#c9d1d9', fontsize=11)
    ax_sup.set_xlabel("Nivel de Intensidad")
    ax_sup.set_ylabel("Frecuencia (n)")
    ax_sup.legend(facecolor='#21262d', edgecolor='#30363d',
                  labelcolor='#c9d1d9', fontsize=9)
    fig_sup.tight_layout()
    st.pyplot(fig_sup)
    plt.close(fig_sup)

    # ── REINICIAR ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="action-banner" style="border-left-color: #7b2ff7;">
      <div class="icon">🔁</div>
      <div class="text">
        <strong>¿Quieres probar con otra imagen o algoritmo?</strong>
        <span>Sube una nueva imagen arriba o usa el botón para limpiar y empezar de nuevo.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗑️ Limpiar y empezar de nuevo"):
        counter = st.session_state.get('uploader_counter', 0) + 1
        st.session_state.clear()
        st.session_state.uploader_counter = counter
        st.rerun()
