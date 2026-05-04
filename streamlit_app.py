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

/* ── FONDO Y TIPOGRAFÍA BASE ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* ── HEADER CUSTOM ── */
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

/* ── DIVIDER ── */
hr { border-color: #21262d !important; }

/* ── WIDGETS STREAMLIT ── */
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

/* ── DATAFRAMES ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── IMÁGENES ── */
img { image-rendering: pixelated; border-radius: 8px; }

/* ── WARNINGS ── */
.stWarning { border-radius: 10px; }

/* ── SLIDERS ── */
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
        return matrix.copy(), 1.0, 0.0, i_min, i_max, pd.DataFrame()
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


# ── ESTADO DE SESIÓN ──────────────────────────────────────────────────────────
for k, v in [("img_gris", None), ("img_res", None), ("df_proc", None), ("params", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⬛ Procesamiento Matricial B&amp;W</h1>
  <p>Expansión lineal · Ecualización de histograma · Análisis de intensidades</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Cargar imagen (PNG / JPG)", type=["png", "jpg", "jpeg"])

# ── FLUJO PRINCIPAL ───────────────────────────────────────────────────────────
if uploaded_file:
    if "fn" not in st.session_state or st.session_state.fn != uploaded_file.name:
        st.session_state.img_gris = None
        st.session_state.img_res = None
        st.session_state.fn = uploaded_file.name

    img_raw = np.array(Image.open(uploaded_file).convert("RGB"))

    # PASO 1 — VERIFICACIÓN / CONVERSIÓN
    if st.session_state.img_gris is None:
        if not verificar_es_bw(img_raw):
            st.warning("⚠️ Imagen a color detectada — se requiere conversión.")
            st.image(img_raw, width=300)
            if st.button("Convertir a Blanco y Negro"):
                st.session_state.img_gris = convertir_a_gris(img_raw)
                st.rerun()
            st.stop()
        else:
            st.session_state.img_gris = img_raw[:, :, 0].copy() if len(img_raw.shape) == 3 else img_raw.copy()

    img_act = st.session_state.img_gris

    # ── SECCIÓN 1: ANÁLISIS DE ENTRADA ───────────────────────────────────────
    st.markdown('<div class="section-label">Paso 1</div><div class="section-title">Análisis de Imagen de Entrada</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("📥 Imagen Original")
        st.image(img_act, caption=f"Resolución: {img_act.shape[1]}×{img_act.shape[0]} px", width='stretch')
        st.markdown(stat_badges(img_act), unsafe_allow_html=True)
        with st.expander("Ver matriz parcial (15×15)"):
            st.dataframe(pd.DataFrame(img_act[:15, :15]), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("📈 Histograma de Intensidades")
        st.pyplot(plot_hist(img_act, color='#00d2ff'))
        h_d, _ = np.histogram(img_act.ravel(), bins=256, range=[0, 256])
        df_h = pd.DataFrame({'Tono': range(256), 'Frecuencia': h_d})
        with st.expander("Ver tabla de frecuencias"):
            st.dataframe(df_h[df_h['Frecuencia'] > 0], width='stretch', height=200)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── SECCIÓN 2: PROCESAMIENTO ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Paso 2</div><div class="section-title">Configuración y Procesamiento</div>', unsafe_allow_html=True)

    # Selector de algoritmo
    metodo = st.radio(
        "Seleccione algoritmo:",
        ["Expansión Lineal (y=mx+b)", "Ecualización Histograma"],
        horizontal=True,
    )

    if "Expansión" in metodo:
        col_sl1, col_sl2 = st.columns([3, 1])
        with col_sl1:
            min_d, max_d = st.slider("Rango de salida deseado:", 0, 255, (0, 255))
        with col_sl2:
            st.metric("Amplitud salida", f"{max_d - min_d + 1} niveles")
        if st.button("▶ PROCESAR — EXPANSIÓN LINEAL"):
            res, m, b, imin, imax, df = expansion_manual(img_act, min_d, max_d)
            st.session_state.img_res = res
            st.session_state.df_proc = df
            st.session_state.params = {"t": "exp", "m": m, "b": b,
                                       "imin": imin, "imax": imax,
                                       "omin": min_d, "omax": max_d}
    else:
        if st.button("▶ PROCESAR — ECUALIZACIÓN DE HISTOGRAMA"):
            res, df = ecualizacion_manual(img_act)
            st.session_state.img_res = res
            st.session_state.df_proc = df
            st.session_state.params = {"t": "equ"}

    # ── RESULTADOS ────────────────────────────────────────────────────────────
    if st.session_state.img_res is not None:
        img_res = st.session_state.img_res
        df_proc = st.session_state.df_proc
        p = st.session_state.params

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

        with st.expander("📋 Tabla de mapeo completa (solo tonos presentes)"):
            st.dataframe(df_proc[df_proc.iloc[:, 1] > 0], width='stretch', height=280)

        # ── ANÁLISIS DE IMAGEN OUTPUT ─────────────────────────────────────────
        st.markdown('<div class="card-accent" style="margin-top:1.2rem;">', unsafe_allow_html=True)
        st.caption("📤 Análisis de imagen resultante (output)")
        out_col1, out_col2, out_col3 = st.columns([1, 1.4, 1], gap="medium")

        with out_col1:
            st.image(img_res, caption="Imagen output", width='stretch')
            st.markdown(stat_badges(img_res), unsafe_allow_html=True)

        with out_col2:
            st.pyplot(plot_hist(img_res, color='#7b2ff7', label='Distribución output'))

        with out_col3:
            h_out, _ = np.histogram(img_res.ravel(), bins=256, range=[0, 256])
            df_out = pd.DataFrame({'Tono': range(256), 'n': h_out})
            st.caption("Tabla de frecuencias output")
            st.dataframe(df_out[df_out['n'] > 0].reset_index(drop=True),
                         width='stretch', height=250)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── COMPARACIÓN FINAL ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Paso 4</div><div class="section-title">Comparación de Resultados</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2, gap="medium")

        with col_r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.caption("🔵 ORIGINAL")
            st.image(img_act, width='stretch')
            st.markdown(stat_badges(img_act), unsafe_allow_html=True)
            st.pyplot(plot_hist(img_act, color='#00d2ff', label='Original'))
            with st.expander("Matriz 15×15"):
                st.dataframe(pd.DataFrame(img_act[:15, :15]), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.caption("🟣 RESULTADO")
            st.image(img_res, width='stretch')
            st.markdown(stat_badges(img_res), unsafe_allow_html=True)
            st.pyplot(plot_hist(img_res, color='#7b2ff7', label='Resultado'))
            with st.expander("Matriz 15×15"):
                st.dataframe(pd.DataFrame(img_res[:15, :15]), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        # ── SUPERPOSICIÓN ─────────────────────────────────────────────────────
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

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem;">
      <div style="font-size:3rem;">⬛</div>
      <p style="color:#8b949e; margin-top:1rem;">Sube una imagen PNG o JPG para comenzar el análisis matricial.</p>
    </div>
    """, unsafe_allow_html=True)