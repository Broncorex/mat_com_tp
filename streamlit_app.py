from typing import Tuple

import streamlit as st
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Ecualizador y expandir imágenes", layout="wide")

st.markdown("""
    <style>
    img { image-rendering: pixelated; border: 1px solid #ccc; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CÁLCULO OPTIMIZADAS ---

def verificar_es_bw(img_array: NDArray[np.uint8]) -> bool:
    if len(img_array.shape) == 2: return True
    r: NDArray[np.uint8] = img_array[:,:,0]
    g: NDArray[np.uint8] = img_array[:,:,1]
    b: NDArray[np.uint8] = img_array[:,:,2]
    return bool(np.array_equal(r, g) and np.array_equal(g, b))

def convertir_a_gris(img_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    gris: NDArray[np.float64] = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    return np.round(gris).astype(np.uint8)

def expansion_manual(
    matrix: NDArray[np.uint8],
    out_min: int,
    out_max: int
) -> Tuple[NDArray[np.uint8], float, float, int | float, int | float, pd.DataFrame]:
    """Optimizado mediante LUT (Look-Up Table)"""
    i_min: int | float = np.min(matrix)
    i_max: int | float = np.max(matrix)
    if i_max == i_min:
        return matrix, 0.0, 0.0, i_min, i_max

    # Calcular m y b una sola vez
    m: float = (out_max - out_min) / (i_max - i_min)
    b: float = out_min - m * i_min

    # OPTIMIZACIÓN: Crear una tabla para los 256 niveles posibles
    niveles: NDArray[np.intp] = np.arange(256)
    # Aplicar fórmula solo a los 256 niveles
    lut: NDArray[np.uint8] = np.clip(np.round(m * niveles + b), 0, 255).astype(np.uint8)

    # Transformar la imagen completa usando la tabla como índice
    res: NDArray[np.uint8] = lut[matrix.astype(np.uint8)]

    original_histograma, _ = np.histogram(matrix.flatten(), bins=256, range=[0, 256])
    output_histograma, _ = np.histogram(res.flatten(), bins=256, range=[0, 256])

    df_proc = pd.DataFrame({
        "Intensidad Original": np.arange(256),
        "Cantidad Original": original_histograma,
        "Cantidad Resultado": output_histograma,
        f"Intensidad Resultado {np.round(m,4)}x + b": lut,

    })
    return res, m, b, i_min, i_max, df_proc

def ecualizacion_manual(
    matrix: NDArray[np.uint8]
) -> Tuple[NDArray[np.uint8], pd.DataFrame]:
    """Optimizado mediante LUT y mapeo de histograma"""
    total_n: int = matrix.size
    hist: NDArray[np.intp]
    hist, _ = np.histogram(matrix.flatten(), bins=256, range=[0, 256])
    prob: NDArray[np.float64] = hist / total_n
    cdf: NDArray[np.float64] = prob.cumsum()

    # OPTIMIZACIÓN: Crear la LUT (sk) para los 256 niveles
    nuevo_tono: NDArray[np.uint8] = np.round(255 * cdf).astype(np.uint8)

    df_procedimiento: pd.DataFrame = pd.DataFrame({
        'rk (Original)': range(256),
        'nk (Frecuencia)': hist,
        'Pr(rk) = nk / n': prob,
        'CDF(rk) = Σ Pr': cdf,
        'sk = (L-1) * CDF': nuevo_tono
    })

    # Mapear la imagen usando la LUT
    res_img: NDArray[np.uint8] = nuevo_tono[matrix.astype(np.uint8)]
    return res_img, df_procedimiento

# --- FLUJO DE LA APLICACIÓN ---

st.title("🛠️ Herramienta de Procesamiento Matricial B&W")

uploaded_file = st.file_uploader("Cargar imagen (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Reset automático si cambia el archivo
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.img_gris = None
        st.session_state.current_file_name = uploaded_file.name

    img_pil = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(img_pil)
    
    # PASO 1: VERIFICACIÓN
    es_bw_actual = verificar_es_bw(img_arr)
    
    if not es_bw_actual and st.session_state.img_gris is None:
        st.warning("⚠️ IMAGEN A COLOR DETECTADA")
        st.image(img_arr, caption="Imagen Original (Color)", width=400)
        if st.button("Convertir a Blanco y Negro ahora"):
            st.session_state.img_gris = convertir_a_gris(img_arr)
            st.rerun()
        st.stop()
    
    if st.session_state.img_gris is None:
        st.session_state.img_gris = img_arr[:,:,0]
        
    img_act = st.session_state.img_gris
    
    # PASO 2: ANÁLISIS
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Datos de Entrada")
        st.image(img_act, caption="Imagen en Gris", width=300)
        st.write("Matriz de Intensidad (15x15):")
        st.dataframe(img_act[:15, :15])
        
    with col2:
        st.subheader("📈 Histograma Inicial")
        fig_orig, ax_orig = plt.subplots(figsize=(5, 3))
        ax_orig.hist(img_act.ravel(), bins=256, color='gray', range=[0,256])
        st.pyplot(fig_orig)
        plt.close(fig_orig)
        
        hist_data, _ = np.histogram(img_act.ravel(), bins=256, range=[0,256])
        df_h = pd.DataFrame({'Tono': range(256), 'Frecuencia': hist_data})
        st.dataframe(df_h[df_h['Frecuencia'] > 0], height=150)

    st.divider()

    # PASO 3: PROCESAMIENTO
    metodo = st.radio("Seleccione Algoritmo:", ["Expansión Lineal (y=mx+b)", "Ecualización Histograma"])
    img_res = None
    
    if "Expansión" in metodo:
        st.subheader("⚙️ Configuración: Expansión Lineal")
        min_dest, max_dest = st.slider("Rango de salida deseado:", 0, 255, (0, 255))
        
        if st.button("Procesar Expansión"):
            img_res, m, b, i_min, i_max, df_proc = expansion_manual(img_act, min_dest, max_dest)
            
            with st.container(border=True):
                st.markdown("### 📊 Desglose Matemático (LUT)")
                st.markdown("Se ha calculado la transformación para cada uno de los 256 niveles de gris:")
                st.latex(r"m = \frac{y_{max} - y_{min}}{x_{max} - x_{min}}")
                st.latex(r"b = y_{min} - m \cdot x_{min}")
                
                st.markdown(f"""
                **Valores obtenidos:**
                * **m:** {m:.4f} | **b:** {b:.4f}
                * **Rango Original:** [{i_min}, {i_max}] -> **Rango Destino:** [{min_dest}, {max_dest}]
                """)
                st.latex(r"y = %.4f \cdot x + (%.4f)" % (m, b))
                st.dataframe(df_proc)

    else:
        st.subheader("⚙️ Configuración: Ecualización")
        if st.button("Procesar Ecualización"):
            img_res, df_proc = ecualizacion_manual(img_act)
            
            st.write("#### Tabla de Procedimiento (Mapeo de Intensidades)")
            st.dataframe(df_proc[df_proc['nk (Frecuencia)'] > 0], use_container_width=True)
            
            with st.container(border=True):
                st.markdown("### ℹ️ Significado de las Variables")
                st.markdown("""
                * **$r_k$**: Tono original.
                * **$P_r(r_k)$**: Frecuencia relativa.
                * **$CDF(r_k)$**: Probabilidad acumulada (Integral del histograma).
                * **$s_k$**: Nuevo tono calculado. La imagen se transforma reemplazando cada $r_k$ por su correspondiente $s_k$.
                """)

    # PASO 4: RESULTADOS
    if img_res is not None:
        st.divider()
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.subheader("📤 Resultado Final")
            st.image(img_res, caption="Imagen Modificada", width=300)
            st.write("Nueva Matriz (15x15):")
            st.dataframe(img_res[:15, :15])
        with col_out2:
            st.subheader("📊 Nuevo Histograma")
            fig_res, ax_res = plt.subplots(figsize=(5, 3))
            ax_res.hist(img_res.ravel(), bins=256, color='blue', range=[0,256])
            st.pyplot(fig_res)
            plt.close(fig_res)
            
            hist_res_data, _ = np.histogram(img_res.ravel(), bins=256, range=[0,256])
            df_hr = pd.DataFrame({'Tono': range(256), 'Frecuencia': hist_res_data})
            st.dataframe(df_hr[df_hr['Frecuencia'] > 0], height=150)

        # PASO 5: Comparador
        st.divider()
        col_com1, col_com2 = st.columns(2)

        with col_com1:
            st.subheader("Original")
            st.image(img_act, caption="Imagen Original", width=300)
            fig_orig, ax_orig = plt.subplots(figsize=(5, 3))
            ax_orig.hist(img_act.ravel(), bins=256, color='blue', range=[0, 256])
            st.pyplot(fig_orig)
            plt.close(fig_orig)

        with col_com2:
            st.subheader("Resultado Final")
            st.image(img_res, caption="Imagen Resultado", width=300)
            fig_res, ax_res = plt.subplots(figsize=(5, 3))
            ax_res.hist(img_res.ravel(), bins=256, color='red', range=[0, 256])
            st.pyplot(fig_res)
            plt.close(fig_res)

        #PASO 6: Comparador de histogramas
        st.subheader("Comparación de Histogramas")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Histograma Original (Azul)
        ax.hist(img_act.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='Original')

        # Histograma Resultado (Rojo)
        ax.hist(img_res.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label='Resultado')

        # Configuración de la gráfica
        ax.set_title("Superposición de Histogramas")
        ax.set_xlabel("Intensidad de Píxel")
        ax.set_ylabel("Frecuencia")
        ax.legend(loc='upper right')  # Añade la leyenda para distinguir los colores

        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("👋 Por favor, carga una imagen para comenzar.")