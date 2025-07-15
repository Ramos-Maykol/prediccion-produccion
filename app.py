import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF
import unicodedata
import os
import datetime
from scipy.stats import t
import re
from streamlit_modal import Modal
# --- Importar las traducciones desde el archivo translations.py ---
from translations import TRANSLATIONS

# --- Inicialización del idioma en session_state ---
if 'language' not in st.session_state:
    st.session_state.language = 'es' # Idioma por defecto: español

def get_text(key, **kwargs):
    """
    Obtiene el texto traducido para una clave dada y el idioma actual.
    Permite formatear cadenas con f-string like syntax usando kwargs.
    """
    lang = st.session_state.language
    text = TRANSLATIONS.get(key, {}).get(lang, f"MISSING_TRANSLATION_{key}")
    return text.format(**kwargs) if kwargs else text

# --- Crear carpetas si no existen ---
os.makedirs("img", exist_ok=True)
os.makedirs("reporte", exist_ok=True)

# ------------------------- UTILIDADES -------------------------
def limpiar_texto(texto):
    if isinstance(texto, str):
        # Normalizar a la forma NFD para descomponer caracteres acentuados
        texto = unicodedata.normalize('NFD', texto)
        # Codificar a ASCII, ignorando los errores (eliminará los acentos y caracteres especiales)
        texto = texto.encode('ascii', 'ignore').decode('utf-8')
        # Eliminar cualquier otro carácter no ASCII que pudiera quedar o caracteres especiales
        texto = re.sub(r'[^\x00-\x7F]+', '', texto)
        return texto
    return str(texto)


def guardar_grafico_pred_vs_real(y_true, y_pred, modelo_nombre, filename, get_text_func):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6,
                color='teal' if modelo_nombre == "ANN" else 'orange' if modelo_nombre == "Random Forest" else 'darkorange')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label=get_text_func("pdf_plot_line_ideal"))
    plt.xlabel(get_text_func('pdf_plot_xlabel_real'))
    plt.ylabel(get_text_func('pdf_plot_ylabel_pred'))
    plt.title(f"{modelo_nombre}: {get_text_func('pdf_plot_title_pred_vs_real')}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def coeficiente_u_theil(y_true, y_pred):
    # Asegurarse de que no haya ceros en el denominador para evitar errores
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.any(y_true == 0):
        # Sumar una pequeña constante para evitar división por cero si hay ceros en y_true
        # Esto es una aproximación y podría ser necesario ajustarla
        y_true = y_true + 1e-6
    if np.any(y_pred == 0):
        y_pred = y_pred + 1e-6

    mse_forecast = np.mean((y_pred - y_true)**2)
    mse_naive = np.mean(np.diff(y_true)**2) # Para el pronóstico ingenuo (y_t - y_{t-1})

    if mse_naive == 0:
        return 0 # Si el pronóstico ingenuo no tiene error, U de Theil es 0 (predicción perfecta)
    return np.sqrt(mse_forecast / mse_naive)

def prueba_diebold_mariano(y_true, y_pred1, y_pred2, h=1, power=2):
    """
    Realiza la prueba de Diebold-Mariano para comparar la precisión de dos pronósticos.
    h: horizonte de pronóstico (por defecto 1)
    power: potencia del error (por defecto 2 para MSE)
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = np.abs(e1)**power - np.abs(e2)**power

    n = len(d)
    if n <= h:
        return np.nan, np.nan # No hay suficientes observaciones para el horizonte

    d_mean = np.mean(d)

    # Calcular la varianza de largo plazo usando la aproximación de Newey-West
    # para manejar la autocorrelación de d
    gamma0 = np.sum((d - d_mean)**2) / n
    gammas = [(np.sum((d[j:] - d_mean) * (d[:-j] - d_mean)) / n) for j in range(1, h)]
    long_run_variance = gamma0 + 2 * np.sum(gammas)

    if long_run_variance <= 0:
        # Puede ocurrir si d es constante o casi constante, o si h es demasiado grande
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(long_run_variance / n)
    p_value = 2 * t.cdf(-np.abs(dm_stat), df=n - 1)
    return dm_stat, p_value

# Modificación clave: Aceptar 'get_text_func'
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_text_func = None # Se asignará al inicializar desde la app

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, limpiar_texto(self.get_text_func("pdf_comparative_report_title")), 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, limpiar_texto(self.get_text_func("pdf_footer")), 0, 0, 'C')

def generar_pdf_resultados(mae_vals, mse_vals, r2_vals, modelos, mejor_modelo, tiempos_entrenamiento, y, modelo_ann, modelo_rf, modelo_xgb, X_scaled, get_text_func):
    pdf = PDF()
    pdf.get_text_func = get_text_func # Asignar la función de traducción
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, limpiar_texto(get_text_func("pdf_comparative_intro")))
    pdf.ln(5)

    # 1. Detalles de la máquina de entrenamiento
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_machine_details_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(get_text_func("pdf_machine_details_intro")))
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, get_text_func("pdf_cpu_label"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_cpu_desc")))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, get_text_func("pdf_ram_label"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_ram_desc")))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, get_text_func("pdf_gpu_label"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_gpu0_desc")))
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_gpu1_desc")))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, get_text_func("pdf_storage_label"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_storage0_desc")))
    pdf.multi_cell(0, 6, limpiar_texto(get_text_func("pdf_storage1_desc")))
    pdf.ln(5)

    # 2. Visualizaciones Exploratorias (EDA)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_eda_header")), ln=1)
    pdf.ln(2)

    eda_images = {
        "hist_tiempo_produccion.png": get_text_func("pdf_eda_hist"),
        "boxplot_tipo_producto.png": get_text_func("pdf_eda_boxplot"),
        "matriz_correlacion.png": get_text_func("pdf_eda_corr"),
        "unidades_vs_tiempo.png": get_text_func("pdf_eda_scatter")
    }

    for img_name, img_desc in eda_images.items():
        img_path = f"img/{img_name}"
        if os.path.exists(img_path):
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 7, limpiar_texto(img_desc), ln=1, align="C")
            pdf.image(img_path, x=pdf.get_x() + 30, w=150)
            pdf.ln(5)
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 7, limpiar_texto(get_text_func("pdf_eda_img_not_found", img_name=img_name)), ln=1, align="C")
            pdf.ln(2)

    # 3. Preprocesamiento de Datos
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_preprocessing_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    # Usar multi_cell para texto con saltos de línea
    pdf.multi_cell(0, 7, limpiar_texto(get_text_func("pdf_preprocessing_desc")))
    pdf.ln(5)

    # 4. Comparación de Modelos (MAE, MSE, R², Tiempo)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_model_comparison_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(get_text_func("pdf_model_comparison_intro")))
    pdf.ln(3)

    # Tabla de métricas
    pdf.set_fill_color(200, 220, 255) # Color de fondo para encabezados
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 8, limpiar_texto(get_text_func("pdf_table_model")), 1, 0, 'C', 1)
    pdf.cell(25, 8, "MAE", 1, 0, 'C', 1)
    pdf.cell(25, 8, "MSE", 1, 0, 'C', 1)
    pdf.cell(20, 8, "R\u00b2", 1, 0, 'C', 1) # R² directamente
    pdf.cell(40, 8, limpiar_texto(get_text_func("pdf_table_time_s")), 1, 1, 'C', 1) # Usar get_text_func
    pdf.set_font("Arial", "", 10)
    pdf.set_fill_color(240, 248, 255) # Color de fondo para filas de datos

    tiempos_tabla = [tiempos_entrenamiento.get(m, "N/A") for m in modelos]

    for i, model in enumerate(modelos):
        pdf.cell(45, 8, limpiar_texto(model), 1, 0, 'L', 0)
        pdf.cell(25, 8, f"{mae_vals[i]:.4f}", 1, 0, 'C', 0)
        pdf.cell(25, 8, f"{mse_vals[i]:.4f}", 1, 0, 'C', 0)
        pdf.cell(20, 8, f"{r2_vals[i]:.4f}", 1, 0, 'C', 0)
        pdf.cell(40, 8, f"{tiempos_tabla[i]:.2f}" if isinstance(tiempos_tabla[i], (int, float)) else limpiar_texto(tiempos_tabla[i]), 1, 1, 'C', 0)
    pdf.ln(5)

    # Gráficos Predicción vs Real
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_pred_vs_real_header")), ln=1)
    pdf.ln(2)

    preds = [modelo_ann.predict(X_scaled).flatten(), modelo_rf.predict(X_scaled), modelo_xgb.predict(X_scaled)]

    for i, (nombre, pred) in enumerate(zip(modelos, preds)):
        img_path = f"img/pred_vs_real_{nombre}.png"
        guardar_grafico_pred_vs_real(y, pred, nombre, img_path, get_text_func) # Pasar get_text_func aquí también
        if os.path.exists(img_path):
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 7, limpiar_texto(f"{nombre}: {get_text_func('pdf_plot_title_pred_vs_real')}"), ln=1, align="C")
            pdf.image(img_path, x=pdf.get_x() + 30, w=150)
            pdf.ln(5)
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 7, limpiar_texto(get_text_func("pdf_eda_img_not_found", img_name=img_path)), ln=1, align="C")
            pdf.ln(2)

    # 5. Coeficiente U de Theil
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_theil_header")), ln=1)
    pdf.ln(3)

    theil_vals = [coeficiente_u_theil(y, pred) for pred in preds]
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 8, limpiar_texto(get_text_func("pdf_table_model")), 1, 0, 'C', 1)
    pdf.cell(45, 8, limpiar_texto(get_text_func("pdf_theil_table_u_theil")), 1, 1, 'C', 1)
    pdf.set_font("Arial", "", 10)
    pdf.set_fill_color(240, 248, 255)
    for i, model in enumerate(modelos):
        pdf.cell(45, 8, limpiar_texto(model), 1, 0, 'L', 0)
        pdf.cell(45, 8, f"{theil_vals[i]:.4f}", 1, 1, 'C', 0)
    pdf.ln(5)

    # 6. Pruebas de Diebold-Mariano
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_diebold_header")), ln=1)
    pdf.ln(3)

    comparaciones = [
        ("ANN vs Random Forest", *prueba_diebold_mariano(y, y_pred_ann, y_pred_rf)),
        ("ANN vs XGBoost", *prueba_diebold_mariano(y, y_pred_ann, y_pred_xgb)),
        ("Random Forest vs XGBoost", *prueba_diebold_mariano(y, y_pred_rf, y_pred_xgb))
    ]

    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(70, 8, limpiar_texto(get_text_func("pdf_diebold_table_comparison")), 1, 0, 'C', 1)
    pdf.cell(40, 8, limpiar_texto(get_text_func("pdf_diebold_table_statistic")), 1, 0, 'C', 1)
    pdf.cell(40, 8, limpiar_texto(get_text_func("pdf_diebold_table_p_value")), 1, 1, 'C', 1)
    pdf.set_font("Arial", "", 10)
    pdf.set_fill_color(240, 248, 255)
    for comp_name, stat, p_val in comparaciones:
        pdf.cell(70, 8, limpiar_texto(comp_name), 1, 0, 'L', 0)
        pdf.cell(40, 8, f"{stat:.4f}" if not np.isnan(stat) else "N/A", 1, 0, 'C', 0)
        pdf.cell(40, 8, f"{p_val:.4f}" if not np.isnan(p_val) else "N/A", 1, 1, 'C', 0)
    pdf.ln(5)

    # 7. Conclusión
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_conclusion_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(get_text_func("pdf_conclusion_text", best_model_name=mejor_modelo)))
    pdf.ln(5)

    pdf_path = "reporte/reporte_comparativo_modelos.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Modificación clave: Aceptar 'get_text_func'
def generar_pdf_prediccion_individual(datos_entrada, modelo_sel, prediccion, otras_predicciones, get_text_func):
    horas_por_dia = 8
    cantidad = datos_entrada.get('Units Produced', 1)
    tiempo_total = prediccion * cantidad
    dias = int(np.ceil(tiempo_total / horas_por_dia))

    hoy = datetime.datetime.now()
    fecha_recojo = hoy + datetime.timedelta(days=dias-1 if dias > 0 else 0)
    fecha_recojo_str = fecha_recojo.strftime('%Y-%m-%d')

    # Traducción de los mensajes de recojo
    if tiempo_total <= horas_por_dia:
        recojo_msg = get_text_func("pdf_recojo_same_day")
    else:
        recojo_msg = get_text_func("pdf_recojo_days", days=dias, date=fecha_recojo_str)

    # Traducción del tipo de producto para el PDF
    # Asume que datos_entrada tiene las columnas 'Product Type_...'
    original_product_type = ""
    if datos_entrada.get('Product Type_Automotive') == 1:
        original_product_type = "Automotive"
    elif datos_entrada.get('Product Type_Electronics') == 1:
        original_product_type = "Electronics"
    elif datos_entrada.get('Product Type_Furniture') == 1:
        original_product_type = "Furniture"
    elif datos_entrada.get('Product Type_Textiles') == 1:
        original_product_type = "Textiles"
    # Ahora busca la traducción correcta basada en el idioma actual del PDF
    tipo_producto_display_pdf = get_text_func(f"pdf_product_type_option_{original_product_type}")

    volumen = datos_entrada.get('Production Volume Cubic Meters', '-')

    pdf_path = "reporte/reporte_prediccion_individual.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Portada
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(40, 60, 120)
    pdf.cell(0, 15, limpiar_texto(get_text_func("pdf_individual_report_title")), ln=1, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"{get_text_func('pdf_individual_report_date_prefix')} {hoy.strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(8)

    # Datos de entrada relevantes
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, get_text_func("pdf_input_data_header"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.set_fill_color(240, 240, 240)

    pdf.cell(60, 8, limpiar_texto(get_text_func("pdf_product_type_label")), 1, 0, 'L', 1)
    pdf.cell(60, 8, limpiar_texto(tipo_producto_display_pdf), 1, 1, 'L', 0)

    pdf.cell(60, 8, limpiar_texto(get_text_func("pdf_volume_label")), 1, 0, 'L', 1)
    pdf.cell(60, 8, str(volumen), 1, 1, 'L', 0)

    pdf.cell(60, 8, limpiar_texto(get_text_func("pdf_quantity_label")), 1, 0, 'L', 1)
    pdf.cell(60, 8, str(cantidad), 1, 1, 'L', 0)

    pdf.cell(60, 8, limpiar_texto(get_text_func("pdf_time_per_unit_label")), 1, 0, 'L', 1)
    pdf.cell(60, 8, f"{prediccion:.2f}", 1, 1, 'L', 0)

    pdf.cell(60, 8, limpiar_texto(get_text_func("pdf_total_time_label")), 1, 0, 'L', 1)
    pdf.cell(60, 8, f"{tiempo_total:.2f}", 1, 1, 'L', 0)

    pdf.ln(5)

    # Predicción principal
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_prediction_header")), ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"{get_text_func('pdf_model_selected_prefix')} {modelo_sel}\n"
                            f"{get_text_func('pdf_time_per_unit_estimated', time=prediccion)}")
    pdf.ln(3)

    # Otras predicciones
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, limpiar_texto(get_text_func("pdf_comparison_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    for modelo, valor in otras_predicciones.items():
        pdf.cell(0, 8, f"{modelo}: {valor:.2f} {get_text_func('pdf_time_per_unit_label')}", ln=1)
    pdf.ln(5)

    # Recomendación de recojo
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 8, limpiar_texto(get_text_func("pdf_recommendation_header")), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, limpiar_texto(recojo_msg))
    pdf.ln(5)

    # Pie de página institucional
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, limpiar_texto(get_text_func("pdf_footer")), 0, 0, 'C')

    pdf.output(pdf_path)
    return pdf_path

# ------------------------- CONFIGURACIÓN DE PÁGINA -------------------------
st.set_page_config(
    page_title=get_text("app_title"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Selector de idioma en la barra lateral ---
st.sidebar.title(get_text("sidebar_config_title"))
st.sidebar.header(get_text("sidebar_lang_header"))
lang_options_display = {
    "Español 🇪🇸": 'es',
    "English 🇬🇧": 'en'
}
selected_lang_key = st.sidebar.radio(
    "",
    options=list(lang_options_display.keys()),
    index=list(lang_options_display.values()).index(st.session_state.language),
    key="language_selector_radio"
)
st.session_state.language = lang_options_display[selected_lang_key]


# Custom CSS (sin cambios significativos, ya que son estilos)
st.markdown(
    """
    <style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; padding-right: 2rem; padding-left: 2rem; padding-bottom: 2rem; }
    h1 { color: #2e7d32; font-size: 2.5em; margin-bottom: 0.5em; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); display: inline-block; vertical-align: middle; }
    h2, h3, h4 { color: #1a5632; margin-top: 1.5em; margin-bottom: 0.8em; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; font-weight: bold; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 0.8em 1.5em; border-radius: 0.5em; border: none; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); transition: all 0.3s ease-in-out; }
    .stButton>button:hover { background-color: #45a049; transform: translateY(-2px); box-shadow: 3px 3px 8px rgba(0,0,0,0.3); }
    .css-1aumx8q { background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stAlert { border-radius: 8px; }
    .stProgress > div > div > div > div { background-color: #4CAF50 !important; }

    .hardware-button-style { background-color: #0d47a1; color: white; padding: 0.6em 1.2em; border-radius: 0.5em; border: none; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); transition: all 0.3s ease-in-out; margin-left: 15px; vertical-align: middle; display: inline-block; cursor: pointer; }
    .hardware-button-style:hover { background-color: #0a3d92; transform: translateY(-1px); box-shadow: 3px 3px 8px rgba(0,0,0,0.3); }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Título y botón de Hardware ---
title_col, button_col = st.columns([0.8, 0.2])

with title_col:
    st.markdown(f"<h1>{get_text('app_title')}</h1>", unsafe_allow_html=True)

# Crea el modal
modal = Modal(
    get_text("hardware_modal_title"),
    key="hardware_modal",
    padding=20,
    max_width=700
)

with button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(get_text("hardware_details_button"), key="hardware_info_button", help=get_text("hardware_details_button_help")):
        modal.open()

# Contenido del modal
if modal.is_open():
    with modal.container():
        st.markdown(get_text("hardware_modal_content"), unsafe_allow_html=True)


st.markdown(get_text("app_intro"))
st.markdown("---")

# ------------------------- CARGA DE MODELOS Y SCALER -------------------------
@st.cache_resource(show_spinner=get_text("spinner_loading_models"))
def cargar_modelos():
    try:
        modelo_ann = load_model("datos/modelos/modelo_produccion_ANN.keras")
        modelo_rf = joblib.load("datos/modelos/modelo_random_forest.pkl")
        modelo_xgb = XGBRegressor()
        modelo_xgb.load_model("datos/modelos/modelo_xgboost.json")
        return modelo_ann, modelo_rf, modelo_xgb
    except Exception as e:
        st.error(f"{get_text('error_loading_models_prefix')} {e}")
        st.stop()

@st.cache_resource(show_spinner=get_text("spinner_loading_scaler"))
def cargar_scaler():
    try:
        return joblib.load("datos/scaler.pkl")
    except Exception as e:
        st.error(f"{get_text('error_loading_scaler_prefix')} {e}")
        st.stop()

modelo_ann, modelo_rf, modelo_xgb = cargar_modelos()
scaler = cargar_scaler()

# ------------------------- TIEMPOS DE ENTRENAMIENTO -------------------------
@st.cache_data(show_spinner=get_text("spinner_loading_training_times"))
def cargar_tiempos_entrenamiento():
    try:
        df_tiempos = pd.read_csv("datos/tiempos_entrenamiento_modelos.csv")
        tiempos = dict(zip(df_tiempos['Modelo'], df_tiempos['Tiempo_Segundos']))
        return tiempos
    except Exception as e:
        st.warning(f"{get_text('warning_loading_training_times_prefix')} {e}")
        return {}

tiempos_entrenamiento = cargar_tiempos_entrenamiento()

# ------------------------- INTERFAZ CON TABS -------------------------
tab1, tab2 = st.tabs([get_text("tab1_title"), get_text("tab2_title")])

with tab1:
    st.header(get_text("tab1_header"))
    st.markdown(get_text("tab1_intro"))
    st.warning(get_text("tab1_warning_files"))

    try:
        X_scaled = np.load("datos/X_test.npy")
        y = np.load("datos/y_test.npy")

        y_pred_ann = modelo_ann.predict(X_scaled).flatten()
        y_pred_rf = modelo_rf.predict(X_scaled)
        y_pred_xgb = modelo_xgb.predict(X_scaled)

        modelos = ["ANN", "Random Forest", "XGBoost"]
        preds = [y_pred_ann, y_pred_rf, y_pred_xgb]

        st.markdown("---")
        st.subheader(get_text("tab1_subheader_viz"))
        st.info(get_text("tab1_info_viz"))

        cols = st.columns(3)
        for i, (nombre, pred) in enumerate(zip(modelos, preds)):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(y, pred, alpha=0.6,
                       color='teal' if nombre == "ANN" else 'orange' if nombre == "Random Forest" else 'darkorange')
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', label=get_text("pdf_plot_line_ideal"))
            ax.set_xlabel(get_text("pdf_plot_xlabel_real"))
            ax.set_ylabel(get_text("pdf_plot_ylabel_pred"))
            ax.set_title(f"**{nombre}**: {get_text('pdf_plot_title_pred_vs_real')}")
            ax.grid(True)
            cols[i].pyplot(fig)
            guardar_grafico_pred_vs_real(y, pred, nombre, f"img/pred_vs_real_{nombre}.png", get_text)

        st.markdown("---")
        st.subheader(get_text("tab1_subheader_metrics"))
        st.info(get_text("tab1_info_metrics"))

        mae_vals = [mean_absolute_error(y, pred) for pred in preds]
        mse_vals = [mean_squared_error(y, pred) for pred in preds]
        r2_vals = [r2_score(y, pred) for pred in preds]
        tiempos_tabla = [tiempos_entrenamiento.get(m, "N/A") for m in modelos]

        df_metricas = pd.DataFrame({
            get_text("pdf_table_model"): modelos,
            "MAE": mae_vals,
            "MSE": mse_vals,
            "R²": r2_vals,
            get_text("pdf_table_time_s"): tiempos_tabla
        })

        st.dataframe(df_metricas.style.format({
            "MAE": "{:.4f}", "MSE": "{:.4f}", "R²": "{:.4f}", get_text("pdf_table_time_s"): "{:.2f}"
        }).highlight_max(subset=['R²'], color='lightgreen')
          .highlight_min(subset=['MAE', 'MSE'], color='salmon'))

        st.markdown("---")
        st.subheader(get_text("tab1_subheader_theil"))
        st.info(get_text("tab1_info_theil"))
        theil_vals = [coeficiente_u_theil(y, pred) for pred in preds]
        df_theil = pd.DataFrame({
            get_text("pdf_table_model"): modelos,
            get_text("pdf_theil_table_u_theil"): theil_vals
        })
        st.dataframe(df_theil.style.format({get_text("pdf_theil_table_u_theil"): "{:.4f}"}).highlight_min(color='skyblue'))

        st.markdown("---")
        st.subheader(get_text("tab1_subheader_diebold"))
        st.info(get_text("tab1_info_diebold"))
        comparaciones = [
            ("ANN vs Random Forest", *prueba_diebold_mariano(y, y_pred_ann, y_pred_rf)),
            ("ANN vs XGBoost", *prueba_diebold_mariano(y, y_pred_ann, y_pred_xgb)),
            ("Random Forest vs XGBoost", *prueba_diebold_mariano(y, y_pred_rf, y_pred_xgb))
        ]

        df_dm = pd.DataFrame(comparaciones, columns=[get_text("pdf_diebold_table_comparison"), get_text("pdf_diebold_table_statistic"), get_text("pdf_diebold_table_p_value")])
        st.dataframe(df_dm.style.format({get_text("pdf_diebold_table_statistic"): "{:.4f}", get_text("pdf_diebold_table_p_value"): "{:.4f}"}))

        st.markdown("---")
        st.subheader(get_text("tab1_subheader_report"))
        st.success(get_text("tab1_success_report"))
        # El mejor modelo se determina por el R² más alto
        mejor_modelo_idx = np.argmax(r2_vals)
        mejor_modelo_nombre = modelos[mejor_modelo_idx]

        if st.button(get_text("tab1_button_generate_report"), key="generate_comp_report"):
            with st.spinner(get_text("tab1_spinner_generating_report")):
                pdf_path = generar_pdf_resultados(
                    mae_vals, mse_vals, r2_vals, modelos, mejor_modelo_nombre,
                    tiempos_entrenamiento, y, modelo_ann, modelo_rf, modelo_xgb, X_scaled,
                    get_text # Pasamos la función de traducción
                )
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            get_text("tab1_download_button"),
                            f,
                            file_name="reporte_comparativo_modelos.pdf",
                            mime="application/pdf"
                        )
                    st.success(get_text("tab1_report_success"))
                else:
                    st.error(get_text("tab1_report_error"))

    except FileNotFoundError:
        st.error(get_text("tab1_error_files_missing"))
        st.info(get_text("tab1_info_files_missing"))
    except Exception as e:
        st.error(f"{get_text('tab1_error_unexpected')} {e}")


with tab2:
    st.header(get_text("tab2_header"))
    st.markdown(get_text("tab2_intro"))
    st.markdown("---")

    with st.form("pred_form"):
        st.markdown(get_text("form_header_details"))

        col_input = st.columns(3)
        with col_input[0]:
            volume = st.number_input(
                get_text("input_volume"),
                value=1.2,
                min_value=0.01,
                help=get_text("help_volume")
            )
        with col_input[1]:
            # Obtener las opciones de tipo de producto traducidas
            product_type_options_display = get_text("options_product_type")
            product_type_es = st.radio(
                get_text("input_product_type"),
                product_type_options_display,
                horizontal=True,
                help=get_text("help_product_type")
            )
        with col_input[2]:
            cantidad = st.number_input(
                get_text("input_quantity"),
                min_value=1,
                value=1,
                step=1,
                help=get_text("help_quantity")
            )

        st.markdown(get_text("form_header_model_selection"))
        modelo_sel = st.selectbox(
            get_text("select_model"),
            ["Random Forest", "ANN", "XGBoost"], # Nombres de modelos no se traducen
            index=0,
            help=get_text("help_select_model")
        )

        submit = st.form_submit_button(get_text("button_predict"))

    if submit:
        # Mapeo de la opción seleccionada por el usuario (traducida)
        # al valor original en inglés que espera el modelo.
        # Creamos un mapeo inverso dinámico para asegurar la compatibilidad.
        mapa_tipo_producto_para_modelo = {
            get_text("options_product_type", lang='es')[i]: ["Automotive", "Electronics", "Furniture", "Textiles"][i]
            for i in range(len(get_text("options_product_type", lang='es')))
        }
        # Aseguramos que también funcione si el usuario cambia a inglés y vuelve a enviar el formulario
        mapa_tipo_producto_para_modelo.update({
            get_text("options_product_type", lang='en')[i]: ["Automotive", "Electronics", "Furniture", "Textiles"][i]
            for i in range(len(get_text("options_product_type", lang='en')))
        })

        product_type_for_model = mapa_tipo_producto_para_modelo[product_type_es]


        tipo_producto_cols = {
            'Product Type_Automotive': 1 if product_type_for_model == "Automotive" else 0,
            'Product Type_Electronics': 1 if product_type_for_model == "Electronics" else 0,
            'Product Type_Furniture': 1 if product_type_for_model == "Furniture" else 0,
            'Product Type_Textiles': 1 if product_type_for_model == "Textiles" else 0
        }

        # Asegúrate de que todas las características esperadas por el scaler estén presentes,
        # incluso si son valores por defecto.
        entrada_dict = {
            'Machine ID': 1,
            'Units Produced': 1, # Se usará 'cantidad' para el cálculo final, pero 1 para el tiempo por unidad
            'Defects': 0,
            'Labour Cost Per Hour': 12.0,
            'Energy Consumption kWh': 200.0,
            'Operator Count': 3,
            'Maintenance Hours': 0.5,
            'Down time Hours': 1.0,
            'Production Volume Cubic Meters': volume,
            'Scrap Rate': 0.01,
            'Rework Hours': 0.5,
            'Quality Checks Failed': 0,
            'Average Temperature C': 25.0,
            'Average Humidity Percent': 60.0,
            'Shift_Night': 0,
            'Shift_Swing': 0,
            **tipo_producto_cols
        }
        entrada = pd.DataFrame([entrada_dict])


        try:
            entrada_scaled = scaler.transform(entrada)
        except Exception as e:
            st.error(f"{get_text('error_scaling_input_data_prefix')} {e}")
            st.stop()

        predicciones = {
            "ANN": float(modelo_ann.predict(entrada_scaled)[0][0]),
            "Random Forest": float(modelo_rf.predict(entrada_scaled)[0]),
            "XGBoost": float(modelo_xgb.predict(entrada_scaled)[0])
        }

        for key in predicciones:
            predicciones[key] = max(0, predicciones[key]) # Asegurarse de que no haya predicciones negativas

        prediccion_seleccionada = predicciones[modelo_sel]
        tiempo_total_estimado = prediccion_seleccionada * cantidad

        st.markdown("---")
        st.subheader(get_text("results_header"))
        st.markdown(get_text("selected_model_msg", model_name=modelo_sel))

        col_result1, col_result2 = st.columns(2)
        with col_result1:
            st.metric(label=get_text("metric_time_per_unit"), value=f"{prediccion_seleccionada:.2f}")
        with col_result2:
            st.metric(label=get_text("metric_total_time", quantity=cantidad), value=f"{tiempo_total_estimado:.2f}")

        horas_por_dia = 8
        dias_estimados = int(np.ceil(tiempo_total_estimado / horas_por_dia))
        if tiempo_total_estimado <= horas_por_dia and tiempo_total_estimado > 0: # Considerar caso de 0 tiempo
            st.success(get_text("success_same_day"))
        elif tiempo_total_estimado == 0:
             st.info("La producción estimada es casi instantánea (0 horas).") # Mensaje especial para 0 horas
        else:
            st.info(get_text("info_days_needed", quantity=cantidad, days=dias_estimados))
            hoy = datetime.datetime.now()
            # Ajustar la fecha de recojo si los días son 0 o 1
            if dias_estimados <= 1:
                fecha_recojo = hoy
            else:
                fecha_recojo = hoy + datetime.timedelta(days=dias_estimados -1) # Si toma 2 días, es mañana (+1 día)
            st.info(get_text("info_pickup_date", date=fecha_recojo.strftime('%d/%m/%Y')))


        st.markdown("---")
        st.subheader(get_text("comparison_header"))
        st.info(get_text("comparison_info"))

        otras_predicciones_total = {
            k: v * cantidad for k, v in predicciones.items() if k != modelo_sel
        }

        df_pred_comparativa = pd.DataFrame(
            {get_text("pdf_table_model"): [modelo_sel] + list(otras_predicciones_total.keys()),
             get_text("comparison_chart_ylabel"): [tiempo_total_estimado] + list(otras_predicciones_total.values())}
        )
        st.dataframe(df_pred_comparativa.style.format({get_text("comparison_chart_ylabel"): "{:.2f}"}))

        fig, ax = plt.subplots(figsize=(8, 5))
        model_names_plot = [modelo_sel] + list(otras_predicciones_total.keys())
        values_plot = [tiempo_total_estimado] + list(otras_predicciones_total.values())

        colors = ['#4CAF50']
        for name in otras_predicciones_total.keys():
            if name == "ANN": colors.append('teal')
            elif name == "Random Forest": colors.append('orange')
            elif name == "XGBoost": colors.append('darkorange')

        bars = ax.bar(model_names_plot, values_plot, color=colors)
        ax.set_ylabel(get_text("comparison_chart_ylabel"))
        ax.set_title(get_text("comparison_chart_title"))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=9)

        st.pyplot(fig)
        st.markdown("---")

        if st.button(get_text("button_download_individual_report"), key="generate_ind_report"):
            # Para el PDF individual, necesitamos los datos de entrada originales del formulario
            # y las predicciones por unidad para los otros modelos
            # Asegúrate de que 'entrada_dict' sea el que pasas, ya que contiene los datos en un formato útil
            # y luego pasas la predicción por unidad y otras predicciones por unidad.
            otras_predicciones_unidad = {k: v for k, v in predicciones.items() if k != modelo_sel}

            pdf_path_individual = generar_pdf_prediccion_individual(
                entrada_dict, modelo_sel, prediccion_seleccionada,
                otras_predicciones_unidad,
                get_text # Pasamos la función de traducción
            )
            if os.path.exists(pdf_path_individual):
                with open(pdf_path_individual, "rb") as f:
                    st.download_button(
                        get_text("tab1_download_button"), # Usamos el mismo botón de descarga
                        f,
                        file_name="reporte_prediccion_individual.pdf",
                        mime="application/pdf"
                    )
                st.success(get_text("individual_report_success"))
            else:
                st.error(get_text("individual_report_error"))