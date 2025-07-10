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
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm
from scipy.stats import t
import re


# --- Crear carpetas si no existen ---
os.makedirs("img", exist_ok=True)
os.makedirs("reporte", exist_ok=True)

# ------------------------- UTILIDADES -------------------------
def limpiar_texto(texto):
    if isinstance(texto, str):
        # Normaliza acentos
        texto = unicodedata.normalize('NFKD', texto).encode('latin-1', 'ignore').decode('latin-1')
        # Elimina emojis (caracteres no imprimibles o fuera del rango aceptado)
        texto = re.sub(r'[^\x20-\x7EñÑáéíóúÁÉÍÓÚüÜ ]+', '', texto)
        return texto
    return str(texto)

def guardar_grafico_pred_vs_real(y_true, y_pred, modelo_nombre, filename):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predicciones", color='teal' if modelo_nombre == "ANN" else 'orange' if modelo_nombre == "Random Forest" else 'darkorange')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Línea ideal")
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title(f"{modelo_nombre}: Predicción vs Real")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def coeficiente_u_theil(y_true, y_pred):
    """
    Calcula el Coeficiente U de Theil para comparar un modelo de regresión.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return num / den

def prueba_diebold_mariano(y_true, y_pred1, y_pred2):
    """
    Realiza la prueba de Diebold-Mariano para comparar dos modelos de regresión.
    Usa como función de pérdida el Error Cuadrático Medio (MSE).
    
    :param y_true: Valores reales (array-like)
    :param y_pred1: Predicciones del modelo 1
    :param y_pred2: Predicciones del modelo 2
    :return: (estadístico DM, valor p)
    
    Interpretación:
    - Estadístico > 0: El modelo 1 tiene menor error → es mejor.
    - Estadístico < 0: El modelo 2 tiene menor error → es mejor.
    - p-valor < 0.05: La diferencia es estadísticamente significativa.
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    # Verificar que las longitudes coincidan
    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("Todas las series deben tener la misma longitud.")

    # Errores de predicción
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Diferencia de pérdidas (invertido para que signo positivo indique que modelo 1 es mejor)
    d = (e2**2) - (e1**2)

    # Estadístico Diebold-Mariano
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)

    DM_stat = d_mean / np.sqrt(d_var / n)

    # p-valor bilateral
    p_value = 2 * t.sf(np.abs(DM_stat), df=n - 1)

    return DM_stat, p_value



def generar_pdf_resultados(mae_vals, mse_vals, r2_vals, modelos, mejor_modelo, tiempos_entrenamiento):
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import os

    # 1. Calcular Theil U
    theil_vals = [
        coeficiente_u_theil(y, modelo_ann.predict(X_scaled).flatten()),
        coeficiente_u_theil(y, modelo_rf.predict(X_scaled)),
        coeficiente_u_theil(y, modelo_xgb.predict(X_scaled))
    ]

    # 2. Crear gráfico comparativo
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].bar(modelos, mae_vals, color='skyblue'); axs[0].set_title("MAE")
    axs[1].bar(modelos, mse_vals, color='lightgreen'); axs[1].set_title("MSE")
    axs[2].bar(modelos, r2_vals, color='orange'); axs[2].set_title("R²")
    plt.tight_layout()
    grafico_path = "img/grafico_comparacion.png"
    fig.savefig(grafico_path)
    plt.close(fig)

    # 3. Inicializar PDF
    pdf_path = "reporte/reporte_comparativo.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # === Encabezado ===
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, limpiar_texto("Informe Comparativo de Modelos de Predicción"), ln=1, align="C")
    pdf.ln(5)

    # === Introducción ===
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, limpiar_texto(
        "Este informe documenta el análisis y evaluación de modelos predictivos aplicados al tiempo de producción. "
        "Se evaluaron los modelos Red Neuronal Artificial (ANN), Random Forest y XGBoost. Las métricas utilizadas "
        "incluyen MAE, MSE, R² y tiempo de entrenamiento. Además, se analizó el coeficiente U de Theil y se realizó una comparación estadística usando la prueba de Diebold-Mariano."
    ))
    pdf.ln(5)

    # === Visualizaciones EDA ===
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("1. Visualizaciones Exploratorias (EDA)"), ln=1)
    for title, img in [
        ("Distribución del tiempo de producción", "img/eda_hist_production_time.png"),
        ("Boxplot por tipo de producto", "img/eda_boxplot_tipo_producto.png"),
        ("Matriz de correlación", "img/eda_heatmap_corr.png"),
        ("Unidades producidas vs tiempo", "img/eda_scatter_units_vs_time.png")
    ]:
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 8, limpiar_texto(title), ln=1)
        if os.path.exists(img):
            pdf.image(img, x=10, w=180)
            pdf.ln(3)
        else:
            pdf.cell(0, 8, limpiar_texto(f"No se encontró {img}"), ln=1)

    # === Preprocesamiento ===
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("2. Preprocesamiento de Datos"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(
        "- Conversión de fechas\n"
        "- Imputación de valores nulos\n"
        "- Eliminación de outliers\n"
        "- Codificación de variables categóricas\n"
        "- Normalización\n"
        "- División en conjunto de entrenamiento y prueba"
    ))
    pdf.ln(5)

    # === Comparación de Modelos ===
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("3. Comparación de Modelos (MAE, MSE, R², Tiempo)"), ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto("Resultados de las métricas evaluadas:"))
    pdf.image(grafico_path, x=10, w=180)
    pdf.ln(5)

    # Gráficos pred vs real
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, limpiar_texto("Gráficos Predicción vs Real"), ln=1)
    for i, modelo in enumerate(modelos):
        img_path = f"img/pred_vs_real_{i}.png"
        if os.path.exists(img_path):
            pdf.cell(0, 8, limpiar_texto(modelo), ln=1)
            pdf.image(img_path, x=10, w=180)
            pdf.ln(3)

    # Tabla de métricas principales
    pdf.set_font("Arial", "B", 12)
    pdf.cell(45, 8, limpiar_texto("Modelo"), 1, align="C")
    pdf.cell(25, 8, "MAE", 1, align="C")
    pdf.cell(25, 8, "MSE", 1, align="C")
    pdf.cell(20, 8, "R²", 1, align="C")
    pdf.cell(40, 8, "Tiempo (s)", 1, align="C")
    pdf.ln()
    pdf.set_font("Arial", "", 12)
    for i, modelo in enumerate(modelos):
        tiempo = tiempos_entrenamiento.get(modelo, "N/A")
        pdf.cell(45, 8, limpiar_texto(modelo), 1)
        pdf.cell(25, 8, f"{mae_vals[i]:.3f}", 1, align="C")
        pdf.cell(25, 8, f"{mse_vals[i]:.3f}", 1, align="C")
        pdf.cell(20, 8, f"{r2_vals[i]:.3f}", 1, align="C")
        pdf.cell(40, 8, f"{tiempo:.2f}" if isinstance(tiempo, (int, float)) else str(tiempo), 1, align="C")
        pdf.ln()
    pdf.ln(5)

    # === Coeficiente U de Theil ===
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("4. Coeficiente U de Theil"), ln=1)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(80, 8, limpiar_texto("Modelo"), 1, align="C")
    pdf.cell(60, 8, "U de Theil", 1, align="C")
    pdf.ln()
    pdf.set_font("Arial", "", 12)
    for i, modelo in enumerate(modelos):
        pdf.cell(80, 8, limpiar_texto(modelo), 1)
        pdf.cell(60, 8, f"{theil_vals[i]:.4f}", 1, align="C")
        pdf.ln()
    pdf.ln(5)

    # === Prueba de Diebold-Mariano ===
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("5. Pruebas de Diebold-Mariano"), ln=1)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(90, 8, limpiar_texto("Comparación"), 1, align="C")
    pdf.cell(40, 8, "Estadística", 1, align="C")
    pdf.cell(40, 8, "p-valor", 1, align="C")
    pdf.ln()

    comparaciones = [
        ("ANN vs Random Forest", *prueba_diebold_mariano(y, modelo_ann.predict(X_scaled).flatten(), modelo_rf.predict(X_scaled))),
        ("ANN vs XGBoost", *prueba_diebold_mariano(y, modelo_ann.predict(X_scaled).flatten(), modelo_xgb.predict(X_scaled))),
        ("Random Forest vs XGBoost", *prueba_diebold_mariano(y, modelo_rf.predict(X_scaled), modelo_xgb.predict(X_scaled)))
    ]
    pdf.set_font("Arial", "", 12)
    for nombre, stat, p in comparaciones:
        pdf.cell(90, 8, limpiar_texto(nombre), 1)
        pdf.cell(40, 8, f"{stat:.4f}", 1, align="C")
        pdf.cell(40, 8, f"{p:.4f}", 1, align="C")
        pdf.ln()

    # === Conclusión ===
    pdf.ln(8)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, limpiar_texto("6. Conclusión"), ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, limpiar_texto(
        f"Según las métricas evaluadas, el modelo recomendado es: {mejor_modelo}. "
        "Este modelo mostró el mejor rendimiento general, reflejado en sus valores de R², menor MAE/MSE y consistencia en las pruebas estadísticas."
    ))

    # Pie de página
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 10, limpiar_texto("Generado por Maykol Ramos - Rodrigez Leon | UNT - Tesis 2025"), 0, 0, 'C')

    pdf.output(pdf_path)

    # Eliminar gráfico temporal
    if os.path.exists(grafico_path):
        os.remove(grafico_path)

    return pdf_path

def generar_pdf_prediccion_individual(datos_entrada, modelo_sel, prediccion, otras_predicciones):
    horas_por_dia = 8
    dias = int(np.ceil(prediccion / horas_por_dia))
    hoy = datetime.datetime.now()
    fecha_recojo = hoy + datetime.timedelta(days=dias-1 if dias > 0 else 0)
    fecha_recojo_str = fecha_recojo.strftime('%Y-%m-%d')

    if prediccion <= horas_por_dia:
        recojo_msg = (
            "✅ Su producto estará listo **el mismo día**. "
            "Por favor, acérquese al finalizar la jornada laboral."
        )
    else:
        recojo_msg = (
            f"⏳ Su producto estará listo en aproximadamente **{dias} día(s) laborable(s)** "
            f"(considerando 8 horas de trabajo por día).\n"
            f"**Fecha estimada de recojo:** {fecha_recojo_str}"
        )

    pdf_path = "reporte/reporte_prediccion_individual.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Portada
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(40, 60, 120)
    pdf.cell(0, 15, limpiar_texto("Reporte de Predicción Individual"), ln=1, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Fecha y hora de generación: {hoy.strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(8)

    # Datos de entrada en tabla
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Datos Ingresados:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.set_fill_color(240, 240, 240)
    for columna, valor in datos_entrada.items():
        pdf.cell(60, 8, limpiar_texto(str(columna)), 1, 0, 'L', 1)
        pdf.cell(60, 8, limpiar_texto(str(valor)), 1, 1, 'L', 0)
    pdf.ln(5)

    # Predicción principal
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, f"Predicción de Tiempo de Producción:", ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Modelo seleccionado: **{modelo_sel}**\n"
                          f"Tiempo estimado: **{prediccion:.2f} horas**")
    pdf.ln(3)

    # Otras predicciones
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Comparación con otros modelos:", ln=1)
    pdf.set_font("Arial", "", 11)
    for modelo, valor in otras_predicciones.items():
        pdf.cell(0, 8, f"{modelo}: {valor:.2f} horas", ln=1)
    pdf.ln(5)

    # Recomendación de recojo
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 8, "Recomendación:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, limpiar_texto(recojo_msg))
    pdf.ln(5)

    # Pie de página institucional
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, limpiar_texto("Generado por Maykol Ramos - Rodriguez Leon - UNT - Tesis 2025"), 0, 0, 'C')

    pdf.output(pdf_path)
    return pdf_path

# ------------------------- CONFIGURACIÓN -------------------------
st.set_page_config(page_title="Predicción de Producción", layout="centered")
st.title("📈 Análisis Comparativo y Predicción de Tiempo de Producción")

# ------------------------- CARGA DE MODELOS Y SCALER -------------------------
@st.cache_resource
def cargar_modelos():
    modelo_ann = load_model("datos/modelos/modelo_produccion_ANN.h5")
    modelo_rf = joblib.load("datos/modelos/modelo_random_forest.pkl")
    modelo_xgb = XGBRegressor()
    modelo_xgb.load_model("datos/modelos/modelo_xgboost.json")
    return modelo_ann, modelo_rf, modelo_xgb

@st.cache_resource
def cargar_scaler():
    return joblib.load("datos/scaler.pkl")

modelo_ann, modelo_rf, modelo_xgb = cargar_modelos()
scaler = cargar_scaler()

# ------------------------- TIEMPOS DE ENTRENAMIENTO -------------------------
@st.cache_data
def cargar_tiempos_entrenamiento():
    try:
        df_tiempos = pd.read_csv("datos/tiempos_entrenamiento_modelos.csv")
        tiempos = dict(zip(df_tiempos['Modelo'], df_tiempos['Tiempo_Segundos']))
        return tiempos
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de tiempos de entrenamiento: {e}")
        return {}

tiempos_entrenamiento = cargar_tiempos_entrenamiento()

# ------------------------- INTERFAZ CON TABS -------------------------
tab1, tab2 = st.tabs(["📊 Comparativa de Modelos", "🧾 Predicción Individual"])

with tab1:
    st.subheader("📊 Comparativa de Modelos con Datos de Prueba")
    try:
        X_scaled = np.load("datos/X_test.npy")
        y = np.load("datos/y_test.npy")

        y_pred_ann = modelo_ann.predict(X_scaled).flatten()
        y_pred_rf = modelo_rf.predict(X_scaled)
        y_pred_xgb = modelo_xgb.predict(X_scaled)

        modelos = ["ANN", "Random Forest", "XGBoost"]
        preds = [y_pred_ann, y_pred_rf, y_pred_xgb]

        # ==== 1. Gráficos Real vs Predicho ====
        st.markdown("### 📈 Modelos:")
        for nombre, pred in zip(modelos, preds):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y, pred, alpha=0.6, color='teal' if nombre == "ANN" else 'orange' if nombre == "Random Forest" else 'darkorange')
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
            ax.set_xlabel("Valores reales")
            ax.set_ylabel("Predicciones")
            ax.set_title(f"{nombre}: Predicción vs Real")
            ax.grid(True)
            st.pyplot(fig)
            guardar_grafico_pred_vs_real(y, pred, nombre, f"img/pred_vs_real_{nombre}.png")

        # ==== 2. Métricas de Evaluación ====
        st.markdown("### 🧮 Medidas de Evaluación (MAE, MSE, R²)")
        mae_vals = [mean_absolute_error(y, pred) for pred in preds]
        mse_vals = [mean_squared_error(y, pred) for pred in preds]
        r2_vals = [r2_score(y, pred) for pred in preds]
        tiempos_tabla = [tiempos_entrenamiento.get(m, "N/A") for m in modelos]

        df_metricas = pd.DataFrame({
            "Modelo": modelos,
            "MAE": mae_vals,
            "MSE": mse_vals,
            "R²": r2_vals,
            "Tiempo Entrenamiento (s)": tiempos_tabla
        })

        st.dataframe(df_metricas.style.format({
            "MAE": "{:.4f}", "MSE": "{:.4f}", "R²": "{:.4f}", "Tiempo Entrenamiento (s)": "{:.2f}"
        }))

        # ==== 3. Coeficiente U de Theil ====
        st.markdown("### 📉 Análisis del Coeficiente U de Theil")
        theil_vals = [coeficiente_u_theil(y, pred) for pred in preds]
        df_theil = pd.DataFrame({
            "Modelo": modelos,
            "Coeficiente U de Theil": theil_vals
        })
        st.dataframe(df_theil.style.format({"Coeficiente U de Theil": "{:.4f}"}))

        # ==== 4. Pruebas de Diebold-Mariano ====
        st.markdown("### 🔍 Pruebas de Comparación Diebold-Mariano")
        comparaciones = [
            ("ANN vs Random Forest", *prueba_diebold_mariano(y, y_pred_ann, y_pred_rf)),
            ("ANN vs XGBoost", *prueba_diebold_mariano(y, y_pred_ann, y_pred_xgb)),
            ("Random Forest vs XGBoost", *prueba_diebold_mariano(y, y_pred_rf, y_pred_xgb))
        ]

        df_dm = pd.DataFrame(comparaciones, columns=["Comparación", "Estadística DM", "p-valor"])
        st.dataframe(df_dm.style.format({"Estadística DM": "{:.4f}", "p-valor": "{:.4f}"}))

        # ==== 5. Generación de Reporte PDF ====
        mejor_modelo = modelos[r2_vals.index(max(r2_vals))]
        if st.button("📄 Generar Reporte PDF Comparativo"):
            pdf_path = generar_pdf_resultados(mae_vals, mse_vals, r2_vals, modelos, mejor_modelo, tiempos_entrenamiento)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Descargar Reporte PDF", f, file_name="reporte_comparativo.pdf", mime="application/pdf")
            else:
                st.error("❌ No se pudo generar el reporte PDF.")

    except Exception as e:
        st.error(f"❌ Error al cargar los datos o modelos: {e}")

with tab2:
    st.subheader("🧾 Predicción Individual")

    with st.form("pred_form"):
        col1, col2 = st.columns(2)

        with col1:
            machine_id = st.number_input("ID Máquina", value=1)
            units = st.number_input("Unidades Producidas", value=100)
            defects = st.number_input("Defectos", value=0)
            labour_cost = st.number_input("Costo Laboral x Hora", value=12.0)
            energy = st.number_input("Consumo Energía (kWh)", value=200.0)
            operator_count = st.number_input("Operarios", value=3)

        with col2:
            maintenance = st.number_input("Horas Mantenimiento", value=0.5)
            downtime = st.number_input("Horas Inactividad", value=1.0)
            volume = st.number_input("Volumen (m³)", value=1.2)
            scrap = st.number_input("Tasa Desechos", value=0.01)
            rework = st.number_input("Horas Retrabajo", value=0.5)
            qc_failed = st.number_input("Inspecciones Fallidas", value=0)
            temperature = st.number_input("Temp. Promedio (°C)", value=25.0)
            humidity = st.number_input("Humedad (%)", value=60.0)

        product_type = st.radio("Tipo de Producto", ["Automotive", "Electronics", "Furniture", "Textiles"], horizontal=True)
        shift = st.radio("Turno", ["Day", "Night", "Swing"], horizontal=True)

        modelo_sel = st.selectbox("Selecciona el modelo para predecir:", ["ANN", "Random Forest", "XGBoost"])

        submit = st.form_submit_button("Predecir")

    if submit:
        tipo_producto = {
            'Product Type_Automotive': 1 if product_type == "Automotive" else 0,
            'Product Type_Electronics': 1 if product_type == "Electronics" else 0,
            'Product Type_Furniture': 1 if product_type == "Furniture" else 0,
            'Product Type_Textiles': 1 if product_type == "Textiles" else 0
        }
        turno = {
            'Shift_Night': 1 if shift == "Night" else 0,
            'Shift_Swing': 1 if shift == "Swing" else 0
        }

        entrada = pd.DataFrame([{
            'Machine ID': machine_id,
            'Units Produced': units,
            'Defects': defects,
            'Labour Cost Per Hour': labour_cost,
            'Energy Consumption kWh': energy,
            'Operator Count': operator_count,
            'Maintenance Hours': maintenance,
            'Down time Hours': downtime,
            'Production Volume Cubic Meters': volume,
            'Scrap Rate': scrap,
            'Rework Hours': rework,
            'Quality Checks Failed': qc_failed,
            'Average Temperature C': temperature,
            'Average Humidity Percent': humidity,
            **tipo_producto,
            **turno
        }])

        entrada_scaled = scaler.transform(entrada)

        predicciones = {
            "ANN": modelo_ann.predict(entrada_scaled)[0][0],
            "Random Forest": modelo_rf.predict(entrada_scaled)[0],
            "XGBoost": modelo_xgb.predict(entrada_scaled)[0]
        }

        st.metric(modelo_sel, f"{predicciones[modelo_sel]:.2f} horas")

        st.write("Predicción de los otros modelos:")
        for nombre, valor in predicciones.items():
            if nombre != modelo_sel:
                st.write(f"{nombre}: **{valor:.2f} horas**")

        fig, ax = plt.subplots()
        ax.bar(["ANN", "Random Forest", "XGBoost"], [predicciones["ANN"], predicciones["Random Forest"], predicciones["XGBoost"]], color=["skyblue", "lightgreen", "orange"])
        ax.set_ylabel("Horas estimadas")
        st.pyplot(fig)

        # --- PDF de reporte individual ---
        if predicciones:
            datos_entrada_dict = entrada.iloc[0].to_dict()
            otras_predicciones = {k: v for k, v in predicciones.items() if k != modelo_sel}
            pdf_path = generar_pdf_prediccion_individual(datos_entrada_dict, modelo_sel, predicciones[modelo_sel], otras_predicciones)

            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Descargar Reporte PDF Individual", f, file_name="reporte_prediccion_individual.pdf", mime="application/pdf")
            else:
                st.error("❌ No se pudo generar el reporte PDF.")
