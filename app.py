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

# --- Crear carpetas si no existen ---
os.makedirs("img", exist_ok=True)
os.makedirs("reporte", exist_ok=True)

# ------------------------- UTILIDADES -------------------------
def limpiar_texto(texto):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto).encode('latin-1', 'ignore').decode('latin-1')
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

def generar_pdf_resultados(mae_vals, mse_vals, r2_vals, modelos, mejor_modelo, tiempos_entrenamiento):
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    import os

    # --------- Gráfico comparativo de métricas ---------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].bar(modelos, mae_vals, color='skyblue'); axs[0].set_title("MAE")
    axs[1].bar(modelos, mse_vals, color='lightgreen'); axs[1].set_title("MSE")
    axs[2].bar(modelos, r2_vals, color='orange'); axs[2].set_title("R²")
    plt.tight_layout()
    grafico_path = "img/grafico_comparacion.png"
    fig.savefig(grafico_path)
    plt.close(fig)

    pdf_path = "reporte/reporte_comparativo.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Portada
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, limpiar_texto("Informe Comparativo de Modelos de Predicción"), ln=1, align="C")
    pdf.ln(10)

    # 1. Introducción
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, limpiar_texto(
        "Este informe documenta el análisis exploratorio, preprocesamiento, entrenamiento y evaluación de modelos "
        "para la predicción del tiempo de producción. Se muestran visualizaciones EDA, estadísticas descriptivas y "
        "la comparación de tres modelos de machine learning."
    ))
    pdf.ln(8)

    # 2. EDA - Visualizaciones
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Exploración de Datos (EDA)", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Distribución de la variable objetivo:", ln=1)
    pdf.image("img/eda_hist_production_time.png", x=10, w=180)
    pdf.ln(3)
    pdf.cell(0, 8, "Boxplot por tipo de producto:", ln=1)
    pdf.image("img/eda_boxplot_tipo_producto.png", x=10, w=180)
    pdf.ln(3)
    pdf.cell(0, 8, "Matriz de correlación:", ln=1)
    pdf.image("img/eda_heatmap_corr.png", x=10, w=180)
    pdf.ln(3)
    pdf.cell(0, 8, "Relación unidades producidas vs tiempo:", ln=1)
    pdf.image("img/eda_scatter_units_vs_time.png", x=10, w=180)
    pdf.ln(8)

    # 3. Preprocesamiento
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Preprocesamiento de Datos", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(
        "- Conversión de fechas\n"
        "- Eliminación de columnas irrelevantes\n"
        "- Imputación/eliminación de nulos\n"
        "- Eliminación de outliers\n"
        "- Codificación de variables categóricas\n"
        "- Normalización de variables numéricas\n"
        "- División en entrenamiento y prueba"
    ))
    pdf.ln(8)

    # 4. Comparativa de Modelos: Gráficos antes de la tabla
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Comparación de Modelos", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(
        "Se entrenaron los modelos: ANN, Random Forest y XGBoost. "
        "Las métricas evaluadas fueron: MAE, MSE, R² y tiempo de entrenamiento."
    ))
    pdf.ln(5)

    # Gráfico comparativo de métricas
    try:
        pdf.image(grafico_path, x=10, w=180)
        pdf.ln(5)
    except:
        pdf.cell(0, 10, "Error al cargar el gráfico comparativo.", ln=1)

    # Gráficos Predicción vs Real
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Gráficos Predicción vs Real:", ln=1)
    for i, modelo in enumerate(modelos):
        img_path = f"img/pred_vs_real_{i}.png"
        if os.path.exists(img_path):
            pdf.cell(0, 8, f"{modelo}:", ln=1)
            pdf.image(img_path, x=10, w=180)
            pdf.ln(3)
    pdf.ln(5)

    # Tabla de resultados
    pdf.set_font("Arial", "B", 12)
    pdf.cell(45, 8, "Modelo", 1, align="C")
    pdf.cell(30, 8, "MAE", 1, align="C")
    pdf.cell(30, 8, "MSE", 1, align="C")
    pdf.cell(30, 8, "R² Score", 1, align="C")
    pdf.cell(55, 8, "Tiempo Entrenamiento (s)", 1, align="C")
    pdf.ln()
    pdf.set_font("Arial", "", 12)
    for i, modelo in enumerate(modelos):
        pdf.cell(45, 8, modelo, 1)
        pdf.cell(30, 8, f"{mae_vals[i]:.3f}", 1, align="C")
        pdf.cell(30, 8, f"{mse_vals[i]:.3f}", 1, align="C")
        pdf.cell(30, 8, f"{r2_vals[i]:.3f}", 1, align="C")
        tiempo = tiempos_entrenamiento.get(modelo, "N/A")
        pdf.cell(55, 8, f"{tiempo:.2f}" if isinstance(tiempo, (float, int)) else str(tiempo), 1, align="C")
        pdf.ln()

    pdf.ln(10)
    # 5. Conclusión
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Conclusión:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, limpiar_texto(
        f"El modelo recomendado es: {mejor_modelo}, por su mejor desempeño en las métricas evaluadas."
    ))

    # Pie de página
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, limpiar_texto("Generado por Maykol Ramos- Rodrigez Leon  - UNT - Tesis 2025"), 0, 0, 'C')

    pdf.output(pdf_path)
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
    modelo_ann = load_model("datos/modelo_produccion_ANN.h5")
    modelo_rf = joblib.load("datos/modelo_random_forest.pkl")
    modelo_xgb = XGBRegressor()
    modelo_xgb.load_model("datos/modelo_xgboost.json")
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
        mae_vals = [mean_absolute_error(y, y_pred_ann), mean_absolute_error(y, y_pred_rf), mean_absolute_error(y, y_pred_xgb)]
        mse_vals = [mean_squared_error(y, y_pred_ann), mean_squared_error(y, y_pred_rf), mean_squared_error(y, y_pred_xgb)]
        r2_vals = [r2_score(y, y_pred_ann), r2_score(y, y_pred_rf), r2_score(y, y_pred_xgb)]
        tiempos_tabla = [tiempos_entrenamiento.get(m, "N/A") for m in modelos]

        df_resultados = pd.DataFrame({
            "Modelo": modelos,
            "MAE": mae_vals,
            "MSE": mse_vals,
            "R²": r2_vals,
            "Tiempo Entrenamiento (s)": tiempos_tabla
        })

        st.dataframe(df_resultados.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "R²": "{:.4f}", "Tiempo Entrenamiento (s)": "{:.2f}"}))

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        axs[0].bar(modelos, mae_vals, color='skyblue'); axs[0].set_title("MAE")
        axs[1].bar(modelos, mse_vals, color='lightgreen'); axs[1].set_title("MSE")
        axs[2].bar(modelos, r2_vals, color='orange'); axs[2].set_title("R²")
        st.pyplot(fig)

        # Gráficos pred vs real
        for i, (nombre, pred) in enumerate(zip(modelos, [y_pred_ann, y_pred_rf, y_pred_xgb])):
            fig, ax = plt.subplots(figsize=(6, 6))
            img_path = f"img/pred_vs_real_{i}.png"
            ax.scatter(y, pred, alpha=0.6, color='teal' if nombre == "ANN" else 'orange' if nombre == "Random Forest" else 'darkorange')
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
            ax.set_xlabel("Valores reales")
            ax.set_ylabel("Predicciones")
            ax.set_title(f"{nombre}: Predicción vs Real")
            ax.grid(True)
            st.pyplot(fig)
            guardar_grafico_pred_vs_real(y, pred, nombre, img_path)

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
