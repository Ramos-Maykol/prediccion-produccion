# Predicción del Tiempo de Producción - Comparación de Modelos ML

## Descripción del Proyecto

Este proyecto tiene como objetivo analizar y comparar el rendimiento de tres modelos de machine learning (ANN, Random Forest y XGBoost) para predecir el tiempo de producción en un entorno de fabricación. El proyecto se divide en **dos fases principales**: procesamiento y entrenamiento en Google Colab, y evaluación con interfaz interactiva en Streamlit.

## Estructura del Proyecto

### 🔬 **Fase 1: Procesamiento y Entrenamiento en Google Colab**

**Descripción:**  
La primera fase se enfoca en la exploración de datos (EDA), preprocesamiento y entrenamiento de los modelos utilizando Google Colab, aprovechando la capacidad de cómputo en la nube y la integración sencilla con Google Drive.

#### Pasos realizados:

- **📊 Descarga y carga del dataset:**
  - Dataset obtenido desde Kaggle: [Machine Productivity Dataset](https://www.kaggle.com/datasets/skywalkerrr/machines-productivity)
  - **⚠️ Importante:** Para facilitar la conexión y manipulación de archivos en Colab, debes subir el archivo descargado a tu Google Drive
  - En el notebook de Colab, se monta Google Drive y se accede al archivo a través de la ruta correspondiente

- **🔍 Exploración de Datos (EDA):**
  - Estadísticas descriptivas
  - Distribución de la variable objetivo (tiempo de producción)
  - Boxplot por tipo de producto
  - Matriz de correlación
  - Relación entre unidades producidas y tiempo

- **⚙️ Preprocesamiento:**
  - Conversión de fechas y eliminación de columnas irrelevantes
  - Imputación/eliminación de valores nulos
  - Eliminación de outliers
  - Codificación de variables categóricas (one-hot)
  - Normalización de variables numéricas
  - División en conjuntos de entrenamiento y prueba

- **🤖 Entrenamiento de modelos:**
  - **Red Neuronal Artificial (ANN)**
  - **Random Forest**
  - **XGBoost**

- **💾 Exportación de resultados:**  
  Se guardan los modelos (`.h5`, `.pkl`, `.json`), el scaler (`scaler.pkl`) y los archivos de prueba (`X_test.npy`, `y_test.npy`) en Google Drive para su uso en Streamlit.

---

### 🚀 **Fase 2: Evaluación, Comparación y Despliegue en Streamlit**

**Descripción:**  
La segunda parte consiste en el desarrollo de una aplicación web interactiva utilizando **Streamlit**, donde se realiza la evaluación completa de los modelos, comparaciones estadísticas y se proporciona una interfaz para predicciones individuales.

#### Características principales:

- **📈 Evaluación de Modelos:**
  - Cálculo de métricas: MAE, MSE, R², U de Theil, tiempo de entrenamiento
  - Visualizaciones: curvas de entrenamiento, predicción vs real, importancia de variables
  - Comparación estadística mediante la prueba de Diebold-Mariano

- **📊 Comparativa de Modelos:**  
  Visualización de métricas y gráficos, incluyendo tabla comparativa, gráficos de barras y resultados de las pruebas estadísticas.

- **🎯 Predicción Individual:**  
  Formulario para ingresar las características de un producto y obtener la predicción del tiempo de producción usando el modelo seleccionado.

- **📄 Generación de Reportes PDF:**  
  Descarga de reportes comparativos y de predicción individual en formato PDF.

## 🛠️ Instalación y Ejecución

### Requerimientos

```bash
pip install streamlit pandas numpy joblib tensorflow xgboost scikit-learn matplotlib seaborn statsmodels scipy fpdf
