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
pip install -r requirements.txt
```
### 📆 Ejecución

```bash
streamlit run app.py
```



## 📊 Resultados

### Comparación de Modelos (sin U de Theil)

| Modelo        | MAE   | MSE    | R²     | Tiempo (s) |
| ------------- | ----- | ------ | ------ | ---------- |
| ANN           | 5.625 | 44.995 | -0.052 | 13.95      |
| Random Forest | 5.561 | 43.346 | -0.013 | 2.14       |
| XGBoost       | 5.761 | 47.924 | -0.120 | 0.20       |

### Coeficiente U de Theil

| Modelo        | U de Theil |
| ------------- | -----------|
| ANN           | 0.2601     |
| Random Forest | 0.2496     |
| XGBoost       | 0.2610     |

### Prueba de Diebold-Mariano

| Comparación              | Estadística | p-valor |
| ------------------------ | ----------- | ------- |
| ANN vs Random Forest     | -1.4407     | 0.1505  |
| ANN vs XGBoost           | 1.7200      | 0.0862  |
| Random Forest vs XGBoost | 3.6825      | 0.0003  |

---

## 📄 Reportes Generados

### 1. Reporte Comparativo de Modelos

Documenta el análisis y evaluación detallada de los tres modelos predictivos, incluyendo:

* Visualizaciones exploratorias (EDA)
* Preprocesamiento de datos
* Comparación de métricas (MAE, MSE, R², Tiempo de entrenamiento)
* Análisis del Coeficiente U de Theil
* Resultados de las pruebas de Diebold-Mariano

### 2. Reporte de Predicción Individual

Se genera cuando el usuario ingresa los datos de un nuevo producto en la interfaz de Streamlit. Muestra la predicción del tiempo de producción utilizando el modelo seleccionado.

---

## 🏆 Conclusión

Según las métricas evaluadas y los resultados de la prueba de Diebold-Mariano, el modelo recomendado es **Random Forest**. Este modelo mostró el mejor rendimiento general, reflejado en sus valores de R², menor MAE/MSE y consistencia en las pruebas estadísticas.

---

## 📋 Notas Importantes

* **Dataset**: Debes descargar manualmente el dataset desde [Kaggle](https://www.kaggle.com/) y subirlo a tu Google Drive antes de iniciar el pipeline en Colab.
* **Separación de fases:**

  * *Colab:* EDA, preprocesamiento y entrenamiento de modelos.
  * *Streamlit:* Evaluación, comparación estadística, visualizaciones y predicción interactiva.
* **Reproducibilidad**: El código de Colab está documentado paso a paso para facilitar la reproducción del entrenamiento de modelos.

---

## 👨‍💼 Autor

**Maykol Ramos - Rodriguez Leon**
Universidad Nacional de Trujillo (UNT) - Tesis 2025

---

## 📂 Estructura de Archivos

```
proyecto/
├── datos/
│   ├── modelos/
│   │   ├── ann_model.h5
│   │   ├── random_forest_model.pkl
│   │   └── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── X_test.npy
│   └── y_test.npy
├── img
├── reporte
├── app.py
├── requirements.txt
├── train_models.ipynb
└── README.md
```

---

## 🔗 Enlaces

* Dataset en Kaggle: [https://www.kaggle.com/datasets/skywalkerrr/machines-productivity](https://www.kaggle.com/datasets/skywalkerrr/machines-productivity)
