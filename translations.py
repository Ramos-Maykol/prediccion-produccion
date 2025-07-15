# translations.py

TRANSLATIONS = {
    "app_title": {
        "es": "🚀 **Optimiza tu Producción** | Análisis y Predicción de Tiempos",
        "en": "🚀 **Optimize Your Production** | Analysis and Prediction of Times"
    },
    "app_intro": {
        "es": "¡Bienvenido al futuro de la **gestión de producción**! Esta herramienta avanzada te permite comparar el rendimiento de diferentes modelos predictivos y obtener estimaciones precisas del tiempo de producción para tus productos.",
        "en": "Welcome to the future of **production management**! This advanced tool allows you to compare the performance of different predictive models and obtain precise production time estimates for your products."
    },
    "hardware_details_button": {
        "es": "Detalles Hardware",
        "en": "Hardware Details"
    },
    "hardware_details_button_help": {
        "es": "Haz clic para ver la configuración de hardware de entrenamiento.",
        "en": "Click to view the training hardware configuration."
    },
    "hardware_modal_title": {
        "es": "Detalles de la Máquina de Entrenamiento",
        "en": "Training Machine Details"
    },
    "hardware_modal_content": {
        "es": """
        <div style="font-size: 1.0em; line-height: 1.6;">
            <p>Los modelos fueron entrenados utilizando la siguiente configuración de hardware:</p>
            <p><strong>Procesador (CPU):</strong> Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4 núcleos, 8 procesadores lógicos)</p>
            <p><strong>Memoria RAM:</strong> 7.8 GB (aproximadamente 8 GB) DDR4 @ 2933 MHz (1 de 2 ranuras usadas, SODIMM)</p>
            <p><strong>Tarjetas Gráficas (GPU):</strong></p>
            <ul>
                <li>GPU 0 (Integrada): Intel(R) UHD Graphics (Memoria compartida: 3.9 GB)</li>
                <li>GPU 1 (Dedicada): NVIDIA GeForce GTX 1050 (Memoria dedicada: 3.0 GB)</li>
            </ul>
            <p><strong>Almacenamiento:</strong></p>
            <ul>
                <li>Disco 0 (E: D:): TOSHIBA MQ04ABF100 (HDD - 932 GB)</li>
                <li>Disco 1 (C:): KINGSTON SNVS500G (SSD - 466 GB)</li>
            </ul>
            <p style="font-style: italic; color: #555;">Estos detalles aseguran la reproducibilidad y el contexto del entrenamiento de los modelos.</p>
        </div>
        """,
        "en": """
        <div style="font-size: 1.0em; line-height: 1.6;">
            <p>The models were trained using the following hardware configuration:</p>
            <p><strong>Processor (CPU):</strong> Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4 cores, 8 logical processors)</p>
            <p><strong>RAM:</strong> 7.8 GB (approximately 8 GB) DDR4 @ 2933 MHz (1 of 2 slots used, SODIMM)</p>
            <p><strong>Graphics Cards (GPU):</strong></p>
            <ul>
                <li>GPU 0 (Integrated): Intel(R) UHD Graphics (Shared memory: 3.9 GB)</li>
                <li>GPU 1 (Dedicated): NVIDIA GeForce GTX 1050 (Dedicated memory: 3.0 GB)</li>
            </ul>
            <p><strong>Storage:</strong></p>
            <ul>
                <li>Disk 0 (E: D:): TOSHIBA MQ04ABF100 (HDD - 932 GB)</li>
                <li>Disk 1 (C:): KINGSTON SNVS500G (SSD - 466 GB)</li>
            </ul>
            <p style="font-style: italic; color: #555;">These details ensure reproducibility and the context of model training.</p>
        </div>
        """
    },
    "tab1_title": {
        "es": "📊 **Análisis Comparativo**",
        "en": "📊 **Comparative Analysis**"
    },
    "tab2_title": {
        "es": "✨ **Predicción Instantánea**",
        "en": "✨ **Instant Prediction**"
    },
    "tab1_header": {
        "es": "✨ Desempeño de Modelos: Una Visión Profunda",
        "en": "✨ Model Performance: A Deep Dive"
    },
    "tab1_intro": {
        "es": "Aquí podrás explorar cómo nuestros diferentes modelos de Machine Learning (Red Neuronal Artificial, Random Forest y XGBoost) se desempeñan al predecir el tiempo de producción. Analizamos métricas clave y comparaciones estadísticas para darte la imagen más clara.",
        "en": "Here you can explore how our different Machine Learning models (Artificial Neural Network, Random Forest, and XGBoost) perform in predicting production time. We analyze key metrics and statistical comparisons to give you the clearest picture."
    },
    "tab1_warning_files": {
        "es": "🚨 **Nota Importante:** Para el análisis comparativo, necesitamos los archivos `X_test.npy` y `y_test.npy` en la carpeta `datos/`.",
        "en": "🚨 **Important Note:** For comparative analysis, we need the `X_test.npy` and `y_test.npy` files in the `datos/` folder."
    },
    "tab1_subheader_viz": {
        "es": "Visualización: Real vs. Predicho 🎯",
        "en": "Visualization: Actual vs. Predicted 🎯"
    },
    "tab1_info_viz": {
        "es": "Observa cómo cada modelo alinea sus predicciones con los valores reales. ¡Cuanto más cerca de la línea roja, mejor!",
        "en": "Observe how each model aligns its predictions with the actual values. The closer to the red line, the better!"
    },
    "tab1_subheader_metrics": {
        "es": "Métricas de Evaluación: Rendimiento en Cifras 📊",
        "en": "Evaluation Metrics: Performance in Figures 📊"
    },
    "tab1_info_metrics": {
        "es": "Un resumen rápido de la precisión y el ajuste de cada modelo. **R² más alto es mejor**, mientras que **MAE/MSE más bajos indican mayor precisión**.",
        "en": "A quick summary of each model's precision and fit. **Higher R² is better**, while **lower MAE/MSE indicate higher precision**."
    },
    "tab1_subheader_theil": {
        "es": "Coeficiente U de Theil: ¿Qué tan bueno es el pronóstico? 📉",
        "en": "Theil's U Coefficient: How good is the forecast? 📉"
    },
    "tab1_info_theil": {
        "es": "Un valor **cercano a 0** indica una predicción perfecta, mientras que un valor **cercano a 1** sugiere que el modelo no es mejor que una predicción ingenua.",
        "en": "A value **close to 0** indicates a perfect prediction, while a value **close to 1** suggests that the model is no better than a naive prediction."
    },
    "tab1_subheader_diebold": {
        "es": "Pruebas de Diebold-Mariano: ¿Hay un ganador claro? 🏆",
        "en": "Diebold-Mariano Tests: Is there a clear winner? 🏆"
    },
    "tab1_info_diebold": {
        "es": """
        Esta prueba compara estadísticamente los errores de dos modelos.
        - **Estadística DM > 0**: El primer modelo es mejor.
        - **Estadística DM < 0**: El segundo modelo es mejor.
        - **p-valor < 0.05**: La diferencia es estadísticamente **significativa**.
        """,
        "en": """
        This test statistically compares the errors of two models.
        - **DM Statistic > 0**: The first model is better.
        - **DM Statistic < 0**: The second model is better.
        - **p-value < 0.05**: The difference is statistically **significant**.
        """
    },
    "tab1_subheader_report": {
        "es": "¡Descarga tu Informe Detallado! 📄",
        "en": "Download Your Detailed Report! 📄"
    },
    "tab1_success_report": {
        "es": "Obtén un PDF completo con todos los análisis, gráficos y métricas para compartir o archivar.",
        "en": "Get a complete PDF with all analyses, graphs, and metrics to share or archive."
    },
    "tab1_button_generate_report": {
        "es": "🚀 Generar Reporte PDF Comparativo",
        "en": "🚀 Generate Comparative PDF Report"
    },
    "tab1_spinner_generating_report": {
        "es": "Generando tu reporte... ¡Esto puede tardar un momento! ⏳",
        "en": "Generating your report... This may take a moment! ⏳"
    },
    "tab1_download_button": {
        "es": "📥 Descargar Reporte PDF",
        "en": "📥 Download PDF Report"
    },
    "tab1_report_success": {
        "es": "¡Reporte generado y listo para descargar!",
        "en": "Report generated and ready to download!"
    },
    "tab1_report_error": {
        "es": "❌ ¡Ups! Hubo un problema al generar el reporte PDF.",
        "en": "❌ Oops! There was a problem generating the PDF report."
    },
    "tab1_error_files_missing": {
        "es": "❌ ¡ERROR! Parece que faltan los archivos de datos de prueba (`X_test.npy` o `y_test.npy`) en la carpeta `datos/`.",
        "en": "❌ ERROR! It seems that the test data files (`X_test.npy` or `y_test.npy`) are missing from the `datos/` folder."
    },
    "tab1_info_files_missing": {
        "es": "Por favor, asegúrate de que estos archivos existan para poder realizar el análisis comparativo.",
        "en": "Please ensure that these files exist to perform the comparative analysis."
    },
    "tab1_error_unexpected": {
        "es": "❌ ¡Se encontró un error inesperado al realizar el análisis comparativo! Por favor, verifica tus datos y modelos. Detalles: ",
        "en": "❌ An unexpected error occurred while performing the comparative analysis! Please check your data and models. Details: "
    },
    "tab2_header": {
        "es": "✨ Predicción Individual: Estima el Tiempo de Producción",
        "en": "✨ Individual Prediction: Estimate Production Time"
    },
    "tab2_intro": {
        "es": "Ingresa los detalles de tu producto para obtener una **estimación instantánea** del tiempo de producción. ¡Planifica mejor tus operaciones!",
        "en": "Enter your product details to get an **instant estimate** of production time. Plan your operations better!"
    },
    "form_header_details": {
        "es": "#### Detalles del Producto a Fabricar 📦",
        "en": "#### Product Details 📦"
    },
    "input_volume": {
        "es": "Volumen (m³)",
        "en": "Volume (m³)"
    },
    "help_volume": {
        "es": "Volumen total en metros cúbicos de la unidad a producir.",
        "en": "Total volume in cubic meters of the unit to be produced."
    },
    "input_product_type": {
        "es": "Tipo de Producto",
        "en": "Product Type"
    },
    "options_product_type": {
        "es": ["Automotriz", "Electrónica", "Muebles", "Textiles"],
        "en": ["Automotive", "Electronics", "Furniture", "Textiles"]
    },
    "help_product_type": {
        "es": "Categoría a la que pertenece el producto.",
        "en": "Category to which the product belongs."
    },
    "input_quantity": {
        "es": "Cantidad a Fabricar",
        "en": "Quantity to Manufacture"
    },
    "help_quantity": {
        "es": "Número de unidades idénticas a producir.",
        "en": "Number of identical units to be produced."
    },
    "form_header_model_selection": {
        "es": "#### Selección del Modelo Predictivo 🤖",
        "en": "#### Predictive Model Selection 🤖"
    },
    "select_model": {
        "es": "¿Qué modelo quieres usar para la predicción principal?",
        "en": "Which model do you want to use for the main prediction?"
    },
    "help_select_model": {
        "es": "Elige el modelo que mejor se adapte a tus necesidades. Random Forest suele ofrecer un buen equilibrio.",
        "en": "Choose the model that best suits your needs. Random Forest often offers a good balance."
    },
    "button_predict": {
        "es": "🚀 ¡Predecir Tiempo Ahora!",
        "en": "🚀 Predict Time Now!"
    },
    "results_header": {
        "es": "Resultados de la Predicción ✨",
        "en": "Prediction Results ✨"
    },
    "selected_model_msg": {
        "es": "**Con el modelo {model_name}:**",
        "en": "**With the {model_name} model:**"
    },
    "metric_time_per_unit": {
        "es": "Tiempo Estimado por Unidad (horas)",
        "en": "Estimated Time per Unit (hours)"
    },
    "metric_total_time": {
        "es": "Tiempo Total Estimado ({quantity} unidades) (horas)",
        "en": "Total Estimated Time ({quantity} units) (hours)"
    },
    "success_same_day": {
        "es": "✅ **¡Excelentes noticias!** Su producto podría estar listo **el mismo día**.",
        "en": "✅ **Excellent news!** Your product could be ready **the same day**."
    },
    "info_days_needed": {
        "es": "⏳ La producción de sus {quantity} unidades tomará aproximadamente **{days} día(s) laborable(s)**.",
        "en": "⏳ The production of your {quantity} units will take approximately **{days} business day(s)**."
    },
    "info_pickup_date": {
        "es": "**Fecha estimada de recojo:** {date}",
        "en": "**Estimated pickup date:** {date}"
    },
    "comparison_header": {
        "es": "Comparativa Rápida con Otros Modelos 📊",
        "en": "Quick Comparison with Other Models 📊"
    },
    "comparison_info": {
        "es": "Mira cómo los otros modelos habrían predicho el tiempo total para tu cantidad.",
        "en": "See how other models would have predicted the total time for your quantity."
    },
    "comparison_chart_title": {
        "es": "Comparativa de Predicciones por Modelo",
        "en": "Prediction Comparison by Model"
    },
    "comparison_chart_ylabel": {
        "es": "Tiempo Total Estimado (horas)",
        "en": "Total Estimated Time (hours)"
    },
    "button_download_individual_report": {
        "es": "Descargar Reporte de Predicción Individual",
        "en": "Download Individual Prediction Report"
    },
    "individual_report_success": {
        "es": "¡Reporte individual generado y listo para descargar!",
        "en": "Individual report generated and ready to download!"
    },
    "individual_report_error": {
        "es": "❌ ¡Ups! Hubo un problema al generar el reporte PDF individual.",
        "en": "❌ Oops! There was a problem generating the individual PDF report."
    },
    # Para el PDF individual
    "pdf_individual_report_title": {
        "es": "Reporte de Predicción Individual",
        "en": "Individual Prediction Report"
    },
    "pdf_individual_report_date_prefix": {
        "es": "Fecha y hora de generación:",
        "en": "Date and time generated:"
    },
    "pdf_input_data_header": {
        "es": "Datos Ingresados:",
        "en": "Entered Data:"
    },
    "pdf_product_type_label": {
        "es": "Tipo de Producto",
        "en": "Product Type"
    },
    # Añadido para el mapeo de opciones de producto en el PDF
    "pdf_product_type_option_Automotive": {"es": "Automotriz", "en": "Automotive"},
    "pdf_product_type_option_Electronics": {"es": "Electrónica", "en": "Electronics"},
    "pdf_product_type_option_Furniture": {"es": "Muebles", "en": "Furniture"},
    "pdf_product_type_option_Textiles": {"es": "Textiles", "en": "Textiles"},

    "pdf_volume_label": {
        "es": "Volumen (m³)",
        "en": "Volume (m³)"
    },
    "pdf_quantity_label": {
        "es": "Cantidad a Fabricar",
        "en": "Quantity to Manufacture"
    },
    "pdf_time_per_unit_label": {
        "es": "Tiempo por Unidad (h)",
        "en": "Time per Unit (h)"
    },
    "pdf_total_time_label": {
        "es": "Tiempo Total Estimado (h)",
        "en": "Total Estimated Time (h)"
    },
    "pdf_prediction_header": {
        "es": "Predicción de Tiempo de Producción:",
        "en": "Production Time Prediction:"
    },
    "pdf_model_selected_prefix": {
        "es": "Modelo seleccionado:",
        "en": "Model selected:"
    },
    "pdf_time_per_unit_estimated": {
        "es": "Tiempo estimado por unidad: {time:.2f} horas",
        "en": "Estimated time per unit: {time:.2f} hours"
    },
    "pdf_comparison_header": {
        "es": "Comparación con otros modelos:",
        "en": "Comparison with other models:"
    },
    "pdf_recommendation_header": {
        "es": "Recomendación:",
        "en": "Recommendation:"
    },
    "pdf_recojo_same_day": {
        "es": "✅ Su producto estará listo **el mismo día**. Por favor, acérquese al finalizar la jornada laboral.",
        "en": "✅ Your product will be ready **the same day**. Please pick it up at the end of the workday."
    },
    "pdf_recojo_days": {
        "es": "⏳ Su producto estará listo en aproximadamente **{days} día(s) laborable(s)** (considerando 8 horas de trabajo por día).\n**Fecha estimada de recojo:** {date}",
        "en": "⏳ Your product will be ready in approximately **{days} business day(s)** (considering 8 working hours per day).\n**Estimated pickup date:** {date}"
    },
    "pdf_footer": {
        "es": "Generado por Maykol Ramos - Rodrigez Leon | UNT - Tesis 2025",
        "en": "Generated by Maykol Ramos - Rodriguez Leon | UNT - Thesis 2025"
    },
    # Para el PDF Comparativo
    "pdf_comparative_report_title": {
        "es": "Informe Comparativo de Modelos de Predicción",
        "en": "Comparative Model Prediction Report"
    },
    "pdf_comparative_intro": {
        "es": "Este informe documenta el análisis y evaluación de modelos predictivos aplicados al tiempo de producción. Se evaluaron los modelos Red Neuronal Artificial (ANN), Random Forest y XGBoost. Las métricas utilizadas incluyen MAE, MSE, R² y tiempo de entrenamiento. Además, se analizó el coeficiente U de Theil y se realizó una comparación estadística usando la prueba de Diebold-Mariano.",
        "en": "This report documents the analysis and evaluation of predictive models applied to production time. Artificial Neural Network (ANN), Random Forest, and XGBoost models were evaluated. Metrics used include MAE, MSE, R², and training time. Additionally, Theil's U coefficient was analyzed, and a statistical comparison was performed using the Diebold-Mariano test."
    },
    "pdf_machine_details_header": {
        "es": "1. Detalles de la Máquina de Entrenamiento",
        "en": "1. Training Machine Details"
    },
    "pdf_machine_details_intro": {
        "es": "Los modelos predictivos fueron entrenados utilizando la siguiente configuración de hardware:",
        "en": "The predictive models were trained using the following hardware configuration:"
    },
    "pdf_cpu_label": {
        "es": "Procesador (CPU):",
        "en": "Processor (CPU):"
    },
    "pdf_cpu_desc": {
        "es": "Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4 núcleos, 8 procesadores lógicos)",
        "en": "Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz (4 cores, 8 logical processors)"
    },
    "pdf_ram_label": {
        "es": "Memoria RAM:",
        "en": "RAM:"
    },
    "pdf_ram_desc": {
        "es": "7.8 GB (aproximadamente 8 GB) DDR4 @ 2933 MHz (1 de 2 ranuras usadas, SODIMM)",
        "en": "7.8 GB (approximately 8 GB) DDR4 @ 2933 MHz (1 of 2 slots used, SODIMM)"
    },
    "pdf_gpu_label": {
        "es": "Tarjetas Gráficas (GPU):",
        "en": "Graphics Cards (GPU):"
    },
    "pdf_gpu0_desc": {
        "es": "GPU 0 (Integrada): Intel(R) UHD Graphics (Memoria compartida: 3.9 GB)",
        "en": "GPU 0 (Integrated): Intel(R) UHD Graphics (Shared memory: 3.9 GB)"
    },
    "pdf_gpu1_desc": {
        "es": "GPU 1 (Dedicada): NVIDIA GeForce GTX 1050 (Memoria dedicada: 3.0 GB)",
        "en": "GPU 1 (Dedicated): NVIDIA GeForce GTX 1050 (Dedicated memory: 3.0 GB)"
    },
    "pdf_storage_label": {
        "es": "Almacenamiento:",
        "en": "Storage:"
    },
    "pdf_storage0_desc": {
        "es": "Disco 0 (E: D:): TOSHIBA MQ04ABF100 (HDD - 932 GB)",
        "en": "Disk 0 (E: D:): TOSHIBA MQ04ABF100 (HDD - 932 GB)"
    },
    "pdf_storage1_desc": {
        "es": "Disco 1 (C:): KINGSTON SNVS500G (SSD - 466 GB)",
        "en": "Disk 1 (C:): KINGSTON SNVS500G (SSD - 466 GB)"
    },
    "pdf_eda_header": {
        "es": "2. Visualizaciones Exploratorias (EDA)",
        "en": "2. Exploratory Visualizations (EDA)"
    },
    "pdf_eda_hist": {
        "es": "Distribución del tiempo de producción",
        "en": "Production time distribution"
    },
    "pdf_eda_boxplot": {
        "es": "Boxplot por tipo de producto",
        "en": "Boxplot by product type"
    },
    "pdf_eda_corr": {
        "es": "Matriz de correlación",
        "en": "Correlation matrix"
    },
    "pdf_eda_scatter": {
        "es": "Unidades producidas vs tiempo",
        "en": "Units produced vs time"
    },
    "pdf_eda_img_not_found": {
        "es": "No se encontró {img_name}",
        "en": "{img_name} not found"
    },
    "pdf_preprocessing_header": {
        "es": "3. Preprocesamiento de Datos",
        "en": "3. Data Preprocessing"
    },
    "pdf_preprocessing_desc": {
        "es": "- Conversión de fechas\n- Imputación de valores nulos\n- Eliminación de outliers\n- Codificación de variables categóricas\n- Normalización\n- División en conjunto de entrenamiento y prueba",
        "en": "- Date conversion\n- Null value imputation\n- Outlier removal\n- Categorical variable encoding\n- Normalization\n- Split into training and test set"
    },
    "pdf_model_comparison_header": {
        "es": "4. Comparación de Modelos (MAE, MSE, R², Tiempo)",
        "en": "4. Model Comparison (MAE, MSE, R², Time)"
    },
    "pdf_model_comparison_intro": {
        "es": "Resultados de las métricas evaluadas:",
        "en": "Results of the evaluated metrics:"
    },
    "pdf_pred_vs_real_header": {
        "es": "Gráficos Predicción vs Real",
        "en": "Prediction vs Actual Graphs"
    },
    "pdf_table_model": {
        "es": "Modelo",
        "en": "Model"
    },
    "pdf_table_time_s": {
        "es": "Tiempo (s)",
        "en": "Time (s)"
    },
    "pdf_theil_header": {
        "es": "5. Coeficiente U de Theil",
        "en": "5. Theil's U Coefficient"
    },
    "pdf_theil_table_u_theil": {
        "es": "U de Theil",
        "en": "Theil's U"
    },
    "pdf_diebold_header": {
        "es": "6. Pruebas de Diebold-Mariano",
        "en": "6. Diebold-Mariano Tests"
    },
    "pdf_diebold_table_comparison": {
        "es": "Comparación",
        "en": "Comparison"
    },
    "pdf_diebold_table_statistic": {
        "es": "Estadística",
        "en": "Statistic"
    },
    "pdf_diebold_table_p_value": {
        "es": "p-valor",
        "en": "p-value"
    },
    "pdf_conclusion_header": {
        "es": "7. Conclusión",
        "en": "7. Conclusion"
    },
    "pdf_conclusion_text": {
        "es": "Según las métricas evaluadas, el modelo recomendado es: {best_model_name}. Este modelo mostró el mejor rendimiento general, reflejado en sus valores de R², menor MAE/MSE y consistencia en las pruebas estadísticas.",
        "en": "According to the evaluated metrics, the recommended model is: {best_model_name}. This model showed the best overall performance, reflected in its R² values, lower MAE/MSE, and consistency in statistical tests."
    },
    "sidebar_config_title": {
        "es": "Configuración",
        "en": "Settings"
    },
    "sidebar_lang_header": {
        "es": "Idioma / Language",
        "en": "Language / Idioma"
    },
    "spinner_loading_models": {
        "es": "Cargando modelos...",
        "en": "Loading models..."
    },
    "error_loading_models_prefix": {
        "es": "Error al cargar los modelos:",
        "en": "Error loading models:"
    },
    "spinner_loading_scaler": {
        "es": "Cargando escalador...",
        "en": "Loading scaler..."
    },
    "error_loading_scaler_prefix": {
        "es": "Error al cargar el escalador:",
        "en": "Error loading scaler:"
    },
    "spinner_loading_training_times": {
        "es": "Cargando tiempos de entrenamiento...",
        "en": "Loading training times..."
    },
    "warning_loading_training_times_prefix": {
        "es": "Advertencia al cargar tiempos de entrenamiento:",
        "en": "Warning loading training times:"
    },
    "pdf_plot_xlabel_real": {
        "es": "Valores Reales",
        "en": "Actual Values"
    },
    "pdf_plot_ylabel_pred": {
        "es": "Valores Predichos",
        "en": "Predicted Values"
    },
    "pdf_plot_title_pred_vs_real": {
        "es": "Predicción vs Real",
        "en": "Prediction vs Actual"
    },
    "pdf_plot_line_ideal": {
        "es": "Línea ideal",
        "en": "Ideal line"
    },
    "error_scaling_input_data_prefix": {
        "es": "Error al escalar los datos de entrada:",
        "en": "Error scaling input data:"
    }
}