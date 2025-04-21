# DATA ANALYSIS & EXPLAINABILITY - FootySage

<div style="display: flex; align-items: center;"> <div style="flex: 1; margin-right: 10px;"> Este módulo del proyecto <strong>FootySage</strong> se centra en la creación de modelos predictivos para partidos de fútbol profesional y el análisis de su explicabilidad. <p>Se utilizan datos de <em>Open Data</em> de <strong>StatsBomb</strong> para predecir el resultado de partidos (local, empate, visitante) en las cinco grandes ligas europeas y entender qué factores influyen más en dichas predicciones.</p> </div> <div style="flex-shrink: 0;"> <img src="https://i0.wp.com/lamediainglesa.com/wp-content/uploads/2020/01/statsbomb_la_media_inglesa.jpg?fit=1000%2C523&ssl=1" alt="StatsBomb Logo" width="200"> </div> </div>

## Objetivo general

- Predecir el resultado de partidos durante la temporada 2015/2016 en: La Liga, Premier League, Serie A, Ligue 1, 1. Bundesliga y en las cinco en general
- Analizar la explicabilidad de los modelos más efectivos usando SHAP, para identificar las características más determinantes en cada liga.

## Estructura del proyecto

- `data/`
  - `processed/`: Partidos procesados con variables limpias.
  - `reduced/`: Datos finales listos para el modelado.

- `models/`: Contiene los mejores modelos entrenados para cada competición.

- `src/`: Contiene el core del procesamiento y análisis:
  - Preprocesamiento y extracción desde la API de StatsBomb.
  - Funciones para entrenamiento, evaluación y análisis SHAP.

- Notebooks:
  - `0_datasets_building.ipynb`: Construcción de datasets.
  - `1_experimentation.ipynb`: Entrenamiento y evaluación de modelos por liga.
  - `2_analysis_<Liga>.ipynb`: Análisis de explicabilidad individual.
  - `2_analysis_Top5Leagues.ipynb`: Análisis de explicabilidad de las ligas de manera global.

##  Preparación de Datos y Modelado

- **Modelos**: Se prueban distintos algoritmos de ML (Random Forest, Logistic Regression, etc.) con distintas técnicas de preprocesado (PCA, MI, oversampling, etc.).
- **Validación**: Se evalúan métricas como Accuracy, F1-score y matriz de confusión por liga.
- **Features**: Variables tácticas, estadísticas y de contexto del partido.

## Explicabilidad con SHAP

Se emplea SHAP para entender:
  - ¿Qué variables tienen más peso para predecir una victoria local?
  - ¿Qué factores diferencian cada liga?
  - ¿Qué influye en el rendimiento de un equipo específico?

## Resultados Clave

- **Precisión general**: Modelos alcanzan buenas tasas de acierto (~61%-72% dependiendo la liga).
- **Comparativa entre ligas**: Diferencias interesantes en el peso de las variables.
- **Aplicabilidad**: Este trabajo sirvió de base para el desarrollo de la app FootySage.

