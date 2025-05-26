# Clasificación de AGNs con Redes Neuronales

El siguiente proyecto es para el curso final de la carrera de física pura; este se llama Investigación Científica. Este proyecto busca clasificar fuentes del catálogo 4FGL-DR4 del Fermi-LAT en distintas categorías de núcleos activos de galaxias (AGNs) mediante una red neuronal artificial (ANN).

Este combina los conocimientos tanto de física como de la carrera de ingeniería en sistemas. 

## 🔍 Objetivo

Desarrollar un modelo basado en ANN que clasifique fuentes en cinco clases:
- FSRQ
- BLL
- BCU
- Otro AGN
- No AGN

## 🧠 Estado del Proyecto

Actualmente se encuentra en las fases de:
- Optimización de hiperparámetros con Optuna y entrenamiento del modelo. 

# 🧪 Metodología

Se utiliza el enfoque **CRISP-ML** muy recomendada para estos proyectos grandes de Machine Learning , con las siguientes etapas:
1. Comprensión del tema astrofísico, sus catálogos y sus datos. (Finalizado)
2. Ingeniería de datos (Exploración y preparación de datos). (Finalizado)
3. Ingeniería de modelos de aprendizaje automático (Diseño del modelo ANN) (en proceso)
4. Implementación del modelo ANN (por realizar)
5. Evaluación de resultados (por realizar)

## 🗂️ Datos

- **Fuente principal**: [4FGL-DR4 (Fermi-LAT 14-year Source Catalog)](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/)
- **Features**: Se han identificado *17* parámetros como posibles entradas para el modelo y *1* con los labels de clases. La selección final está en curso.

## ⚙️ Herramientas y Dependencias

- Python 3.10+
- Bibliotecas:
  - numpy
  - pandas
  - matplotlib 
  - seaborn
  - scikit-learn
  - tensorflow / keras
  - astropy
  - optuna
  - os
  - time
  - livelossplot
  - joblib

## 📁 Estructura del Proyecto

AGN-Classification/

├── data/                # Datos originales y procesados

├── src/           # El source del proyecto donde se encuentran las carpetas con todo el código

├── results-deployment/              # El modelo final exportado

├── plots/             # Visualizaciones y resultados estadísticos

├── README.md



