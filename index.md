# Heart Disease MLOps

Este proyecto reúne el ciclo completo de trabajo para un caso de clasificación binaria en salud, desde la exploración inicial de los datos hasta la construcción de un pipeline reproducible listo para despliegue. El objetivo es predecir si un paciente presenta riesgo de enfermedad cardiaca, cuidando especialmente aspectos como la calidad de los datos, el data leakage y la validación correcta del modelo.

A lo largo del book se encontrará el recorrido técnico del proyecto. El primero se enfoca en la exploración de datos y en detectar posibles fugas de información que podrían inflar artificialmente el rendimiento. El segundo presenta el entrenamiento del modelo con Pipeline y validación cruzada, buscando una solución más robusta, repetible y cercana a un flujo de producción.

Además del modelado, el repositorio incluye la API, la configuración para contenedores, los manifiestos de Kubernetes y el flujo de despliegue en GitHub Pages, para que el proyecto se vea como una solución MLOps completa y no solo como un experimento de notebook.

## Contenido

- Notebook 1: leakage en datos
- Notebook 2: pipeline y validacion cruzada
