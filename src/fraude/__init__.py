"""Detección de transacciones fraudulentas con tarjeta de crédito.

El paquete separa el pipeline en capas con una responsabilidad cada una:

- `datos`             recolección, validación del esquema y particionamiento
- `preprocesamiento`  transformación de variables y escalado sin fuga
- `modelos`           construcción de los modelos que el estudio compara
- `umbral`            elección del punto de corte por costo esperado
- `metricas`          evaluación adecuada a un desbalance extremo
"""

__version__ = "0.2.0"
