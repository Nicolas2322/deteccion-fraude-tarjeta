# Interfaz del detector

Tres capas, con una frontera deliberada entre ellas.

| Capa | Archivo | Responsabilidad |
|---|---|---|
| Decisión | `src/fraude/servicio.py` | Convierte una probabilidad en una acción. No sabe nada de HTTP |
| Transporte | `app/api.py` | Traduce entre peticiones HTTP y el servicio. No contiene lógica de negocio |
| Presentación | `app/web/` | Recoge datos y muestra resultados. No decide nada |

La frontera importa: durante el proyecto el modelo cambió tres veces
—reponderación retirada, regularización, cambio de familia— y ninguna de esas
veces hubo que tocar `app/web/`.

## Arranque

```
python scripts/exportar.py datos/creditcard.csv --modelo mlp --costo 5
uvicorn app.api:aplicacion --reload
```

La primera orden entrena el modelo, calcula el umbral por costo y la curva de
sensibilidad, y guarda las tres cosas juntas en `app/servicio.joblib`. Van
juntas a propósito: aplicar un escalador a un modelo entrenado con otro es el
error más fácil de cometer al pasar a producción, y además silencioso.

## Puntos de la API

| Método y ruta | Qué hace |
|---|---|
| `GET /api/estado` | Modelo, umbral vigente, curva de costo y columnas requeridas |
| `GET /api/umbral?costo=` | Umbral óptimo para un costo de alerta falsa dado |
| `POST /api/transaccion` | Veredicto de una transacción |
| `POST /api/lote` | Veredicto de un CSV completo, con resumen y cola ordenada por riesgo |

La documentación interactiva que genera FastAPI queda en `/docs`.

## Capturas

`scripts/capturas.py` las toma con un navegador real sobre la aplicación en
marcha. Las imágenes de los informes se generan así y no a mano, de modo que
cuando la interfaz cambie basta con volver a ejecutarlo.
