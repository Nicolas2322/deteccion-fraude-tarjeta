# Detección de transacciones fraudulentas

Redes neuronales tabulares frente a *Gradient Boosting* sobre transacciones
reales con tarjeta, con el umbral de decisión elegido por costo económico
esperado en lugar de por una métrica abstracta.

**Demostración en vivo:** <https://nicolas2322.github.io/deteccion-fraude-tarjeta/>
— el modelo se ejecuta en tu navegador; ninguna transacción sale de tu equipo.

| | |
|---|---|
| Asignatura | Aplicaciones en Deep Learning, Especialización en Inteligencia Artificial |
| Institución | Universidad de Cundinamarca |
| Autor | Yohan Nicolás Gómez Castañeda |
| Datos | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Université Libre de Bruxelles |
| Estado | Pipeline, modelos, interfaz y despliegue completos; búsqueda de hiperparámetros pendiente |

---

## El problema

284.807 transacciones europeas de dos días de septiembre de 2013, de las cuales
492 son fraude: un **0,173 %**. Con ese desbalance, un clasificador que responda
siempre «no hay fraude» acierta el 99,83 % de las veces y no sirve para nada.

De ahí se derivan las tres decisiones que ordenan todo el proyecto:

| Decisión | Motivo |
|---|---|
| La métrica principal es **AUPRC**, no exactitud ni ROC-AUC | Con esta prevalencia es la única que distingue un modelo útil de uno trivial (Saito y Rehmsmeier, 2015) |
| El umbral se elige **minimizando el costo esperado** | Bloquear un fraude de 2.000 y uno de 5 no valen lo mismo; el criterio correcto es económico, no estadístico (Elkan, 2001) |
| La evaluación usa **dos protocolos** y reporta los dos | Validación cruzada estratificada y prueba sobre el tramo temporal reservado; la distancia entre ambas mide el optimismo de una partición aleatoria sobre lo que en realidad es un flujo |

La pregunta de investigación es si la ventaja que los árboles suelen tener sobre
las redes en datos tabulares se reduce aquí. Grinsztajn et al. (2022) sostienen
que esa ventaja procede de que los árboles parten según los ejes, y que decae
sobre datos rotados. Las variables `V1`–`V28` son componentes principales, es
decir, exactamente una rotación.

**La respuesta medida: no solo se reduce, se invierte.**

---

## Resultados

Tramo de prueba temporal, costo de alerta falsa igual a cinco. Línea base del
AUPRC: 0,0013. Informes completos en [`informes/`](informes/), figuras en
[`figuras/`](figuras/).

| | Logística | LightGBM | Perceptrón | FT-Transformer |
|---|---|---|---|---|
| AUPRC, prueba temporal | 0,7576 | 0,6663 | **0,8175** | 0,7949 |
| AUPRC, validación cruzada | 0,7289 | 0,7943 | 0,8335 | 0,6998 |
| ROC-AUC | 0,9856 | 0,8916 | 0,9678 | 0,9703 |
| Error de calibración | 0,1109 | 0,0009 | 0,0008 | **0,0004** |
| Fraudes detectados (de 75) | 55 | 50 | **59** | 53 |
| Alertas falsas | 19 | 62 | 20 | **11** |
| Costo total | 2.982 | 2.735 | **2.499** | 3.978 |

> **Ninguna de estas cifras procede de una búsqueda de hiperparámetros.** Valen
> como pruebas iniciales, no como resultado final. La razón para posponer esa
> búsqueda está en la sección siguiente.

### Una cifra única de AUPRC no es reproducible aquí

[`scripts/estabilidad.py`](scripts/estabilidad.py) entrena seis veces sobre
submuestras aleatorias del 80 % del tramo de desarrollo y evalúa siempre sobre
el mismo tramo de prueba. Lo único que cambia entre repeticiones es qué filas
vio el modelo; los ajustes son deterministas y hay una prueba que lo comprueba.

| Configuración | Media | Desviación | Rango |
|---|---|---|---|
| LightGBM, 500 árboles, 31 hojas | 0,3904 | 0,2813 | [0,1381 – 0,8209] |
| LightGBM, 200 árboles, 15 hojas, mín. 50 por hoja | 0,7014 | 0,0427 | [0,6472 – 0,7497] |
| Regresión logística | 0,7543 | 0,0102 | [0,7414 – 0,7699] |
| Perceptrón multicapa | 0,8179 | 0,0052 | [0,8103 – 0,8236] |

El ensamble profundo puede dar el peor o el mejor AUPRC de todo el estudio sin
que nada cambie salvo la submuestra. Con 417 fraudes en desarrollo, quitar al
azar un 20 % de las filas se lleva unos 83 de ellos, y 500 árboles con 31 hojas
tienen capacidad de sobra para memorizar casos concretos.

Consecuencias, ya aplicadas:

1. **LightGBM va regularizado por defecto.** La media sube de 0,3904 a 0,7014,
   la desviación baja de 0,2813 a 0,0427 y la cifra pasa a ser reproducible
   entre ejecuciones. Restringir más vuelve a empeorar ambas.
2. **El informe final reportará media, desviación y rango**, no cifras únicas.
3. **La búsqueda de hiperparámetros espera** a que la medición sea estable:
   buscar sobre una puntuación con desviación de 0,28 seleccionaría la
   configuración más afortunada, no la mejor.

El ROC-AUC de la configuración inestable era 0,8916 y la exactitud rondaba el
99,8 %. Solo el AUPRC delató el problema.

### Por qué LightGBM no repondera la clase minoritaria

Activar `is_unbalance` saturaba las probabilidades: 3.013 transacciones del
tramo de prueba empatadas en exactamente 1,0, AUPRC de 0,0147 frente a 0,7719
sin reponderación, y exhaustividad nula porque el umbral por costo no hallaba
ningún corte dentro del empate. Es el efecto que describen Dal Pozzolo et al.
(2015): reponderar desplaza la probabilidad, y aquí la colapsó.

---

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/Nicolas2322/deteccion-fraude-tarjeta
cd deteccion-fraude-tarjeta
pip install -r requirements.txt
pip install -e .
```

Las dependencias están ancladas a versión exacta en
[`requirements.txt`](requirements.txt). El paquete declara extras opcionales,
por si se prefiere instalar solo lo necesario:

| Extra | Qué añade | Para qué |
|---|---|---|
| *(núcleo)* | numpy, pandas, scikit-learn, pyyaml | Pipeline completo y línea base lineal |
| `arboles` | lightgbm, xgboost | Familia de árboles |
| `redes` | torch | FT-Transformer |
| `web` | fastapi, uvicorn, joblib | Interfaz e API |
| `dev` | pytest, pytest-cov, ruff | Verificación |

```bash
pip install -e ".[arboles,redes,web,dev]"
```

### Los datos

El archivo completo pesa unos 150 MB y no está versionado. Se descarga de
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y se deja en
`datos/`:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p datos --unzip
```

El repositorio incluye en [`datos_muestra/`](datos_muestra/) una muestra de
5.000 transacciones con los 492 fraudes, suficiente para inspeccionar el
esquema y ejecutar el pipeline sin descargar nada. **No es representativa**
—su prevalencia es del 9,84 % frente al 0,173 % real— y no sirve para
reproducir métricas; su propio README lo advierte.

---

## Uso

### Explorar y validar el conjunto

```bash
python scripts/explorar.py datos/creditcard.csv
```

```
Conjunto completo : 284,807 transacciones | 492 fraudes (0.173%) | 48.0 h
Desarrollo        : 227,846 transacciones | 417 fraudes (0.183%) | 40.3 h
Prueba temporal   :  56,961 transacciones |  75 fraudes (0.132%) |  7.7 h
```

### Entrenar y evaluar un modelo

```bash
python scripts/evaluar.py datos/creditcard.csv --modelo mlp --costo 5 --cache cache
```

```
Modelo: mlp   costo de alerta falsa: 5.0
  Umbral elegido en desarrollo : 0.207394
  AUPRC  validación cruzada    : 0.8335  (línea base 0.0018)
  AUPRC  prueba temporal       : 0.8175
  Exhaustividad en prueba      : 0.787
  Alertas falsas en prueba     : 20
  Costo total en prueba        : 2498.79
```

Modelos disponibles: `logistica`, `lightgbm`, `xgboost`, `mlp`,
`ft_transformer`. `--cache` guarda las predicciones de cada pliegue para que un
entrenamiento caro se pueda reanudar; el FT-Transformer tarda media hora sin
GPU.

### Otros scripts

| Orden | Qué hace |
|---|---|
| `python scripts/estabilidad.py datos/creditcard.csv` | Mide la dispersión del AUPRC frente a la submuestra de entrenamiento |
| `python scripts/protocolos.py datos/creditcard.csv --cache cache` | Descompone la diferencia entre los dos protocolos de validación |
| `python scripts/figuras.py datos/creditcard.csv --cache cache` | Genera las figuras de los informes desde las predicciones en caché |
| `python scripts/exportar.py datos/creditcard.csv --modelo mlp` | Exporta modelo, escalador, umbral y curva de costo para la interfaz |
| `python scripts/exportar_navegador.py` | Convierte ese artefacto a JSON para el despliegue estático |
| `python scripts/capturas.py` | Toma las capturas de los informes con un navegador real |

### Levantar la interfaz

```bash
python scripts/exportar.py datos/creditcard.csv --modelo mlp --costo 5
uvicorn app.api:aplicacion --reload
```

En <http://127.0.0.1:8000>. La documentación interactiva de la API queda en
`/docs`.

---

## La interfaz

Está pensada para un analista de fraude, no para un científico de datos: la
pantalla muestra una **acción** —bloquear, revisar o permitir— y no una
probabilidad suelta. El control principal no es un botón de «predecir» sino el
**costo de una alerta falsa**, que es el único parámetro que la institución
conoce mejor que el modelo, y moverlo desplaza el umbral sobre la curva
calculada al exportar.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/estado` | Modelo, umbral vigente, curva de costo y columnas requeridas |
| `GET /api/umbral?costo=` | Umbral óptimo para un costo de alerta falsa dado |
| `POST /api/transaccion` | Veredicto de una transacción |
| `POST /api/lote` | Veredicto de un CSV, con resumen y cola ordenada por riesgo |

```bash
curl -s "http://127.0.0.1:8000/api/umbral?costo=50"
# {"costo":50.0,"umbral":0.4948455380805106}
```

El front end está escrito en HTML, CSS y JavaScript sin dependencias ni paso de
compilación. No contiene lógica de negocio: toda decisión sobre qué significa
una probabilidad vive en [`src/fraude/servicio.py`](src/fraude/servicio.py). Esa
frontera se puso a prueba sola —el modelo cambió tres veces durante el
proyecto— y la interfaz no cambió ninguna.

---

## Despliegue

<https://nicolas2322.github.io/deteccion-fraude-tarjeta/>

La página de [`docs/`](docs/) es una versión estática: el modelo se exporta a
JSON y se ejecuta en el navegador mediante
[`docs/modelo.js`](docs/modelo.js), treinta líneas de aritmética sin
dependencias. No hay servidor que mantener y ninguna transacción sale del
equipo del usuario.

Que las dos implementaciones del mismo cálculo coincidan no se da por supuesto:
[`tests/test_navegador.py`](tests/test_navegador.py) ejecuta ambas sobre las
mismas transacciones y exige que las probabilidades coincidan hasta la décima
cifra decimal.

> **Esta arquitectura es para que el trabajo sea comprobable, no una
> recomendación.** Un detector de fraude cuyos pesos son públicos es un detector
> que un defraudador puede estudiar para esquivarlo. Un despliegue real usaría
> la versión con servidor, en [`app/`](app/).

---

## Estructura

```
src/fraude/          Paquete: siete capas, dependencia en un solo sentido
  datos.py           Validación del esquema, orden temporal y particiones
  preprocesamiento.py  Transformación de variables y escalado sin fuga
  modelos.py         Fábrica común de las tres familias
  ft_transformer.py  FT-Transformer en PyTorch, implementado desde cero
  umbral.py          Búsqueda exacta del umbral por costo
  metricas.py        AUPRC, ROC-AUC, calibración y cuentas en el umbral
  pipeline.py        Orquestación, con caché de predicciones
  servicio.py        Decisión: de probabilidad a acción
app/                 API y front end (versión con servidor)
docs/                Despliegue estático (GitHub Pages)
scripts/             Interfaces de línea de órdenes
tests/               94 pruebas de propiedades
datos_muestra/       Muestra de 5.000 transacciones
informes/            Métricas en JSON, generadas por el pipeline
figuras/             Figuras de los informes
configs/             Semilla, particiones, costos y presupuesto de búsqueda
```

### El pipeline

1. **Ingesta** — lee el CSV e impone el orden temporal
2. **Validación** — 31 columnas, tipos, nulos, etiquetas e importes; falla pronto
3. **Partición temporal** — 80 % desarrollo, 20 % prueba, sin mezclar periodos
4. **Pliegues** — cinco, estratificados, conservando la prevalencia
5. **Preprocesamiento** — `log(1+importe)`, hora cíclica, escalado solo con el entrenamiento
6. **Entrenamiento** — un modelo por pliegue
7. **Predicción** — fuera de partición: ningún modelo puntúa lo que vio
8. **Umbral** — minimiza `C(t) = A(t) + c·n(t)` sobre las predicciones fuera de partición
9. **Prueba temporal** — el umbral fijado se aplica sin modificar al tramo reservado
10. **Informe** — métricas bajo ambos protocolos, con su diferencia

El orden importa: el umbral se fija **antes** de ver el tramo de prueba. Si se
eligiera sobre los datos con los que se reporta, el resultado sería optimista.

---

## Verificación

```bash
pytest --cov=src/fraude --cov=app --cov-fail-under=85
ruff check src tests scripts app
```

**94 pruebas, 96 % de cobertura, análisis estático sin advertencias.** La
integración continua las ejecuta en cada envío de cambios, en menos de un
minuto, porque las pruebas trabajan sobre conjuntos sintéticos: no hacen falta
los datos, ni credenciales de Kaggle, ni PyTorch.

Las pruebas comprueban **propiedades y no cifras**. Las más exigentes:

- La búsqueda vectorizada del umbral coincide con una evaluación por fuerza
  bruta sobre todos los candidatos, para seis valores del costo
- El escalado de la evaluación usa los parámetros del entrenamiento, verificado
  desplazando artificialmente la media de la validación
- Las 23:00 quedan más cerca de las 00:00 que de las 12:00 tras la codificación
  cíclica
- La inferencia en JavaScript coincide con la de Python hasta la décima cifra

Esa orientación viene de una lección: en el proyecto anterior, un defecto del
preprocesamiento atravesó 27 pruebas y una cobertura del 100 %, porque ninguna
comprobaba la propiedad que fallaba. La cobertura mide qué líneas se ejecutan,
no qué propiedades se verifican.

### Reproducibilidad

Una corrida queda descrita por [`configs/experimento.yaml`](configs/experimento.yaml)
más el identificador del commit. La semilla está fijada en 42, las dependencias
ancladas a versión exacta, y ningún parámetro vive escondido en el código.

---

## Limitaciones

- **Ningún modelo ha recibido búsqueda de hiperparámetros.** Las comparaciones
  valen como pruebas iniciales, no como resultado final.
- **La estabilidad del FT-Transformer está sin medir.** Si resultara tan
  inestable como el ensamble de árboles, su 0,7949 quedaría invalidado igual
  que quedó el de LightGBM.
- **El presupuesto de épocas del FT-Transformer lo limita el cómputo**: diez por
  pliegue, cuando en pruebas sintéticas seguía mejorando más allá de la
  treintena. Probablemente su cifra lo subestima.
- **El periodo es de dos días.** No permite observar deriva de concepto, que
  Dal Pozzolo et al. (2018) identifican como un desafío central del problema
  real.
- **Las variables no son interpretables.** `V1`–`V28` son componentes
  principales sujetas a confidencialidad, de modo que el sistema no puede
  explicar una decisión en términos de negocio.
- **La interfaz no comunica incertidumbre.** Muestra una probabilidad con
  cuatro decimales y nada que indique cuánta confianza merece el cuarto.

---

## Referencias

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., y Bontempi, G. (2015). *Calibrating probability with undersampling for unbalanced classification*. IEEE SSCI.
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., y Bontempi, G. (2018). *Credit card fraud detection: A realistic modeling and a novel learning strategy*. IEEE TNNLS, 29(8), 3784–3797.
- Elkan, C. (2001). *The foundations of cost-sensitive learning*. IJCAI.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., y Babenko, A. (2021). *Revisiting deep learning models for tabular data*. NeurIPS.
- Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS, Datasets and Benchmarks.
- Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.
- Saito, T., y Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets*. PLOS ONE, 10(3), e0118432.

## Licencia y atribución

Código: proyecto académico de la Universidad de Cundinamarca.
Datos: Machine Learning Group, Université Libre de Bruxelles, en colaboración
con Worldline. Licencia abierta con atribución.
