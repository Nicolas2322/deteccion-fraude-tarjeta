# Detección de transacciones fraudulentas

Redes neuronales tabulares frente a *Gradient Boosting* sobre el conjunto
*Credit Card Fraud Detection*, con elección del umbral por **costo económico
esperado** en lugar de por una métrica abstracta.

Tercer proyecto de la asignatura **Aplicaciones en Deep Learning**
· Especialización en Inteligencia Artificial · Universidad de Cundinamarca
· Yohan Nicolás Gómez Castañeda

---

## El problema en una línea

492 fraudes entre 284.807 transacciones, el **0,173 %**. Con ese desbalance, un
clasificador que responda siempre «no hay fraude» alcanza una exactitud del
**99,83 %** sin detectar un solo caso. La exactitud queda descartada de entrada
y la métrica principal es el **AUPRC**, cuyo valor de referencia sin información
es la propia prevalencia, 0,0017.

Los dos errores tampoco cuestan lo mismo: dejar pasar un fraude cuesta el
importe de la transacción, que está en los datos; bloquear una compra legítima
cuesta fricción con el cliente, que es un parámetro de la institución. Por eso
el umbral se elige minimizando el costo total y no fijándolo en 0,5.

## Estructura

```
src/fraude/
├── datos.py             Recolección, validación del esquema y particionamiento
├── preprocesamiento.py  Transformación de variables y escalado sin fuga
├── modelos.py           Construcción de los modelos comparados
├── umbral.py            Elección del punto de corte por costo esperado
├── metricas.py          Evaluación adecuada a un desbalance extremo
└── pipeline.py          Orquestación de extremo a extremo

scripts/
├── explorar.py          Valida el archivo y describe el conjunto
└── evaluar.py           Ejecuta el pipeline y escribe las métricas

tests/                   59 pruebas, 96 % de cobertura
configs/                 Hiperparámetros y presupuesto de búsqueda
```

Cada módulo tiene una sola responsabilidad y ninguno lee el archivo original
salvo `datos.py`, de modo que un cambio en la fuente se corrige en un solo sitio.

## El pipeline

```
 1. Datos            creditcard.csv → validación de esquema → orden temporal
 2. Partición        último 20 % del periodo reservado como prueba
 3. Preprocesamiento log(1+importe), hora cíclica, escalado ajustado solo con
                     el entrenamiento de cada pliegue
 4. Entrenamiento    validación cruzada estratificada de 5 pliegues
 5. Predicción       una probabilidad por transacción, de un modelo que no la vio
 6. Umbral           minimiza  C(t) = A(t) + c·n(t)   sobre esas predicciones
 7. Prueba           el umbral se aplica sin modificar al tramo temporal reservado
```

El umbral se fija en desarrollo y se reutiliza en prueba. Elegirlo sobre la
prueba produciría una estimación optimista de algo que en producción hay que
decidir de antemano.

## Decisiones de diseño

**Dos protocolos de validación en paralelo.** La validación cruzada selecciona
modelos; la partición temporal estima el desempeño. La diferencia entre ambas
mide el optimismo que introduce una partición aleatoria sobre datos que en
realidad son un flujo (Dal Pozzolo et al., 2018).

**El umbral como decisión económica.** `umbral.umbral_optimo` implementa la
búsqueda exacta: la función de costo solo cambia de valor cuando una transacción
cruza el umbral, de modo que basta evaluar los cortes entre transacciones
ordenadas por probabilidad. Una ordenación y una pasada de sumas acumuladas,
O(n log n). `umbral.umbral_por_importe` extiende la regla a un umbral que decrece
con el importe: bloquear conviene cuando `p > c / (c + a)`.

**Escalado sin fuga.** `Escalador` es una clase con ajuste explícito, no una
función, para que resulte imposible escalar la validación con sus propios
parámetros.

**Importaciones diferidas.** LightGBM y PyTorch se importan dentro de la función
que los necesita. El núcleo del pipeline se instala, prueba y ejecuta solo con
numpy, pandas y scikit-learn, y la integración continua tarda menos de un minuto.

## Uso

```bash
pip install -r requirements.txt
pip install -e .

# Los datos no se versionan. Descargar creditcard.csv de Kaggle en datos/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

python scripts/explorar.py datos/creditcard.csv
python scripts/evaluar.py datos/creditcard.csv --modelo lightgbm --costo 5
```

Salida de `evaluar.py`: AUPRC bajo ambos protocolos y su diferencia, umbral
elegido, exhaustividad, alertas falsas y costo total en el tramo de prueba, más
un JSON con la curva de sensibilidad frente al costo de la alerta falsa.

## Verificación

```bash
ruff check src tests scripts
pytest --cov=src/fraude --cov-fail-under=85
```

59 pruebas, 96 % de cobertura, sin advertencias de estilo. Se ejecutan en cada
*push* mediante GitHub Actions.

Las pruebas trabajan sobre conjuntos sintéticos con el mismo esquema que el
archivo real, de modo que la verificación no depende de descargar 150 MB ni de
credenciales de Kaggle. Comprueban propiedades y no cifras: que el esquema
inválido falle pronto, que la partición temporal no mezcle periodos, que el
escalado de la evaluación use los parámetros del entrenamiento, que la hora 23:00
quede cerca de la 00:00 en la codificación cíclica, y que la búsqueda vectorizada
del umbral coincida con una evaluación por fuerza bruta en todos los candidatos.

## Datos

| | |
|---|---|
| Fuente | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Machine Learning Group, Université Libre de Bruxelles |
| Transacciones | 284.807 |
| Fraudes | 492 (0,173 %) |
| Periodo | Dos días de septiembre de 2013, tarjetahabientes europeos |
| Variables | V1–V28 (componentes principales), Time, Amount, Class |

Las V1 a V28 son componentes principales porque los atributos originales están
sujetos a confidencialidad. Eso tiene dos consecuencias: no hay interpretabilidad
de negocio, y el conjunto está ya rotado, lo que según Grinsztajn et al. (2022)
debería reducir la ventaja que los árboles obtienen de la alineación con los ejes.
El proyecto pone esa predicción a prueba.

## Resultados medidos

Ejecución sobre el archivo real, costo de alerta falsa igual a cinco. Los
informes completos están en `informes/` y las figuras en `figuras/`, generados
por el propio pipeline.

| | Logística | LightGBM | Perceptrón | FT-Transformer |
|---|---|---|---|---|
| AUPRC, validación cruzada | 0,7289 | 0,7943 | 0,8335 | 0,6998 |
| AUPRC, prueba temporal | 0,7576 | 0,6663 | **0,8175** | 0,7949 |
| ROC-AUC, prueba temporal | 0,9856 | 0,8916 | 0,9678 | 0,9703 |
| Error de calibración | 0,1109 | 0,0009 | 0,0008 | **0,0004** |
| Fraudes detectados (de 75) | 55 | 50 | **59** | 53 |
| Alertas falsas | 19 | 62 | 20 | **11** |
| Costo total | 2.982 | 2.735 | **2.499** | 3.978 |

La línea base del AUPRC en el tramo de prueba es 0,0013.

Las dos arquitecturas neuronales superan a la familia de árboles. El orden es
contrario al habitual en datos tabulares, y era previsible: Grinsztajn et al.
(2022) predicen que la ventaja de los árboles se reduce sobre datos rotados, y
V1–V28 son componentes principales, es decir, exactamente una rotación.

**Ninguna de estas cifras procede de una búsqueda de hiperparámetros.** Valen
como pruebas iniciales, no como resultado final. La razón para posponer esa
búsqueda está justo debajo.

## Una cifra única de AUPRC no es reproducible aquí

`scripts/estabilidad.py` entrena seis veces sobre submuestras aleatorias del
80 % del tramo de desarrollo y evalúa siempre sobre el mismo tramo de prueba.
Lo único que cambia entre repeticiones es qué filas vio el modelo; los ajustes
son deterministas, y hay una prueba que lo comprueba.

| Configuración | Media | Desviación | Rango |
|---|---|---|---|
| LightGBM, 500 árboles, 31 hojas | 0,3904 | 0,2813 | [0,1381 – 0,8209] |
| LightGBM, 200 árboles, 15 hojas, mín. 50 por hoja | 0,7014 | 0,0427 | [0,6472 – 0,7497] |
| Regresión logística | 0,7543 | 0,0102 | [0,7414 – 0,7699] |
| Perceptrón multicapa | 0,8179 | 0,0052 | [0,8103 – 0,8236] |

El ensamble profundo puede dar el peor o el mejor AUPRC de todo el estudio sin
que nada cambie salvo la submuestra. Con 417 fraudes en el tramo de desarrollo,
quitar al azar un 20 % de las filas se lleva unos 83 de ellos, y un ensamble de
500 árboles con 31 hojas tiene capacidad de sobra para memorizar casos concretos
en lugar de aprender regularidades.

Consecuencias, ya aplicadas:

1. **LightGBM va regularizado por defecto** (200 árboles, 15 hojas, mínimo 50
   observaciones por hoja). La media sube de 0,3904 a 0,7014 y la desviación
   baja de 0,2813 a 0,0427. Restringir más vuelve a empeorar ambas.
2. **El informe final reportará media, desviación y rango**, no cifras únicas.
3. **La búsqueda de hiperparámetros espera** a que la medición sea estable:
   buscar sobre una puntuación con desviación de 0,28 seleccionaría la
   configuración más afortunada, no la mejor.

El ROC-AUC de la configuración inestable sobre el tramo de prueba era 0,8916 y
la exactitud rondaba el 99,8 %. Solo el AUPRC delató el problema.

## Por qué LightGBM no repondera la clase minoritaria

Activar `is_unbalance` saturaba las probabilidades: 3.013 transacciones del
tramo de prueba empatadas en exactamente 1,0, AUPRC de 0,0147 frente a 0,7719
sin reponderación, y exhaustividad nula porque el umbral por costo no encontraba
ningún corte dentro del empate. Es el efecto que describen Dal Pozzolo et al.
(2015): reponderar desplaza la probabilidad, y aquí la colapsó.

## Estado

Pipeline completo y ejecutado sobre el archivo real con cuatro modelos de tres
familias. FT-Transformer implementado desde cero en PyTorch (tokenizador por
variable, token `[CLS]`, bloques pre-norma con ReGLU). Pendientes la búsqueda de
hiperparámetros con presupuesto común sobre el protocolo de repeticiones, la
medida de estabilidad del FT-Transformer y el análisis comparativo final.

## Referencias

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., y Bontempi, G. (2015). *Calibrating probability with undersampling for unbalanced classification*. IEEE SSCI.
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., y Bontempi, G. (2018). *Credit card fraud detection: A realistic modeling and a novel learning strategy*. IEEE TNNLS, 29(8), 3784–3797.
- Elkan, C. (2001). *The foundations of cost-sensitive learning*. IJCAI.
- Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS, Datasets and Benchmarks.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., y Babenko, A. (2021). *Revisiting deep learning models for tabular data*. NeurIPS.
