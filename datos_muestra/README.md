# Muestra del conjunto

`creditcard_muestra.csv` son 5.000 de las 284.807 transacciones del conjunto
original, con el mismo esquema de 31 columnas.

**Esta muestra no es representativa, y a propósito.** Contiene los 492 fraudes
del conjunto completo y 4.508 transacciones legítimas elegidas al azar, de modo
que su prevalencia es del 9,84 % frente al 0,173 % real. Está pensada para dos
cosas y solo para esas dos:

1. **Inspeccionar el esquema** sin descargar 150 MB.
2. **Ejecutar el pipeline de extremo a extremo** y comprobar que funciona.

**No sirve para reproducir ninguna métrica de los informes.** Todas las cifras
de `informes/` y del documento de avance proceden del archivo completo. Con una
prevalencia cincuenta y siete veces mayor, la línea base del AUPRC, el umbral
por costo y la dificultad del problema son otros.

La muestra se construyó ordenando el archivo original por `Time`, tomando todas
las filas con `Class == 1` y una selección aleatoria de las demás con la semilla
42.

## Para obtener el archivo completo

Descargar de
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> y dejar
`creditcard.csv` en `datos/`, que está excluido del control de versiones. Con
la interfaz de línea de órdenes de Kaggle configurada:

```
kaggle datasets download -d mlg-ulb/creditcardfraud -p datos --unzip
python scripts/explorar.py datos/creditcard.csv
```

La segunda orden valida el esquema y debe informar de 284.807 transacciones y
492 fraudes. Si no lo hace, el archivo no es el esperado.

## Atribución

Machine Learning Group, Université Libre de Bruxelles, en colaboración con
Worldline. Licencia abierta con atribución. Citar como indica el `README.md`
principal del repositorio.
