/* Inferencia del modelo en el navegador.

   Reimplementa en JavaScript exactamente lo que el pipeline hace en Python:
   la transformación de variables, el escalado con los parámetros del
   entrenamiento y el paso hacia adelante del perceptrón multicapa. No hay
   dependencias; la aritmética de una red de tres capas son tres productos de
   matrices.

   Que esta implementación coincida con la de Python no se da por supuesto: hay
   una prueba automatizada, `tests/test_navegador.py`, que ejecuta las dos
   sobre las mismas transacciones y exige que las probabilidades coincidan
   hasta la décima cifra decimal. Si alguien cambia el preprocesamiento en
   Python y olvida cambiarlo aquí, esa prueba falla.
*/

"use strict";

const SEGUNDOS_POR_DIA = 86400;

export class ModeloLocal {
  constructor(paquete) {
    this.p = paquete;
    this.umbral = paquete.umbral;
    this.curva = paquete.curva;
  }

  static async cargar(url = "modelo.json") {
    const respuesta = await fetch(url);
    if (!respuesta.ok) throw new Error(`No se pudo cargar el modelo (${respuesta.status})`);
    return new ModeloLocal(await respuesta.json());
  }

  /* Mismas transformaciones que `fraude.preprocesamiento.transformar`:
     el importe pasa por log(1 + x) y el tiempo se convierte en hora del día
     codificada con seno y coseno. El orden de las variables lo fija el propio
     paquete, de modo que no puede desincronizarse del entrenamiento. */
  vector(t) {
    const hora = ((t.Time % SEGUNDOS_POR_DIA) + SEGUNDOS_POR_DIA) % SEGUNDOS_POR_DIA / 3600;
    const angulo = 2 * Math.PI * hora / 24;
    const derivadas = {
      importe_log: Math.log1p(t.Amount),
      hora_sen: Math.sin(angulo),
      hora_cos: Math.cos(angulo)
    };
    return this.p.variables.map((v) => (v in derivadas ? derivadas[v] : t[v]));
  }

  escalar(x) {
    const { media, desviacion } = this.p.escalador;
    return x.map((v, i) => (v - media[i]) / desviacion[i]);
  }

  /* Paso hacia adelante: ReLU en las capas ocultas, sigmoide en la salida. */
  probabilidad(transaccion) {
    let h = this.escalar(this.vector(transaccion));
    this.p.capas.forEach((capa, indice) => {
      const ultima = indice === this.p.capas.length - 1;
      const salida = new Array(capa.sesgos.length).fill(0);
      for (let j = 0; j < salida.length; j++) {
        let suma = capa.sesgos[j];
        for (let i = 0; i < h.length; i++) suma += h[i] * capa.pesos[i][j];
        salida[j] = ultima ? suma : Math.max(0, suma);
      }
      h = salida;
    });
    return 1 / (1 + Math.exp(-h[0]));
  }

  umbralPara(costo) {
    const c = this.curva;
    if (!c || !c.length) return this.umbral;
    if (costo <= c[0].costo) return c[0].umbral;
    if (costo >= c[c.length - 1].costo) return c[c.length - 1].umbral;
    for (let i = 1; i < c.length; i++) {
      if (costo <= c[i].costo) {
        const t = (costo - c[i - 1].costo) / (c[i].costo - c[i - 1].costo);
        return c[i - 1].umbral + t * (c[i].umbral - c[i - 1].umbral);
      }
    }
    return this.umbral;
  }

  /* Mismas reglas que `fraude.servicio`: tres acciones y una franja de
     revisión justo por debajo del umbral. */
  juzgar(transaccion, umbral) {
    const u = umbral ?? this.umbral;
    const p = this.probabilidad(transaccion);
    const accion = p >= u ? "bloquear" : (p >= u * 0.75 ? "revisar" : "permitir");
    return {
      probabilidad: p,
      umbral: u,
      accion,
      importe: transaccion.Amount,
      riesgo_esperado: p * transaccion.Amount,
      margen: p - u
    };
  }

  estado() {
    return {
      modelo: this.p.modelo,
      umbral: this.p.umbral,
      costo_alerta_falsa: this.p.costo_alerta_falsa,
      columnas_requeridas: ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"],
      curva: this.curva
    };
  }
}

/* Lectura de CSV suficiente para este esquema: solo números, sin comillas ni
   comas dentro de los campos. Se valida que estén las columnas esperadas y se
   nombra la primera que falte, igual que hace el validador de Python. */
export function leerCSV(texto, requeridas) {
  const lineas = texto.trim().split(/\r?\n/);
  if (lineas.length < 2) throw new Error("El archivo no tiene filas de datos.");
  const cabecera = lineas[0].split(",").map((c) => c.trim());
  const faltan = requeridas.filter((c) => !cabecera.includes(c));
  if (faltan.length) {
    throw new Error(`Faltan ${faltan.length} columnas: ${faltan.slice(0, 6).join(", ")}` +
      (faltan.length > 6 ? " …" : ""));
  }
  const indice = Object.fromEntries(cabecera.map((c, i) => [c, i]));
  return lineas.slice(1).filter((l) => l.trim()).map((linea, n) => {
    const celdas = linea.split(",");
    const fila = {};
    for (const columna of requeridas) {
      const valor = Number(celdas[indice[columna]]);
      if (Number.isNaN(valor)) {
        throw new Error(`Fila ${n + 2}: el valor de ${columna} no es numérico.`);
      }
      fila[columna] = valor;
    }
    return fila;
  });
}
