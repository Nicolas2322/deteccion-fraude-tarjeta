/* Detector de transacciones fraudulentas — versión estática.

   Es la misma interfaz que la versión con servidor, con una sola diferencia:
   en lugar de llamar a la API, llama a `ModeloLocal`, que ejecuta el modelo en
   el propio navegador. Todo lo demás —estructura, estilos, validación,
   mensajes— es idéntico, y esa es precisamente la ventaja de haber puesto la
   lógica de decisión fuera de la capa de presentación.
*/

"use strict";

import { ModeloLocal, leerCSV } from "./modelo.js";

const COMPONENTES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const MAXIMO_FILAS = 20000;
const ACCIONES = {
  bloquear: {
    titulo: "Bloquear la transacción",
    sub: "La probabilidad estimada supera el umbral vigente.",
    explica: "El sistema recomienda detener la operación y notificar al titular. " +
             "La decisión final corresponde al analista."
  },
  revisar: {
    titulo: "Enviar a revisión",
    sub: "Queda por debajo del umbral, pero lo bastante cerca como para no ignorarla.",
    explica: "La franja de revisión existe porque un caso que roza el umbral no es " +
             "equivalente a uno que queda muy por debajo, y tratarlos igual desperdicia " +
             "información que el modelo sí tiene."
  },
  permitir: {
    titulo: "Permitir la transacción",
    sub: "La probabilidad estimada queda claramente por debajo del umbral.",
    explica: "No se requiere intervención. El riesgo esperado que se deja pasar queda " +
             "registrado en el resumen del lote."
  }
};

/* Dos transacciones reales del conjunto, incluidas para que cualquiera pueda
   probar la interfaz sin tener que inventarse 28 componentes principales. */
const EJEMPLOS = {
  legitima: {
    Time: 406, Amount: 2.69,
    V1: -1.3598071336738, V2: -0.0727811733098497, V3: 2.53634673796914,
    V4: 1.37815522427443, V5: -0.338320769942518, V6: 0.462387777762292,
    V7: 0.239598554061257, V8: 0.0986979012610507, V9: 0.363786969611213,
    V10: 0.0907941719789316, V11: -0.551599533260813, V12: -0.617800855762348,
    V13: -0.991389847235408, V14: -0.311169353699879, V15: 1.46817697209427,
    V16: -0.470400525259478, V17: 0.207971241929242, V18: 0.0257905801985591,
    V19: 0.403992960255733, V20: 0.251412098239705, V21: -0.018306777944153,
    V22: 0.277837575558899, V23: -0.110473910188767, V24: 0.0669280749146731,
    V25: 0.128539358273528, V26: -0.189114843888824, V27: 0.133558376740387,
    V28: -0.0210530534538215
  },
  fraude: {
    Time: 406, Amount: 0,
    V1: -2.3122265423263, V2: 1.95199201064158, V3: -1.60985073229769,
    V4: 3.9979055875468, V5: -0.522187864667764, V6: -1.42654531920595,
    V7: -2.53738730624579, V8: 1.39165724829804, V9: -2.77008927719433,
    V10: -2.77227214465915, V11: 3.20203320709635, V12: -2.89990738849473,
    V13: -0.595221881324605, V14: -4.28925378244217, V15: 0.389724120274487,
    V16: -1.14074717980657, V17: -2.83005567450437, V18: -0.0168224681808257,
    V19: 0.416955705037907, V20: 0.126910559061474, V21: 0.517232370861764,
    V22: -0.0350493686052974, V23: -0.465211076182388, V24: 0.320198198514526,
    V25: 0.0445191674731724, V26: 0.177839798284401, V27: 0.261145002567677,
    V28: -0.143275874698919
  }
};

const $ = (id) => document.getElementById(id);
const numero = (v, d = 2) => v.toLocaleString("es-ES",
  { minimumFractionDigits: d, maximumFractionDigits: d });

let estadoServicio = null;
let modelo = null;

/* ───────────── Arranque ───────────── */

document.addEventListener("DOMContentLoaded", async () => {
  construirRejilla();
  conectarPestanas();
  conectarFormulario();
  conectarLote();
  conectarCosto();
  await cargarEstado();
});

function construirRejilla() {
  $("rejilla-v").innerHTML = COMPONENTES.map((c) => `
    <div>
      <label for="${c}">${c}</label>
      <input type="number" id="${c}" name="${c}" step="any" inputmode="decimal" required>
    </div>`).join("");
}

async function cargarEstado() {
  try {
    modelo = await ModeloLocal.cargar("modelo.json");
    estadoServicio = modelo.estado();
    $("estado-modelo").textContent = estadoServicio.modelo;
    $("estado-umbral").textContent = numero(estadoServicio.umbral, 4);
    $("estado-columnas").textContent = estadoServicio.columnas_requeridas.length;
    $("costo").value = estadoServicio.costo_alerta_falsa;
    dibujarCurva(estadoServicio.curva);
    actualizarCosto();
  } catch (e) {
    mostrarError($("error-una"), `No se pudo cargar el modelo: ${e.message}`);
  }
}

/* ───────────── Control de costo ───────────── */

function conectarCosto() {
  $("costo").addEventListener("input", actualizarCosto);
}

function umbralLocal(costo) {
  // Se interpola sobre la curva que el servidor ya envió, para que el control
  // responda sin una petición por cada movimiento del deslizador.
  const c = estadoServicio?.curva;
  if (!c || !c.length) return estadoServicio?.umbral ?? 0.5;
  if (costo <= c[0].costo) return c[0].umbral;
  if (costo >= c[c.length - 1].costo) return c[c.length - 1].umbral;
  for (let i = 1; i < c.length; i++) {
    if (costo <= c[i].costo) {
      const t = (costo - c[i - 1].costo) / (c[i].costo - c[i - 1].costo);
      return c[i - 1].umbral + t * (c[i].umbral - c[i - 1].umbral);
    }
  }
  return estadoServicio.umbral;
}

function actualizarCosto() {
  const costo = parseFloat($("costo").value);
  const u = umbralLocal(costo);
  $("costo-valor").textContent = `${numero(costo, 1)} unidades por alerta falsa`;
  $("umbral-vivo").textContent = numero(u, 4);
  $("umbral-nota").textContent = costo < 3
    ? "Costo bajo: el sistema alerta con poca evidencia y producirá más alertas falsas."
    : costo > 60
      ? "Costo alto: el sistema exige mucha evidencia y dejará pasar más fraude."
      : "Se bloquea cuando la probabilidad de fraude supera este valor.";
  marcarCurva(costo);
}

/* ───────────── Curva ───────────── */

function dibujarCurva(curva) {
  if (!curva || !curva.length) return;
  const svg = $("curva");
  const W = 320, H = 120, M = { t: 10, r: 10, b: 22, l: 34 };
  const lx = (c) => M.l + (Math.log10(c) - Math.log10(curva[0].costo)) /
    (Math.log10(curva[curva.length - 1].costo) - Math.log10(curva[0].costo)) *
    (W - M.l - M.r);
  const ly = (u) => H - M.b - u * (H - M.t - M.b);
  const d = curva.map((p, i) => `${i ? "L" : "M"}${lx(p.costo).toFixed(1)},${ly(p.umbral).toFixed(1)}`).join("");

  svg.innerHTML = `
    <line x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"
          stroke="#d9dbe0"/>
    <line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}" stroke="#d9dbe0"/>
    ${[0, 0.5, 1].map((u) => `<text x="${M.l - 5}" y="${ly(u) + 3.5}"
        font-size="8" fill="#5a5f68" text-anchor="end">${u.toFixed(1).replace(".", ",")}</text>`).join("")}
    ${[1, 10, 100].map((c) => `<text x="${lx(c)}" y="${H - M.b + 12}"
        font-size="8" fill="#5a5f68" text-anchor="middle">${c}</text>`).join("")}
    <path d="${d}" fill="none" stroke="#1b1d21" stroke-width="1.6"/>
    <circle id="marca-curva" r="3.5" fill="#8c1d18"/>`;
  svg.dataset.min = curva[0].costo;
  svg.dataset.max = curva[curva.length - 1].costo;
}

function marcarCurva(costo) {
  const svg = $("curva"), marca = $("marca-curva");
  if (!marca) return;
  const W = 320, H = 120, M = { t: 10, r: 10, b: 22, l: 34 };
  const min = parseFloat(svg.dataset.min), max = parseFloat(svg.dataset.max);
  const c = Math.min(Math.max(costo, min), max);
  marca.setAttribute("cx", M.l + (Math.log10(c) - Math.log10(min)) /
    (Math.log10(max) - Math.log10(min)) * (W - M.l - M.r));
  marca.setAttribute("cy", H - M.b - umbralLocal(c) * (H - M.t - M.b));
}

/* ───────────── Pestañas ───────────── */

function conectarPestanas() {
  const tabs = [["tab-una", "panel-una"], ["tab-lote", "panel-lote"]];
  tabs.forEach(([t]) => $(t).addEventListener("click", () => {
    tabs.forEach(([ti, pi]) => {
      const activo = ti === t;
      $(ti).setAttribute("aria-selected", String(activo));
      $(pi).hidden = !activo;
    });
  }));
}

/* ───────────── Una transacción ───────────── */

function conectarFormulario() {
  $("btn-ejemplo-legitima").addEventListener("click", () => rellenar(EJEMPLOS.legitima));
  $("btn-ejemplo-fraude").addEventListener("click", () => rellenar(EJEMPLOS.fraude));
  $("btn-limpiar").addEventListener("click", () => {
    $("formulario").reset();
    $("veredicto").hidden = true;
    $("error-una").hidden = true;
  });
  $("formulario").addEventListener("submit", enviarTransaccion);
}

function rellenar(ejemplo) {
  Object.entries(ejemplo).forEach(([k, v]) => { if ($(k)) $(k).value = v; });
  $("error-una").hidden = true;
}

async function enviarTransaccion(evento) {
  evento.preventDefault();
  $("error-una").hidden = true;

  const valores = {};
  const faltan = [];
  ["Time", "Amount", ...COMPONENTES].forEach((c) => {
    const bruto = $(c).value.trim();
    if (bruto === "" || Number.isNaN(Number(bruto))) faltan.push(c);
    else valores[c] = Number(bruto);
  });
  if (faltan.length) {
    mostrarError($("error-una"),
      `Faltan ${faltan.length} valores o no son numéricos: ${faltan.slice(0, 8).join(", ")}` +
      (faltan.length > 8 ? " …" : "") + ". Puedes cargar un caso de ejemplo para probar.");
    $(faltan[0]).focus();
    return;
  }

  try {
    pintarVeredicto(modelo.juzgar(valores, umbralLocal(parseFloat($("costo").value))));
  } catch (e) {
    mostrarError($("error-una"), e.message);
  }
}

function pintarVeredicto(v) {
  const info = ACCIONES[v.accion];
  $("sello").textContent = v.accion;
  $("sello").className = `sello ${v.accion}`;
  $("veredicto-titulo").textContent = info.titulo;
  $("veredicto-sub").textContent = info.sub;
  $("v-prob").textContent = numero(v.probabilidad * 100, 2) + " %";
  $("v-umbral").textContent = numero(v.umbral * 100, 2) + " %";
  $("v-margen").textContent = (v.margen >= 0 ? "+" : "−") +
    numero(Math.abs(v.margen) * 100, 2) + " pp";
  $("v-riesgo").textContent = numero(v.riesgo_esperado, 2);
  $("barra-valor").style.width = `${Math.min(v.probabilidad, 1) * 100}%`;
  $("barra-umbral").style.left = `${Math.min(v.umbral, 1) * 100}%`;
  $("v-explicacion").textContent = info.explica;
  $("veredicto").hidden = false;
}

/* ───────────── Lote ───────────── */

function conectarLote() {
  const zona = $("soltar"), entrada = $("archivo");
  zona.addEventListener("click", () => entrada.click());
  zona.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); entrada.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => zona.addEventListener(t, (e) => {
    e.preventDefault(); zona.classList.add("encima");
  }));
  ["dragleave", "drop"].forEach((t) => zona.addEventListener(t, (e) => {
    e.preventDefault(); zona.classList.remove("encima");
  }));
  zona.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) enviarLote(e.dataTransfer.files[0]);
  });
  entrada.addEventListener("change", () => {
    if (entrada.files.length) enviarLote(entrada.files[0]);
  });
}

async function enviarLote(archivo) {
  $("error-lote").hidden = true;
  $("soltar").querySelector(".soltar-titulo").textContent =
    `Procesando ${archivo.name}…`;

  const umbral = umbralLocal(parseFloat($("costo").value));

  try {
    if (!archivo.name.toLowerCase().endsWith(".csv")) {
      throw new Error("El archivo debe ser un CSV con las columnas del esquema.");
    }
    const filas = leerCSV(await archivo.text(), estadoServicio.columnas_requeridas);
    if (filas.length > MAXIMO_FILAS) {
      throw new Error(`El archivo tiene ${filas.length.toLocaleString("es-ES")} filas y el ` +
        `máximo por envío es ${MAXIMO_FILAS.toLocaleString("es-ES")}. Divídelo en partes.`);
    }
    const datos = resolverLote(filas, umbral);
    pintarLote(datos);
    $("soltar").querySelector(".soltar-titulo").textContent =
      `${archivo.name} — ${datos.resumen.transacciones.toLocaleString("es-ES")} filas. ` +
      "Pulsa para cargar otro.";
  } catch (e) {
    mostrarError($("error-lote"), e.message);
    $("soltar").querySelector(".soltar-titulo").textContent =
      "Arrastra el CSV aquí o pulsa para elegirlo";
  }
}

function pintarLote(datos) {
  const r = datos.resumen;
  $("r-bloquear").textContent = r.bloquear.toLocaleString("es-ES");
  $("r-revisar").textContent = r.revisar.toLocaleString("es-ES");
  $("r-permitir").textContent = r.permitir.toLocaleString("es-ES");
  $("r-importe").textContent = numero(r.importe_bloqueado, 2);
  $("r-riesgo").textContent = numero(r.riesgo_que_pasa, 2);

  $("tabla-lote").tBodies[0].innerHTML = datos.filas.map((f) => `
    <tr>
      <td class="num">${f.fila}</td>
      <td><span class="marca-accion ${f.accion}">${f.accion}</span></td>
      <td class="num">${numero(f.probabilidad * 100, 2)} %</td>
      <td class="num">${numero(f.importe, 2)}</td>
      <td class="num">${numero(f.riesgo_esperado, 2)}</td>
    </tr>`).join("");

  $("resultado-lote").hidden = false;
}

/* Agregación del lote. En la versión con servidor esto lo hace
   `fraude.servicio.resumen_lote`; aquí se reproduce para que la página
   estática no necesite ningún servicio. La prueba de paridad comprueba que
   ambas implementaciones coincidan. */
function resolverLote(filas, umbral) {
  const veredictos = filas.map((f) => modelo.juzgar(f, umbral));
  const cuenta = (a) => veredictos.filter((v) => v.accion === a).length;
  const orden = veredictos
    .map((v, i) => [i, v])
    .sort((a, b) => b[1].riesgo_esperado - a[1].riesgo_esperado)
    .slice(0, 200);
  return {
    umbral,
    resumen: {
      transacciones: veredictos.length,
      bloquear: cuenta("bloquear"),
      revisar: cuenta("revisar"),
      permitir: cuenta("permitir"),
      importe_bloqueado: veredictos.filter((v) => v.accion === "bloquear")
        .reduce((s, v) => s + v.importe, 0),
      riesgo_que_pasa: veredictos.filter((v) => v.accion === "permitir")
        .reduce((s, v) => s + v.riesgo_esperado, 0)
    },
    filas: orden.map(([i, v]) => ({ fila: i + 2, ...v }))
  };
}

/* ───────────── Utilidades ───────────── */

function mostrarError(nodo, mensaje) {
  nodo.textContent = mensaje;
  nodo.hidden = false;
}
