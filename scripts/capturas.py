"""Toma capturas de la interfaz con un navegador real.

Uso:
    uvicorn app.api:aplicacion --port 8008 &
    python scripts/capturas.py --url http://127.0.0.1:8008 --salida capturas

Las capturas del informe se generan así y no a mano, de modo que cuando la
interfaz cambie basta con volver a ejecutar este script para que las imágenes
del documento dejen de estar desactualizadas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ANCHO, ALTO = 980, 1000


def main() -> None:
    ap = argparse.ArgumentParser(description="Capturas de la interfaz")
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--salida", type=Path, default=Path("capturas"))
    ap.add_argument("--csv", type=Path, default=Path("/tmp/lote.csv"),
                    help="CSV de ejemplo para la captura del modo lote")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        navegador = p.chromium.launch(channel="chrome", args=["--no-sandbox"])
        pagina = navegador.new_page(viewport={"width": ANCHO, "height": ALTO},
                                    device_scale_factor=2)
        pagina.goto(args.url, wait_until="networkidle")

        # 1. Pantalla inicial completa
        pagina.screenshot(path=args.salida / "01_inicio.png", full_page=True)

        # 2. Control de costo en un valor alto: el umbral sube
        pagina.fill("#costo", "80")
        pagina.dispatch_event("#costo", "input")
        pagina.wait_for_timeout(250)
        pagina.locator("section.costo").screenshot(path=args.salida / "02_costo_alto.png")
        pagina.fill("#costo", "5")
        pagina.dispatch_event("#costo", "input")

        # 3. Validación: enviar el formulario vacío
        pagina.click("#btn-limpiar")
        pagina.click("button.principal")
        pagina.wait_for_selector("#error-una:not([hidden])")
        pagina.locator("#panel-una").screenshot(path=args.salida / "03_validacion.png")

        # 4. Veredicto de bloqueo sobre un caso de fraude real
        pagina.click("#btn-ejemplo-fraude")
        pagina.click("button.principal")
        pagina.wait_for_selector("#veredicto:not([hidden])")
        pagina.wait_for_timeout(300)
        pagina.locator("#veredicto").screenshot(path=args.salida / "04_veredicto_bloquear.png")

        # 5. Veredicto de permiso sobre un caso legítimo
        pagina.click("#btn-ejemplo-legitima")
        pagina.click("button.principal")
        pagina.wait_for_timeout(300)
        pagina.locator("#veredicto").screenshot(path=args.salida / "05_veredicto_permitir.png")

        # 6. Modo lote
        pagina.click("#tab-lote")
        if args.csv.exists():
            pagina.set_input_files("#archivo", str(args.csv))
            pagina.wait_for_selector("#resultado-lote:not([hidden])", timeout=60_000)
            pagina.wait_for_timeout(500)
            pagina.locator("#panel-lote").screenshot(path=args.salida / "06_lote.png")

        # 7. Vista estrecha, para evidenciar la adaptación
        estrecha = navegador.new_page(viewport={"width": 430, "height": 1000},
                                      device_scale_factor=2)
        estrecha.goto(args.url, wait_until="networkidle")
        estrecha.click("#btn-ejemplo-fraude")
        estrecha.click("button.principal")
        estrecha.wait_for_selector("#veredicto:not([hidden])")
        estrecha.wait_for_timeout(300)
        estrecha.screenshot(path=args.salida / "07_estrecha.png", full_page=True)

        navegador.close()

    for archivo in sorted(args.salida.glob("*.png")):
        print(f"  {archivo}  ({archivo.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
