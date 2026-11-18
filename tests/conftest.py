"""Configuración común de pytest: permite importar los módulos de code/.

Se agrega code/ a la ruta (no se importa como paquete porque `code` choca
con un módulo de la biblioteca estándar de Python).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
