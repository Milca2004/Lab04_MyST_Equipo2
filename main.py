"""Punto de entrada único del pipeline: `python main.py`. Semilla global 42.

Dueña: Paula.

Compatibilidad con Windows: Windows arranca subprocesos con "spawn", por lo
que todo el pipeline se ejecuta dentro de `if __name__ == "__main__":` y los
entornos vectorizados de SB3 usan DummyVecEnv por defecto (no SubprocVecEnv).
"""
import sys
from pathlib import Path

# La carpeta se llama code/ (exigido por el enunciado) y Python ya trae un
# módulo estándar llamado `code`; por eso se agrega code/ a la ruta y se
# importan los módulos por su nombre (data, env, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

SEED = 42


def main():
    """Ejecuta el pipeline completo. Pendiente de implementar por Paula."""
    raise NotImplementedError("main.py: pendiente (Paula)")


if __name__ == "__main__":
    main()
