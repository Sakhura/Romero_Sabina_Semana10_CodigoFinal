"""
conftest.py
-----------
Agrega la carpeta `src/` al PYTHONPATH para que las pruebas de
integración puedan hacer `import data_ingestion`, `import eda`, etc.,
igual que lo hacen los scripts `main.py` y los módulos entre sí.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))
sys.path.insert(0, str(RAIZ_PROYECTO))
