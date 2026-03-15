from typing import Any

activate = False

def DBG(txt: Any) -> None:
    if activate:
        print(f"DEBUG: {txt}")