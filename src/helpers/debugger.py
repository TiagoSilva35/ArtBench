from typing import Any

activate = True

def DBG(txt: Any) -> None:
    if activate:
        print(f"DEBUG: {txt}")