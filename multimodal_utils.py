import json
import time
from uuid import uuid4
from pathlib import Path


def save_multimodal_event(filepath, intent_source, intent_label, **extra_fields):
    """
    Guarda un evento unimodal en JSON para que lo lea el integrador.

    Formato comun:
        source: "text" o "vision"
        intent: etiqueta semantica detectada
        timestamp: instante Unix en segundos
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "source": intent_source,
        "intent": intent_label,
        "timestamp": time.time(),
    }
    event.update(extra_fields)

    temp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=2)

    for attempt in range(12):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.05)
