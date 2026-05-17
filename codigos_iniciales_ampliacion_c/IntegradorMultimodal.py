import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
TEXT_EVENT_PATH = PROJECT_DIR / "event_text.json"
VISION_EVENT_PATH = PROJECT_DIR / "event_vision.json"
FUSION_EVENT_PATH = PROJECT_DIR / "event_fusion.json"


RECIPE_LIBRARY = {
    "tomate": {
        "nombre": "Tostas de tomate aprovechado",
        "extras": ["pan", "ajo", "aceite"],
        "pasos": [
            "Rallar el tomate con ajo y aceite.",
            "Tostar el pan.",
            "Cubrir el pan y servir con hojas verdes si quedan.",
        ],
    },
    "pasta": {
        "nombre": "Pasta salteada de nevera",
        "extras": ["ajo", "aceite", "queso"],
        "pasos": [
            "Cocer la pasta.",
            "Saltear los restos de verduras.",
            "Mezclar todo y terminar con queso si hay.",
        ],
    },
    "arroz": {
        "nombre": "Arroz rapido de aprovechamiento",
        "extras": ["cebolla", "aceite", "huevo"],
        "pasos": [
            "Sofreir cebolla o verdura picada.",
            "Anadir arroz cocido o rapido.",
            "Terminar con huevo revuelto.",
        ],
    },
    "default": {
        "nombre": "Bol Chef Zero Waste",
        "extras": ["aceite", "ajo", "pan"],
        "pasos": [
            "Cortar los ingredientes disponibles.",
            "Saltear primero los ingredientes duros.",
            "Servir con pan, arroz o pasta si hace falta base.",
        ],
    },
}

REPLACEMENTS = {
    "sin_lactosa": {
        "queso": "queso sin lactosa o levadura nutricional",
        "leche": "bebida vegetal",
        "yogur": "yogur vegetal",
        "default": "version vegetal o sin lactosa",
    },
    "sin_gluten": {
        "pan": "pan sin gluten o tortitas de maiz",
        "pasta": "pasta de maiz o arroz",
        "default": "arroz, quinoa o patata",
    },
    "vegetariano": {
        "pollo": "tofu salteado",
        "atun": "garbanzos machacados con limon",
        "default": "legumbres o tofu",
    },
    "vegano": {
        "huevo": "harina de garbanzo con agua",
        "queso": "levadura nutricional",
        "pollo": "tofu o garbanzos",
        "default": "legumbres, tofu o frutos secos",
    },
}


VALID_FUSIONS = {
    ("anadir_ingredientes", "iniciar_receta"): {
        "action": "recomendar_receta_con_ingredientes",
        "message": "Iniciar una recomendacion de receta usando los ingredientes indicados por texto.",
    },
    ("recomendar_receta", "iniciar_receta"): {
        "action": "confirmar_recomendacion_receta",
        "message": "Confirmar el inicio de la receta solicitada por texto.",
    },
    ("ajustar_raciones", "aumentar_raciones"): {
        "action": "partir_en_mas_raciones",
        "message": "Ajustar la receta al numero de raciones indicado por texto.",
    },
    ("sustituir_ingrediente", "marcar_sustitucion"): {
        "action": "sustituir_ingrediente",
        "message": "Aplicar la sustitucion del ingrediente indicado por texto.",
    },
}

TEXT_TO_VISUAL_INTENT = {
    "anadir_ingredientes": "iniciar_receta",
    "recomendar_receta": "iniciar_receta",
    "ajustar_raciones": "aumentar_raciones",
    "sustituir_ingrediente": "marcar_sustitucion",
}

VISUAL_TO_TEXT_INTENT = {
    "aumentar_raciones": "ajustar_raciones",
    "marcar_sustitucion": "sustituir_ingrediente",
}

LOW_TEXT_CONFIDENCE = 0.45
HIGH_TEXT_CONFIDENCE = 0.78
LOW_VISUAL_CONFIDENCE = 0.55
HIGH_VISUAL_CONFIDENCE = 0.78


@dataclass
class Event:
    source: str
    intent: str
    timestamp: datetime
    payload: dict


class MultimodalIntegrator:
    def __init__(self, time_window_ms=3000):
        self.time_window = timedelta(milliseconds=time_window_ms)
        self.text_events = []
        self.visual_events = []
        self.reported_invalid_pairs = set()

    def add_text_event(self, intent, timestamp=None, **payload):
        self.text_events.append(Event("text", intent, self._parse_time(timestamp), payload))
        return self.check_fusion()

    def add_visual_event(self, intent, timestamp=None, **payload):
        self.visual_events.append(Event("vision", intent, self._parse_time(timestamp), payload))
        return self.check_fusion()

    def add_event(self, event_data):
        source = event_data.get("source")
        intent = event_data.get("intent")
        if source not in {"text", "vision"} or not intent:
            return []

        payload = dict(event_data)
        timestamp = payload.pop("timestamp", None)
        payload.pop("source", None)
        payload.pop("intent", None)

        if source == "text":
            return self.add_text_event(intent, timestamp, **payload)
        return self.add_visual_event(intent, timestamp, **payload)

    def check_fusion(self):
        self._discard_old_events()
        executed = []

        for text_event in list(self.text_events):
            for visual_event in list(self.visual_events):
                if not self._inside_window(text_event, visual_event):
                    continue

                rule = VALID_FUSIONS.get((text_event.intent, visual_event.intent))
                if rule is None:
                    self._report_invalid_pair(text_event, visual_event)
                    continue

                result = self._execute_action(text_event, visual_event, rule)
                executed.append(result)
                self.text_events.remove(text_event)
                self.visual_events.remove(visual_event)
                break

        return executed

    def _execute_action(self, text_event, visual_event, rule):
        action_result = self._build_action_result(text_event, visual_event, rule)
        result = {
            "source": "fusion",
            "action": rule["action"],
            "message": rule["message"],
            "action_result": action_result,
            "text_intent": text_event.intent,
            "visual_intent": visual_event.intent,
            "text_timestamp": text_event.timestamp.timestamp(),
            "visual_timestamp": visual_event.timestamp.timestamp(),
            "fusion_timestamp": time.time(),
            "delta_ms": round(
                abs((text_event.timestamp - visual_event.timestamp).total_seconds()) * 1000,
                2,
            ),
            "text_payload": text_event.payload,
            "visual_payload": visual_event.payload,
        }
        self._save_fusion(result)
        print(
            "\n[{}] FUSION VALIDA: texto='{}' + gesto='{}' ({} ms)".format(
                datetime.now().strftime("%H:%M:%S"),
                text_event.intent,
                visual_event.intent,
                result["delta_ms"],
            )
        )
        print("ACCION FINAL:", rule["message"])
        self._print_action_result(action_result)
        return result

    def _build_action_result(self, text_event, visual_event, rule):
        entities = text_event.payload.get("entities") or {}
        action = rule["action"]

        if action in {"recomendar_receta_con_ingredientes", "confirmar_recomendacion_receta"}:
            ingredients = entities.get("ingredientes") or []
            principal = entities.get("ingrediente_principal") or (ingredients[0] if ingredients else None)
            return {"tipo": "receta", "receta": build_recipe(principal, ingredients, entities)}

        if action == "partir_en_mas_raciones":
            servings = entities.get("raciones") or 2
            return {
                "tipo": "raciones",
                "raciones": servings,
                "nota": f"Dividir o escalar la receta para {servings} raciones.",
            }

        if action == "sustituir_ingrediente":
            ingredient = entities.get("ingrediente")
            restriction = entities.get("restriccion")
            replacement = replacement_for(ingredient, restriction)
            return {
                "tipo": "sustitucion",
                "ingrediente": ingredient,
                "restriccion": restriction,
                "sustituto": replacement,
            }

        return {"tipo": "accion", "descripcion": rule["message"]}

    def _print_action_result(self, action_result):
        result_type = action_result.get("tipo")

        if result_type == "receta":
            recipe = action_result["receta"]
            print("RECETA PROPUESTA:", recipe["nombre"])
            print("Ingredientes:", ", ".join(recipe["ingredientes"]))
            print(f"Tiempo: {recipe['tiempo']} minutos | Raciones: {recipe['raciones']}")
            print("Pasos:")
            for index, step in enumerate(recipe["pasos"], start=1):
                print(f"  {index}. {step}")
            if recipe["faltan"]:
                print("Faltaria comprar:", ", ".join(recipe["faltan"]))
            else:
                print("No faltan ingredientes basicos.")

        elif result_type == "raciones":
            print(f"RACIONES: {action_result['nota']}")

        elif result_type == "sustitucion":
            print(
                "SUSTITUCION:",
                f"{action_result.get('ingrediente')} -> {action_result.get('sustituto')}",
                f"({action_result.get('restriccion')})",
            )

    def _report_invalid_pair(self, text_event, visual_event):
        pair_key = (
            text_event.intent,
            visual_event.intent,
            text_event.timestamp.timestamp(),
        )
        if pair_key in self.reported_invalid_pairs:
            return
        self.reported_invalid_pairs.add(pair_key)
        print(
            "[{}] Fusion ignorada: texto='{}' + gesto='{}' no es una combinacion valida.".format(
                datetime.now().strftime("%H:%M:%S"),
                text_event.intent,
                visual_event.intent,
            )
        )

    def _inside_window(self, text_event, visual_event):
        return abs(text_event.timestamp - visual_event.timestamp) <= self.time_window

    def _discard_old_events(self):
        now = datetime.now()
        self.text_events = [
            event for event in self.text_events if now - event.timestamp <= self.time_window
        ]
        self.visual_events = [
            event for event in self.visual_events if now - event.timestamp <= self.time_window
        ]

    def _save_fusion(self, result):
        temp_path = FUSION_EVENT_PATH.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        temp_path.replace(FUSION_EVENT_PATH)

    def _parse_time(self, timestamp):
        if timestamp is None:
            return datetime.now()
        if isinstance(timestamp, datetime):
            return timestamp
        return datetime.fromtimestamp(float(timestamp))


def build_recipe(principal, ingredients, entities):
    available = unique([value for value in ingredients if value])
    ingredient = principal or (available[0] if available else "default")
    profile = RECIPE_LIBRARY.get(ingredient, RECIPE_LIBRARY["default"])
    base_ingredients = [] if ingredient == "default" else [ingredient]
    recipe_ingredients = unique(base_ingredients + profile["extras"] + available)
    missing = [item for item in profile["extras"] if item not in available]

    return {
        "nombre": profile["nombre"],
        "ingrediente_principal": None if ingredient == "default" else ingredient,
        "tiempo": entities.get("tiempo") or 25,
        "raciones": entities.get("raciones") or 2,
        "ingredientes": recipe_ingredients,
        "faltan": missing,
        "pasos": profile["pasos"],
    }


def replacement_for(ingredient, restriction):
    if ingredient is None or restriction is None:
        return "sustituto compatible disponible"
    table = REPLACEMENTS.get(restriction, {})
    return table.get(ingredient, table.get("default", "otro ingrediente disponible"))


def unique(values):
    result = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return result


def read_event_file(filepath):
    for attempt in range(8):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, PermissionError):
            if attempt == 7:
                return None
            time.sleep(0.03)
    return None


def latest_event_timestamp(filepath):
    event = read_event_file(filepath)
    if event is None:
        return 0.0
    return float(event.get("timestamp", 0.0))


def run_integrator(time_window_ms):
    integrator = MultimodalIntegrator(time_window_ms=time_window_ms)
    print("Integrador multimodal iniciado.")
    print(f"Ventana temporal: {time_window_ms} ms")
    print(f"Canal texto: {TEXT_EVENT_PATH}")
    print(f"Canal vision: {VISION_EVENT_PATH}")
    print("Pulsa Ctrl+C para salir.")

    last_text_timestamp = latest_event_timestamp(TEXT_EVENT_PATH)
    last_vision_timestamp = latest_event_timestamp(VISION_EVENT_PATH)
    if last_text_timestamp > 0 or last_vision_timestamp > 0:
        print("Ignorando eventos JSON existentes antes del arranque.")

    try:
        while True:
            text_data = read_event_file(TEXT_EVENT_PATH)
            if text_data and text_data.get("timestamp", 0) > last_text_timestamp:
                last_text_timestamp = text_data["timestamp"]
                print(f"[texto] {text_data.get('intent')}")
                integrator.add_event(text_data)

            vision_data = read_event_file(VISION_EVENT_PATH)
            if vision_data and vision_data.get("timestamp", 0) > last_vision_timestamp:
                last_vision_timestamp = vision_data["timestamp"]
                confidence = vision_data.get("confidence", "?")
                print(f"[vision] {vision_data.get('intent')} ({confidence})")
                integrator.add_event(vision_data)

            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nIntegrador detenido.")


def run_demo():
    integrator = MultimodalIntegrator(time_window_ms=3000)
    now = datetime.now()

    print("Demo de fusion tardia con eventos asincronos.")
    integrator.add_text_event(
        "anadir_ingredientes",
        now,
        entities={
            "ingredientes": ["tomate", "pan", "queso"],
            "ingrediente_principal": "tomate",
            "raciones": 2,
        },
    )
    integrator.add_visual_event(
        "iniciar_receta",
        now + timedelta(seconds=1.2),
        gesture="pizca_sal",
    )

    integrator.add_visual_event(
        "iniciar_receta",
        now + timedelta(seconds=4.0),
        gesture="pizca_sal",
    )
    integrator.add_text_event("lista_compra", now + timedelta(seconds=4.8))

    integrator.add_text_event(
        "ajustar_raciones",
        now + timedelta(seconds=10.0),
        entities={"raciones": 4},
    )
    integrator.add_visual_event(
        "aumentar_raciones",
        now + timedelta(seconds=11.1),
        confidence=0.9,
        gesture="corte_cuchillo",
    )

    integrator.add_text_event(
        "sustituir_ingrediente",
        now + timedelta(seconds=15.0),
        entities={"ingrediente": "queso", "restriccion": "sin_lactosa"},
    )
    integrator.add_visual_event(
        "marcar_sustitucion",
        now + timedelta(seconds=16.0),
        confidence=0.93,
        gesture="sustituir_ingrediente",
    )


def main():
    parser = argparse.ArgumentParser(description="Integrador de fusion semantica tardia.")
    parser.add_argument("--window-ms", type=int, default=3000)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_integrator(args.window_ms)


if __name__ == "__main__":
    main()
