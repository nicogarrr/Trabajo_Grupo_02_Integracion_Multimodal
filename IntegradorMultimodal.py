import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
TEXT_EVENT_PATH = PROJECT_DIR / "event_text.json"
VISION_EVENT_PATH = PROJECT_DIR / "event_vision.json"
GAZE_EVENT_PATH = PROJECT_DIR / "event_gaze.json"
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
    "iniciar_receta": ["recomendar_receta", "anadir_ingredientes"],
    "aumentar_raciones": ["ajustar_raciones"],
    "marcar_sustitucion": ["sustituir_ingrediente"],
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

    @property
    def confidence(self):
        """Confianza del canal que produjo este evento."""
        return float(self.payload.get("confidence", 0.5))


class MultimodalIntegrator:
    def __init__(self, time_window_ms=3000):
        self.time_window = timedelta(milliseconds=time_window_ms)
        self.text_events = []
        self.visual_events = []
        self.gaze_events = []
        self.reported_invalid_pairs = set()

    def add_text_event(self, intent, timestamp=None, **payload):
        self.text_events.append(Event("text", intent, self._parse_time(timestamp), payload))
        return self.check_fusion()

    def add_visual_event(self, intent, timestamp=None, **payload):
        self.visual_events.append(Event("vision", intent, self._parse_time(timestamp), payload))
        return self.check_fusion()

    def add_gaze_event(self, intent, timestamp=None, **payload):
        """Registra un evento de mirada/cabeza. No dispara fusion por si solo."""
        self.gaze_events.append(Event("gaze", intent, self._parse_time(timestamp), payload))
        return []

    def add_event(self, event_data):
        source = event_data.get("source")
        intent = event_data.get("intent")
        if source not in {"text", "vision", "gaze"} or not intent:
            return []

        payload = dict(event_data)
        timestamp = payload.pop("timestamp", None)
        payload.pop("source", None)
        payload.pop("intent", None)

        if source == "text":
            return self.add_text_event(intent, timestamp, **payload)
        if source == "gaze":
            return self.add_gaze_event(intent, timestamp, **payload)
        return self.add_visual_event(intent, timestamp, **payload)

    def check_fusion(self):
        self._discard_old_events()
        executed = []

        for text_event in list(self.text_events):
            for visual_event in list(self.visual_events):
                if not self._inside_window(text_event, visual_event):
                    continue

                rule = VALID_FUSIONS.get((text_event.intent, visual_event.intent))
                if rule is not None:
                    result = self._execute_action(text_event, visual_event, rule)
                    executed.append(result)
                    self.text_events.remove(text_event)
                    self.visual_events.remove(visual_event)
                    break

                # --- Ampliacion C: Desambiguacion Mutua ---
                disambiguation = self._try_disambiguate(
                    text_event, visual_event,
                )
                if disambiguation is not None:
                    result = self._execute_action(
                        disambiguation["text_event"],
                        disambiguation["visual_event"],
                        disambiguation["rule"],
                        disambiguation_info=disambiguation["info"],
                    )
                    executed.append(result)
                    self.text_events.remove(text_event)
                    self.visual_events.remove(visual_event)
                    break

                self._report_invalid_pair(text_event, visual_event)

        return executed

    # ------------------------------------------------------------------ #
    #  Ampliacion C – Desambiguacion Mutua                                #
    # ------------------------------------------------------------------ #

    def _try_disambiguate(self, text_event, visual_event):
        """
        Intenta corregir la intencion de un canal con baja confianza usando
        la alta certeza del otro canal.

        Caso 1 – Texto ambiguo, gesto claro:
            Si la confianza textual esta por debajo de LOW_TEXT_CONFIDENCE y la
            confianza visual es >= HIGH_VISUAL_CONFIDENCE, se infiere la
            intencion textual esperada a partir del gesto usando
            VISUAL_TO_TEXT_INTENT.  Si la nueva pareja (texto_corregido, gesto)
            esta en VALID_FUSIONS, se acepta la fusion.

        Caso 2 – Gesto ambiguo, texto claro:
            Si la confianza visual esta por debajo de LOW_VISUAL_CONFIDENCE y la
            confianza textual es >= HIGH_TEXT_CONFIDENCE, se infiere la
            intencion visual esperada a partir del texto usando
            TEXT_TO_VISUAL_INTENT.  Si la nueva pareja (texto, gesto_corregido)
            esta en VALID_FUSIONS, se acepta la fusion.

        Devuelve un dict con los eventos (posiblemente con intent corregido),
        la regla de fusion aplicable y un bloque informativo de
        desambiguacion, o None si no se pudo desambiguar.
        """

        text_conf = self._event_confidence(text_event)
        visual_conf = self._event_confidence(visual_event)

        # Caso 1: texto ambiguo, vision segura
        if text_conf < LOW_TEXT_CONFIDENCE and visual_conf >= HIGH_VISUAL_CONFIDENCE:
            candidates = VISUAL_TO_TEXT_INTENT.get(visual_event.intent, [])
            for corrected_text_intent in candidates:
                rule = VALID_FUSIONS.get((corrected_text_intent, visual_event.intent))
                if rule is not None:
                    info = self._build_disambiguation_info(
                        corrected_channel="text",
                        original_intent=text_event.intent,
                        corrected_intent=corrected_text_intent,
                        corrected_by_channel="vision",
                        corrected_by_intent=visual_event.intent,
                        low_confidence=text_conf,
                        high_confidence=visual_conf,
                    )
                    corrected_text = Event(
                        source=text_event.source,
                        intent=corrected_text_intent,
                        timestamp=text_event.timestamp,
                        payload=text_event.payload,
                    )
                    return {
                        "text_event": corrected_text,
                        "visual_event": visual_event,
                        "rule": rule,
                        "info": info,
                    }

        # Caso 2: gesto ambiguo, texto seguro
        if visual_conf < LOW_VISUAL_CONFIDENCE and text_conf >= HIGH_TEXT_CONFIDENCE:
            corrected_visual_intent = TEXT_TO_VISUAL_INTENT.get(text_event.intent)
            if corrected_visual_intent is not None:
                rule = VALID_FUSIONS.get((text_event.intent, corrected_visual_intent))
                if rule is not None:
                    info = self._build_disambiguation_info(
                        corrected_channel="vision",
                        original_intent=visual_event.intent,
                        corrected_intent=corrected_visual_intent,
                        corrected_by_channel="text",
                        corrected_by_intent=text_event.intent,
                        low_confidence=visual_conf,
                        high_confidence=text_conf,
                    )
                    corrected_visual = Event(
                        source=visual_event.source,
                        intent=corrected_visual_intent,
                        timestamp=visual_event.timestamp,
                        payload=visual_event.payload,
                    )
                    return {
                        "text_event": text_event,
                        "visual_event": corrected_visual,
                        "rule": rule,
                        "info": info,
                    }

        return None

    def _event_confidence(self, event):
        """Extrae la confianza de un evento, con fallback a la clasificacion interna."""
        conf = event.payload.get("confidence")
        if conf is not None:
            return float(conf)
        classification = event.payload.get("classification", {})
        if isinstance(classification, dict):
            model_conf = classification.get("model_confidence")
            if model_conf is not None:
                return float(model_conf)
        return 0.5

    def _build_disambiguation_info(self, **kwargs):
        return {
            "disambiguation": True,
            "corrected_channel": kwargs["corrected_channel"],
            "original_intent": kwargs["original_intent"],
            "corrected_intent": kwargs["corrected_intent"],
            "corrected_by_channel": kwargs["corrected_by_channel"],
            "corrected_by_intent": kwargs["corrected_by_intent"],
            "low_confidence": round(kwargs["low_confidence"], 4),
            "high_confidence": round(kwargs["high_confidence"], 4),
        }

    def _execute_action(self, text_event, visual_event, rule, disambiguation_info=None):
        action_result = self._build_action_result(text_event, visual_event, rule)
        gaze_context = self._find_gaze_context(text_event, visual_event)
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
        if disambiguation_info is not None:
            result["disambiguation"] = disambiguation_info
        if gaze_context is not None:
            result["gaze_context"] = gaze_context
        self._save_fusion(result)

        if disambiguation_info is not None:
            self._print_disambiguation(disambiguation_info)

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
        if gaze_context is not None:
            print(
                "FOCO DE ATENCION: seccion '{}' (zona {}, yaw={}, pitch={})".format(
                    gaze_context.get("section", "?"),
                    gaze_context.get("zone", "?"),
                    gaze_context.get("yaw", "?"),
                    gaze_context.get("pitch", "?"),
                )
            )
        return result

    # ------------------------------------------------------------------ #
    #  Ampliacion D – Contexto de mirada (Gaze)                           #
    # ------------------------------------------------------------------ #

    def _find_gaze_context(self, text_event, visual_event):
        """
        Busca el evento gaze mas reciente que caiga dentro de la ventana
        temporal de la fusion texto+gesto.  El canal gaze es pasivo:
        si no hay evento de mirada disponible, devuelve None y la fusion
        procede normalmente sin contexto espacial.
        """
        if not self.gaze_events:
            return None

        fusion_center = min(text_event.timestamp, visual_event.timestamp)
        best_gaze = None
        best_delta = None

        for gaze_event in self.gaze_events:
            delta = abs((gaze_event.timestamp - fusion_center).total_seconds())
            if delta <= self.time_window.total_seconds():
                if best_delta is None or delta < best_delta:
                    best_gaze = gaze_event
                    best_delta = delta

        if best_gaze is None:
            return None

        return {
            "gaze_intent": best_gaze.intent,
            "zone": best_gaze.payload.get("zone", "desconocida"),
            "section": best_gaze.payload.get("section", "desconocida"),
            "yaw": best_gaze.payload.get("yaw"),
            "pitch": best_gaze.payload.get("pitch"),
            "confidence": best_gaze.payload.get("confidence"),
            "gaze_timestamp": best_gaze.timestamp.timestamp(),
        }

    def _print_disambiguation(self, info):
        print(
            "\n[{}] DESAMBIGUACION MUTUA: canal '{}' corregido por canal '{}'".format(
                datetime.now().strftime("%H:%M:%S"),
                info["corrected_channel"],
                info["corrected_by_channel"],
            )
        )
        print(
            "  Intent original del {}: '{}' (confianza {:.2f})".format(
                info["corrected_channel"],
                info["original_intent"],
                info["low_confidence"],
            )
        )
        print(
            "  Corregido a '{}' gracias a '{}' del canal {} (confianza {:.2f})".format(
                info["corrected_intent"],
                info["corrected_by_intent"],
                info["corrected_by_channel"],
                info["high_confidence"],
            )
        )

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
        self.gaze_events = [
            event for event in self.gaze_events if now - event.timestamp <= self.time_window
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
    print(f"Canal gaze:  {GAZE_EVENT_PATH}")
    print("Pulsa Ctrl+C para salir.")

    last_text_timestamp = latest_event_timestamp(TEXT_EVENT_PATH)
    last_vision_timestamp = latest_event_timestamp(VISION_EVENT_PATH)
    last_gaze_timestamp = latest_event_timestamp(GAZE_EVENT_PATH)
    if last_text_timestamp > 0 or last_vision_timestamp > 0 or last_gaze_timestamp > 0:
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

            gaze_data = read_event_file(GAZE_EVENT_PATH)
            if gaze_data and gaze_data.get("timestamp", 0) > last_gaze_timestamp:
                last_gaze_timestamp = gaze_data["timestamp"]
                zone = gaze_data.get("zone", "?")
                section = gaze_data.get("section", "?")
                print(f"[gaze] {gaze_data.get('intent')} -> {section} ({zone})")
                integrator.add_event(gaze_data)

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

    # ------------------------------------------------------------------ #
    #  Escenarios de Desambiguacion Mutua (Ampliacion C)                  #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DEMO AMPLIACION C: Desambiguacion Mutua")
    print("=" * 60)

    # Escenario 1: Texto ambiguo corregido por gesto claro.
    # El canal PLN recibe una frase ruidosa como "quiero algo de cocina receta"
    # que activa varias reglas y clasifica como 'lista_compra' con confianza 0.30.
    # El gesto pizca_sal se detecta con confianza alta (0.92).
    # El integrador infiere que la intencion textual real era 'anadir_ingredientes'
    # o 'recomendar_receta' (la que empareje con el gesto) y ejecuta la accion.
    print("\n--- Escenario 1: Texto ambiguo, gesto claro ---")
    print("  Texto PLN dice: 'lista_compra' (conf=0.30) -- frase ruidosa")
    print("  Vision dice: 'iniciar_receta' (conf=0.92) -- gesto pizca_sal claro")
    print("  Esperado: el gesto corrige el texto a 'recomendar_receta'.")

    integrator_c1 = MultimodalIntegrator(time_window_ms=3000)
    now_c1 = datetime.now()
    integrator_c1.add_text_event(
        "lista_compra",
        now_c1,
        confidence=0.30,
        entities={"ingredientes": ["tomate"]},
        raw_text="quiero algo de cocina receta",
        classification={"source": "rules", "reason": "frase ambigua: varias intenciones textuales activadas"},
    )
    integrator_c1.add_visual_event(
        "iniciar_receta",
        now_c1 + timedelta(seconds=0.8),
        confidence=0.92,
        gesture="pizca_sal",
    )

    # Escenario 2: Gesto ambiguo corregido por texto claro.
    # El modelo visual detecta 'aumentar_raciones' con confianza muy baja (0.40)
    # porque la mano estaba parcialmente fuera de cuadro.
    # El canal textual dice 'sustituir_ingrediente' con confianza alta (0.95).
    # El integrador infiere que el gesto real era 'marcar_sustitucion'.
    print("\n--- Escenario 2: Gesto ambiguo, texto claro ---")
    print("  Texto PLN dice: 'sustituir_ingrediente' (conf=0.95) -- frase clara")
    print("  Vision dice: 'aumentar_raciones' (conf=0.40) -- gesto borroso")
    print("  Esperado: el texto corrige el gesto a 'marcar_sustitucion'.")

    integrator_c2 = MultimodalIntegrator(time_window_ms=3000)
    now_c2 = datetime.now()
    integrator_c2.add_text_event(
        "sustituir_ingrediente",
        now_c2,
        confidence=0.95,
        entities={"ingrediente": "leche", "restriccion": "sin_lactosa"},
        raw_text="sustituye la leche por algo sin lactosa",
    )
    integrator_c2.add_visual_event(
        "aumentar_raciones",
        now_c2 + timedelta(seconds=1.0),
        confidence=0.40,
        gesture="sustituir_ingrediente",
    )

    # Escenario 3: Ambos canales con baja confianza → no se desambigua.
    # Ni el texto ni la vision son fiables, la fusion se rechaza.
    print("\n--- Escenario 3: Ambos canales con baja confianza ---")
    print("  Texto PLN dice: 'lista_compra' (conf=0.25)")
    print("  Vision dice: 'aumentar_raciones' (conf=0.35)")
    print("  Esperado: NO se desambigua, fusion rechazada.")

    integrator_c3 = MultimodalIntegrator(time_window_ms=3000)
    now_c3 = datetime.now()
    integrator_c3.add_text_event(
        "lista_compra",
        now_c3,
        confidence=0.25,
    )
    integrator_c3.add_visual_event(
        "aumentar_raciones",
        now_c3 + timedelta(seconds=0.5),
        confidence=0.35,
    )

    # ------------------------------------------------------------------ #
    #  Escenarios de Ampliacion D: Head/Gaze Tracking                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DEMO AMPLIACION D: Head/Gaze Tracking (tercer canal)")
    print("=" * 60)

    # Escenario D1: Fusion trimodal — texto + gesto + mirada
    # El usuario dice los ingredientes, hace pizca_sal, y mira hacia la
    # zona izquierda de la pantalla (seccion ingredientes).
    print("\n--- Escenario D1: Fusion trimodal (texto + gesto + mirada) ---")
    print("  Texto: 'anadir_ingredientes'")
    print("  Gesto: 'iniciar_receta' (pizca_sal)")
    print("  Gaze: mirando a la izquierda -> seccion 'ingredientes'")
    print("  Esperado: fusion valida + contexto gaze 'ingredientes'.")

    integrator_d1 = MultimodalIntegrator(time_window_ms=3000)
    now_d1 = datetime.now()
    integrator_d1.add_gaze_event(
        "mirada_ingredientes",
        now_d1,
        zone="izquierda",
        section="ingredientes",
        yaw=-22.5,
        pitch=3.0,
        confidence=0.85,
    )
    integrator_d1.add_text_event(
        "anadir_ingredientes",
        now_d1 + timedelta(seconds=0.5),
        entities={
            "ingredientes": ["tomate", "cebolla"],
            "ingrediente_principal": "tomate",
        },
    )
    integrator_d1.add_visual_event(
        "iniciar_receta",
        now_d1 + timedelta(seconds=1.2),
        confidence=0.91,
        gesture="pizca_sal",
    )

    # Escenario D2: Fusion bimodal sin gaze — funciona igual que antes
    print("\n--- Escenario D2: Fusion bimodal sin gaze (compatibilidad) ---")
    print("  Texto: 'ajustar_raciones'")
    print("  Gesto: 'aumentar_raciones' (corte_cuchillo)")
    print("  Gaze: NO hay evento de mirada")
    print("  Esperado: fusion valida SIN contexto gaze.")

    integrator_d2 = MultimodalIntegrator(time_window_ms=3000)
    now_d2 = datetime.now()
    integrator_d2.add_text_event(
        "ajustar_raciones",
        now_d2,
        entities={"raciones": 6},
    )
    integrator_d2.add_visual_event(
        "aumentar_raciones",
        now_d2 + timedelta(seconds=0.8),
        confidence=0.88,
        gesture="corte_cuchillo",
    )

    # Escenario D3: Sustitucion con mirada a pasos
    print("\n--- Escenario D3: Sustitucion con mirada a 'pasos' ---")
    print("  Texto: 'sustituir_ingrediente'")
    print("  Gesto: 'marcar_sustitucion'")
    print("  Gaze: mirando a la derecha -> seccion 'pasos'")
    print("  Esperado: fusion valida + contexto gaze 'pasos'.")

    integrator_d3 = MultimodalIntegrator(time_window_ms=3000)
    now_d3 = datetime.now()
    integrator_d3.add_gaze_event(
        "mirada_pasos",
        now_d3,
        zone="derecha",
        section="pasos",
        yaw=19.0,
        pitch=1.5,
        confidence=0.82,
    )
    integrator_d3.add_text_event(
        "sustituir_ingrediente",
        now_d3 + timedelta(seconds=0.3),
        entities={"ingrediente": "queso", "restriccion": "vegano"},
    )
    integrator_d3.add_visual_event(
        "marcar_sustitucion",
        now_d3 + timedelta(seconds=1.0),
        confidence=0.90,
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
