#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import unicodedata
from urllib.request import urlretrieve
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from multimodal_utils import save_multimodal_event

try:
    import msvcrt
except ImportError:
    msvcrt = None


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
VISION_EVENT_PATH = PROJECT_DIR / "event_vision.json"
TEXT_EVENT_PATH = PROJECT_DIR / "event_text.json"

MODEL_ID = "openai/clip-vit-base-patch32"
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MIN_TEXT_SCORE = 0.35
DEFAULT_MIN_IMAGE_SCORE = 0.20
DEFAULT_IMAGE_MARGIN = 0.04

KNOWN_INGREDIENTS = [
    "tomate",
    "pan",
    "queso",
    "pollo",
    "arroz",
    "pasta",
    "lentejas",
    "garbanzos",
    "huevo",
    "atun",
    "calabacin",
    "cebolla",
    "zanahoria",
    "patata",
]

RESTRICTION_ALIASES = {
    "sin lactosa": "sin_lactosa",
    "lactosa": "sin_lactosa",
    "sin gluten": "sin_gluten",
    "gluten": "sin_gluten",
    "vegano": "vegano",
    "vegetariano": "vegetariano",
}


@dataclass
class GestureSemantic:
    gesture: str
    visual_intent: str
    text_intent: str
    prompts: list[str]


@dataclass
class HandCrop:
    image: np.ndarray
    rect: tuple[int, int, int, int]


GESTURE_SEMANTICS = [
    GestureSemantic(
        gesture="pizca_sal",
        visual_intent="iniciar_receta",
        text_intent="anadir_ingredientes",
        prompts=[
            "a hand making a pinch gesture to start cooking",
            "pinching fingers like adding a pinch of salt",
            "a close up of a hand with thumb and index finger pinched together",
            "thumb and index finger touching in a pinch sign",
            "gesture meaning let's cook a recipe",
            "vamos a cocinar una receta",
            "empezar a cocinar con ingredientes",
        ],
    ),
    GestureSemantic(
        gesture="corte_cuchillo",
        visual_intent="aumentar_raciones",
        text_intent="ajustar_raciones",
        prompts=[
            "a hand gesture like cutting with a knife",
            "knife cutting gesture to divide food into more portions",
            "a flat hand making a chopping motion",
            "a hand shaped like a knife chopping food",
            "gesture meaning split the recipe into more servings",
            "partir la receta en mas raciones",
            "ajustar la receta a mas raciones",
        ],
    ),
    GestureSemantic(
        gesture="sustituir_ingrediente",
        visual_intent="marcar_sustitucion",
        text_intent="sustituir_ingrediente",
        prompts=[
            "a hand crossing gesture meaning replace an ingredient",
            "cross gesture for ingredient substitution",
            "two fingers crossing in an x sign",
            "hands making an x sign to reject or replace something",
            "gesture meaning change this ingredient",
            "sustituir un ingrediente por otro",
            "cambiar ingrediente por restriccion alimentaria",
        ],
    ),
]


class ClipZeroShotMatcher:
    def __init__(self, model_id=MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cargando CLIP: {model_id} ({self.device})")
        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.gesture_prompts = []
        self.prompt_to_semantic = []
        for semantic in GESTURE_SEMANTICS:
            for prompt in semantic.prompts:
                self.gesture_prompts.append(prompt)
                self.prompt_to_semantic.append(semantic)

        self.prompt_embeddings = self.encode_texts(self.gesture_prompts)

    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return normalize(embeddings).cpu().numpy()

    def encode_image(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return normalize(embedding).cpu().numpy()[0]

    def match_text_command(self, command):
        command_embedding = self.encode_texts([command])[0]
        similarities = self.prompt_embeddings @ command_embedding
        return self._best_by_semantic(similarities, "text_command", command)

    def match_camera_frame(self, frame_bgr):
        image_embedding = self.encode_image(frame_bgr)
        similarities = self.prompt_embeddings @ image_embedding
        return self._best_by_semantic(similarities, "camera_frame", None)

    def match_direct_image_text(self, frame_bgr, command):
        image_embedding = self.encode_image(frame_bgr)
        text_embedding = self.encode_texts([command])[0]
        return float(np.dot(image_embedding, text_embedding))

    def _best_by_semantic(self, similarities, source, raw_text):
        scores = {}
        best_prompt = {}
        for index, similarity in enumerate(similarities):
            semantic = self.prompt_to_semantic[index]
            key = semantic.visual_intent
            if key not in scores or similarity > scores[key]:
                scores[key] = float(similarity)
                best_prompt[key] = self.gesture_prompts[index]

        best_intent = max(scores, key=scores.get)
        semantic = next(item for item in GESTURE_SEMANTICS if item.visual_intent == best_intent)
        return {
            "source": source,
            "raw_text": raw_text,
            "gesture": semantic.gesture,
            "visual_intent": semantic.visual_intent,
            "text_intent": semantic.text_intent,
            "score": scores[best_intent],
            "matched_prompt": best_prompt[best_intent],
            "all_scores": scores,
            "best_prompts": best_prompt,
        }


class HandRegionExtractor:
    def __init__(self, padding=0.45):
        self.padding = padding
        model_path = ensure_hand_landmarker_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def crop(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(image)
        if not result.hand_landmarks:
            return None

        best_rect = None
        best_area = 0
        for landmarks in result.hand_landmarks:
            xs = [landmark.x * width for landmark in landmarks]
            ys = [landmark.y * height for landmark in landmarks]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            area = max(1, (x2 - x1) * (y2 - y1))
            if area > best_area:
                best_area = area
                best_rect = (x1, y1, x2, y2)

        if best_rect is None:
            return None

        x1, y1, x2, y2 = best_rect
        box_width = x2 - x1
        box_height = y2 - y1
        pad = int(max(box_width, box_height) * self.padding)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        return HandCrop(frame_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2))

    def close(self):
        self.landmarker.close()


def ensure_hand_landmarker_model():
    if HAND_LANDMARKER_PATH.exists():
        return HAND_LANDMARKER_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando modelo MediaPipe Hands para recortar la mano...")
    print(HAND_LANDMARKER_URL)
    temp_path = HAND_LANDMARKER_PATH.with_suffix(".task.tmp")
    urlretrieve(HAND_LANDMARKER_URL, temp_path)
    temp_path.replace(HAND_LANDMARKER_PATH)
    print(f"Modelo de mano guardado en: {HAND_LANDMARKER_PATH}")
    return HAND_LANDMARKER_PATH


def normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def draw_status(
    frame,
    camera_match,
    command_match,
    direct_score,
    min_image_score,
    min_text_score,
    hand_crop,
):
    lines = [f"Texto: {command_match['visual_intent']} ({command_match['score']:.3f})"]

    if camera_match is None:
        lines.extend(
            [
                "Camara: pendiente",
                "Mano: detectada" if hand_crop is not None else "Mano: no detectada",
                "Pulsa ESPACIO/ENTER para analizar la mano",
                "T cambia texto | Q sale",
            ]
        )
        color = (0, 220, 255)
    else:
        lines.extend(
            [
                f"Camara: {camera_match['visual_intent']} ({camera_match['score']:.3f})",
                f"Imagen-texto: {direct_score:.3f}",
                "ROI: recorte de mano",
            ]
        )

    if camera_match is not None and is_publishable(
        camera_match,
        command_match,
        min_image_score,
        min_text_score,
    ):
        lines.append("ZERO-SHOT: listo para publicar")
        color = (0, 255, 0)
    elif camera_match is not None and camera_match["visual_intent"] == command_match["visual_intent"]:
        lines.append(f"Coincide, pero camara baja (< {min_image_score:.2f})")
        color = (0, 220, 255)
    elif camera_match is not None:
        lines.append("ZERO-SHOT: no coincide")
        color = (0, 165, 255)

    y = 35
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 32

    if hand_crop is not None:
        x1, y1, x2, y2 = hand_crop.rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def is_publishable(camera_match, command_match, min_image_score, min_text_score):
    return (
        camera_match["visual_intent"] == command_match["visual_intent"]
        and camera_match["score"] >= min_image_score
        and command_match["score"] >= min_text_score
    )


def align_camera_match_with_text(camera_match, command_match, max_margin):
    target_intent = command_match["visual_intent"]
    if camera_match["visual_intent"] == target_intent:
        return camera_match, False

    target_score = camera_match["all_scores"].get(target_intent)
    if target_score is None:
        return camera_match, False

    margin = camera_match["score"] - target_score
    if margin > max_margin:
        return camera_match, False

    semantic = next(item for item in GESTURE_SEMANTICS if item.visual_intent == target_intent)
    adjusted = dict(camera_match)
    adjusted.update(
        {
            "gesture": semantic.gesture,
            "visual_intent": semantic.visual_intent,
            "text_intent": semantic.text_intent,
            "score": target_score,
            "matched_prompt": camera_match["best_prompts"].get(target_intent, camera_match["matched_prompt"]),
            "original_visual_intent": camera_match["visual_intent"],
            "clip_margin": margin,
        }
    )
    return adjusted, True


def read_control_key():
    key = cv2.waitKeyEx(1)
    if key != -1:
        return key

    if msvcrt is not None and msvcrt.kbhit():
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()
            return -1
        return ord(char)

    return -1


def normalize_control_key(key):
    if key == -1:
        return None
    if key in (27, ord("q"), ord("Q")):
        return "quit"
    if key in (ord("t"), ord("T")):
        return "text"
    if key in (13, 32):
        return "publish"
    return None


def publish_zero_shot_events(command, command_match, camera_match, direct_score):
    entities = extract_entities(command, command_match["text_intent"])
    save_multimodal_event(
        TEXT_EVENT_PATH,
        "text",
        command_match["text_intent"],
        raw_text=command,
        entities=entities,
        zero_shot=True,
        clip_text_visual_intent=command_match["visual_intent"],
        clip_text_score=round(command_match["score"], 4),
    )
    save_multimodal_event(
        VISION_EVENT_PATH,
        "vision",
        camera_match["visual_intent"],
        gesture=camera_match["gesture"],
        zero_shot=True,
        clip_image_score=round(camera_match["score"], 4),
        clip_direct_image_text_score=round(direct_score, 4),
    )


def extract_entities(command, text_intent):
    normalized = normalize_text(command)
    entities = {}

    if text_intent in {"anadir_ingredientes", "recomendar_receta"}:
        ingredients = [item for item in KNOWN_INGREDIENTS if contains_word(normalized, item)]
        if ingredients:
            entities["ingredientes"] = ingredients
            entities["ingrediente_principal"] = ingredients[0]

    if text_intent == "ajustar_raciones":
        servings = extract_servings(normalized)
        if servings is not None:
            entities["raciones"] = servings

    if text_intent == "sustituir_ingrediente":
        ingredient = next((item for item in KNOWN_INGREDIENTS if contains_word(normalized, item)), None)
        restriction = next(
            (value for alias, value in RESTRICTION_ALIASES.items() if alias in normalized),
            None,
        )
        if ingredient is not None:
            entities["ingrediente"] = ingredient
        if restriction is not None:
            entities["restriccion"] = restriction

    return entities


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text


def contains_word(text, word):
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def extract_servings(text):
    match = re.search(r"\b(\d+)\s*(raciones|personas|platos|comensales)\b", text)
    if match is None:
        return None
    return int(match.group(1))


def run_interactive(args):
    matcher = ClipZeroShotMatcher(args.model)
    hand_extractor = HandRegionExtractor(padding=args.hand_padding)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    command = input("Comando textual libre para emparejar con gesto: ").strip()
    if not command:
        command = "quiero empezar a cocinar con estos ingredientes"

    command_match = matcher.match_text_command(command)
    print_match("Texto", command_match)

    last_camera_match = None
    last_direct_score = 0.0

    print("Controles:")
    print("- ESPACIO o ENTER: recorta la mano, analiza el gesto y publica si texto + gesto coinciden.")
    print("- T: cambiar texto. El nuevo texto se escribe en esta consola.")
    print("- Q o ESC: salir.")
    print("Puedes pulsar las teclas con foco en la ventana de video o en esta consola.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer la camara.")
                break

            hand_crop = hand_extractor.crop(frame)
            display = frame.copy()
            if last_camera_match is None:
                last_direct_score = 0.0

            draw_status(
                display,
                last_camera_match,
                command_match,
                last_direct_score,
                args.min_image_score,
                args.min_text_score,
                hand_crop,
            )
            cv2.imshow("ChefZeroWaste - CLIP Zero-Shot", display)
            action = normalize_control_key(read_control_key())

            if action == "quit":
                break
            if action == "text":
                command = input("Nuevo comando textual libre: ").strip() or command
                command_match = matcher.match_text_command(command)
                print_match("Texto", command_match)
                last_camera_match = None
                continue
            if action == "publish":
                if hand_crop is None:
                    last_camera_match = None
                    print("No se analiza: MediaPipe no detecta la mano. Acercala o subela al centro de la camara.")
                    continue

                raw_camera_match = matcher.match_camera_frame(hand_crop.image)
                last_camera_match, adjusted_by_margin = align_camera_match_with_text(
                    raw_camera_match,
                    command_match,
                    args.image_margin,
                )
                last_direct_score = matcher.match_direct_image_text(hand_crop.image, command)
                print_match("Camara recorte mano", last_camera_match)
                if adjusted_by_margin:
                    print(
                        "Ajuste por margen CLIP: "
                        f"top original='{last_camera_match['original_visual_intent']}', "
                        f"intencion textual='{last_camera_match['visual_intent']}', "
                        f"margen={last_camera_match['clip_margin']:.4f}."
                    )
                print(f"Similitud directa mano-texto: {last_direct_score:.4f}")
                if is_publishable(
                    last_camera_match,
                    command_match,
                    args.min_image_score,
                    args.min_text_score,
                ):
                    publish_zero_shot_events(command, command_match, last_camera_match, last_direct_score)
                    print("Eventos zero-shot publicados para el integrador.")
                elif last_camera_match["visual_intent"] == command_match["visual_intent"]:
                    print(
                        "No se publican eventos: el texto coincide, "
                        f"pero la confianza visual {last_camera_match['score']:.4f} "
                        f"no supera el umbral {args.min_image_score:.2f}."
                    )
                else:
                    print("No se publican eventos: el texto y el gesto no emparejan semanticamente.")
    finally:
        cap.release()
        hand_extractor.close()
        cv2.destroyAllWindows()


def run_self_test(args):
    matcher = ClipZeroShotMatcher(args.model)
    commands = [
        "quiero empezar a cocinar con lo que tengo",
        "divide la receta para mas personas",
        "cambia este ingrediente por una alternativa sin lactosa",
    ]

    print("Autotest CLIP zero-shot texto -> gesto")
    for command in commands:
        match = matcher.match_text_command(command)
        print_match(command, match)

    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.putText(dummy, "ChefZeroWaste", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    image_match = matcher.match_camera_frame(dummy)
    direct_score = matcher.match_direct_image_text(dummy, commands[0])
    print_match("Imagen sintetica", image_match)
    print(f"Similitud directa imagen-texto sintetica: {direct_score:.4f}")
    print("Autotest completado.")


def print_match(label, match):
    print(
        f"{label}: {match['visual_intent']} | gesto={match['gesture']} | "
        f"score={match['score']:.4f} | prompt='{match['matched_prompt']}'"
    )
    scores = " | ".join(
        f"{intent}={score:.4f}"
        for intent, score in sorted(match["all_scores"].items())
    )
    print(f"  Scores: {scores}")


def main():
    parser = argparse.ArgumentParser(description="Ampliacion CLIP zero-shot vision + texto.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--self-test", action="store_true", help="Carga CLIP y prueba embeddings sin camara.")
    parser.add_argument("--min-text-score", type=float, default=DEFAULT_MIN_TEXT_SCORE)
    parser.add_argument("--min-image-score", type=float, default=DEFAULT_MIN_IMAGE_SCORE)
    parser.add_argument("--image-margin", type=float, default=DEFAULT_IMAGE_MARGIN)
    parser.add_argument("--hand-padding", type=float, default=0.45)
    args = parser.parse_args()
    if args.self_test:
        run_self_test(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
