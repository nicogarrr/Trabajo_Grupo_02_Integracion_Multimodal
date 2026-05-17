#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ampliacion D — Head / Gaze Tracking + Reconocimiento de Gestos.

Este script abre la webcam y ejecuta en cada frame:
  1. MediaPipe Gesture Recognizer  -> publica event_vision.json
  2. MediaPipe Face Landmarker + solvePnP -> publica event_gaze.json

La cabeza se rastrea de forma pasiva: se calculan los angulos de Euler
(yaw, pitch, roll) a partir de 6 landmarks faciales y un modelo 3D
generico con cv2.solvePnP.  Los angulos se mapean a zonas logicas de la
pantalla que representan secciones de la interfaz de ChefZeroWaste.

Zonas:
    izquierda  (yaw < -12)  -> ingredientes
    derecha    (yaw >  12)  -> pasos
    arriba     (pitch < -10) -> titulo
    centro     (resto)       -> receta

Uso:
    python vision/gaze_head_tracking.py           # webcam
    python vision/gaze_head_tracking.py --demo     # sin webcam
    python vision/gaze_head_tracking.py --no-gestures  # solo head tracking
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_DIR = Path(__file__).resolve().parent.parent
GESTURE_MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"
FACE_MODEL_PATH = PROJECT_DIR / "models" / "face_landmarker.task"
EVENT_VISION_PATH = PROJECT_DIR / "event_vision.json"
EVENT_GAZE_PATH = PROJECT_DIR / "event_gaze.json"

sys.path.append(str(PROJECT_DIR))
from multimodal_utils import save_multimodal_event  # noqa: E402

# ------------------------------------------------------------------ #
#  Configuracion de gestos (identica a vision/main.py)                #
# ------------------------------------------------------------------ #

INTENCIONES_VISUALES = {
    "corte_cuchillo": "aumentar_raciones",
    "pizca_sal": "iniciar_receta",
    "sustituir_ingrediente": "marcar_sustitucion",
}
GESTURE_CONFIDENCE_THRESHOLD = 0.70
GESTURE_STABLE_FRAMES = 6

# ------------------------------------------------------------------ #
#  Configuracion de Head Tracking                                     #
# ------------------------------------------------------------------ #

# Indices de landmarks clave de Face Landmarker (478 landmarks)
# Usamos 6 puntos: nariz, menton, esquina ojo izq, esquina ojo der,
# esquina boca izq, esquina boca der.
FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Modelo 3D generico del rostro (coordenadas aproximadas en mm)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],           # Punta de la nariz
    [0.0, -63.6, -12.5],       # Menton
    [-43.3, 32.7, -26.0],      # Esquina ojo izquierdo
    [43.3, 32.7, -26.0],       # Esquina ojo derecho
    [-28.9, -28.9, -24.1],     # Esquina boca izquierda
    [28.9, -28.9, -24.1],      # Esquina boca derecha
], dtype=np.float64)

# Umbrales de angulos para mapear zonas
YAW_LEFT_THRESHOLD = -12.0
YAW_RIGHT_THRESHOLD = 12.0
PITCH_UP_THRESHOLD = -10.0

GAZE_STABLE_FRAMES = 8

# Zonas -> intenciones gaze -> secciones de la interfaz
GAZE_ZONES = {
    "izquierda": {"intent": "mirada_ingredientes", "section": "ingredientes"},
    "derecha": {"intent": "mirada_pasos", "section": "pasos"},
    "arriba": {"intent": "mirada_titulo", "section": "titulo"},
    "centro": {"intent": "mirada_receta", "section": "receta"},
}

# Colores para la visualizacion
ZONE_COLORS = {
    "izquierda": (255, 165, 0),   # naranja
    "derecha": (0, 200, 255),     # cyan
    "arriba": (200, 100, 255),    # violeta
    "centro": (0, 255, 0),        # verde
}


# ------------------------------------------------------------------ #
#  Funciones de Head Tracking                                         #
# ------------------------------------------------------------------ #

def estimate_head_pose(landmarks_2d, frame_shape):
    """
    Calcula yaw, pitch y roll a partir de landmarks 2D y el modelo 3D.

    Usa cv2.solvePnP con el metodo SOLVEPNP_ITERATIVE.
    Devuelve (yaw, pitch, roll) en grados o None si falla.
    """
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    points_2d = np.array(landmarks_2d, dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL,
        points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    pitch = float(euler_angles[0][0])
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])

    return yaw, pitch, roll


def classify_gaze_zone(yaw, pitch):
    """Clasifica la zona de la pantalla segun yaw y pitch."""
    if yaw < YAW_LEFT_THRESHOLD:
        return "izquierda"
    if yaw > YAW_RIGHT_THRESHOLD:
        return "derecha"
    if pitch < PITCH_UP_THRESHOLD:
        return "arriba"
    return "centro"


def compute_gaze_confidence(yaw, pitch, zone):
    """
    Calcula una confianza para el evento gaze basada en lo lejos que estan
    los angulos del umbral.  Cuanto mas pronunciado el giro, mayor confianza.
    """
    if zone == "izquierda":
        deviation = abs(yaw - YAW_LEFT_THRESHOLD)
    elif zone == "derecha":
        deviation = abs(yaw - YAW_RIGHT_THRESHOLD)
    elif zone == "arriba":
        deviation = abs(pitch - PITCH_UP_THRESHOLD)
    else:
        deviation = min(
            abs(yaw - YAW_LEFT_THRESHOLD),
            abs(yaw - YAW_RIGHT_THRESHOLD),
            abs(pitch - PITCH_UP_THRESHOLD),
        )
    # Mapear desviacion [0, 30] -> confianza [0.5, 0.99]
    confidence = min(0.99, 0.5 + deviation / 60.0)
    return round(confidence, 4)


def draw_head_axes(frame, landmarks_2d, yaw, pitch, roll):
    """Dibuja los ejes de orientacion de la cabeza sobre el frame."""
    nose_tip = tuple(int(v) for v in landmarks_2d[0])
    axis_length = 60

    # Eje X (rojo) -> yaw
    x_end = (
        int(nose_tip[0] + axis_length * np.cos(np.radians(yaw))),
        int(nose_tip[1] - axis_length * np.sin(np.radians(pitch))),
    )
    cv2.arrowedLine(frame, nose_tip, x_end, (0, 0, 255), 2, tipLength=0.3)

    # Eje Y (verde) -> pitch
    y_end = (
        int(nose_tip[0] - axis_length * np.sin(np.radians(yaw))),
        int(nose_tip[1] - axis_length * np.cos(np.radians(pitch))),
    )
    cv2.arrowedLine(frame, nose_tip, y_end, (0, 255, 0), 2, tipLength=0.3)

    # Eje Z (azul) -> roll
    z_end = (
        int(nose_tip[0] + axis_length * np.sin(np.radians(roll))),
        int(nose_tip[1] + axis_length * np.cos(np.radians(roll))),
    )
    cv2.arrowedLine(frame, nose_tip, z_end, (255, 0, 0), 2, tipLength=0.3)


# ------------------------------------------------------------------ #
#  Creacion de reconocedores (Tasks API)                              #
# ------------------------------------------------------------------ #

def crear_reconocedor_gestos():
    """Crea el reconocedor de gestos de MediaPipe Tasks."""
    if not GESTURE_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se ha encontrado el modelo: {GESTURE_MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH))
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def crear_face_landmarker():
    """Crea el Face Landmarker de MediaPipe Tasks."""
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo de face landmarker: {FACE_MODEL_PATH}\n"
            f"Descargalo con:\n"
            f"  python -c \"import urllib.request; "
            f"urllib.request.urlretrieve("
            f"'https://storage.googleapis.com/mediapipe-models/"
            f"face_landmarker/face_landmarker/float16/latest/"
            f"face_landmarker.task', 'models/face_landmarker.task')\""
        )

    base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ------------------------------------------------------------------ #
#  Funciones de publicacion de eventos                                #
# ------------------------------------------------------------------ #

def publicar_evento_visual(nombre_gesto, confianza):
    """Publica un evento visual en event_vision.json."""
    intencion_visual = INTENCIONES_VISUALES.get(nombre_gesto)
    if intencion_visual is None:
        return

    save_multimodal_event(
        EVENT_VISION_PATH,
        "vision",
        intencion_visual,
        gesture=nombre_gesto,
        confidence=round(float(confianza), 4),
    )
    print(f"[gesto] {intencion_visual} [{nombre_gesto}] ({confianza:.2f})")


def publicar_evento_gaze(zone, yaw, pitch, confidence):
    """Publica un evento gaze en event_gaze.json."""
    zone_info = GAZE_ZONES[zone]
    save_multimodal_event(
        EVENT_GAZE_PATH,
        "gaze",
        zone_info["intent"],
        zone=zone,
        section=zone_info["section"],
        yaw=round(float(yaw), 2),
        pitch=round(float(pitch), 2),
        confidence=confidence,
    )
    print(f"[gaze] {zone_info['intent']} -> seccion '{zone_info['section']}' "
          f"(yaw={yaw:.1f}, pitch={pitch:.1f}, conf={confidence})")


# ------------------------------------------------------------------ #
#  Bucle principal                                                    #
# ------------------------------------------------------------------ #

def main_camera(enable_gestures=True):
    """Bucle principal con webcam: head tracking + gestos opcionales."""
    # Inicializar Face Landmarker (Tasks API)
    try:
        face_landmarker = crear_face_landmarker()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Inicializar Gesture Recognizer (opcional)
    recognizer = None
    if enable_gestures:
        try:
            recognizer = crear_reconocedor_gestos()
        except FileNotFoundError as e:
            print(f"Aviso: {e}")
            print("Gestos desactivados. Solo se ejecutara head tracking.")
            enable_gestures = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    # Estado de estabilidad para gestos
    candidato_gesto = None
    frames_gesto = 0
    ultimo_gesto = None

    # Estado de estabilidad para gaze
    candidato_zona = None
    frames_zona = 0
    ultima_zona_publicada = None

    print("Head tracking + gestos iniciado.")
    print(f"Modelo face landmarker: {FACE_MODEL_PATH.name}")
    if enable_gestures:
        print(f"Modelo gestos: {GESTURE_MODEL_PATH.name}")
    print(f"Eventos gaze en: {EVENT_GAZE_PATH}")
    if enable_gestures:
        print(f"Eventos gestos en: {EVENT_VISION_PATH}")
    print("Pulsa 'q' en la ventana de video para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la camara.")
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ---- HEAD TRACKING (Face Landmarker Tasks API) ----
        face_result = face_landmarker.detect(mp_image)
        gaze_text = "Gaze: sin cara detectada"
        gaze_color = (128, 128, 128)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            points_2d = []
            for idx in FACE_LANDMARK_INDICES:
                lm = face_landmarks[idx]
                points_2d.append([lm.x * w, lm.y * h])

            pose = estimate_head_pose(points_2d, frame.shape)
            if pose is not None:
                yaw, pitch, roll = pose
                zone = classify_gaze_zone(yaw, pitch)
                confidence = compute_gaze_confidence(yaw, pitch, zone)
                gaze_color = ZONE_COLORS.get(zone, (255, 255, 255))
                section = GAZE_ZONES[zone]["section"]
                gaze_text = f"Gaze: {section} (yaw={yaw:.0f} pitch={pitch:.0f})"

                # Dibujar ejes de orientacion
                draw_head_axes(frame, points_2d, yaw, pitch, roll)

                # Estabilidad de zona
                if zone == candidato_zona:
                    frames_zona += 1
                else:
                    candidato_zona = zone
                    frames_zona = 1

                if frames_zona >= GAZE_STABLE_FRAMES and zone != ultima_zona_publicada:
                    publicar_evento_gaze(zone, yaw, pitch, confidence)
                    ultima_zona_publicada = zone
            else:
                candidato_zona = None
                frames_zona = 0
        else:
            candidato_zona = None
            frames_zona = 0
            ultima_zona_publicada = None

        # ---- GESTURE RECOGNITION ----
        gesture_text = ""
        gesture_color = (0, 0, 255)

        if enable_gestures and recognizer is not None:
            recognition_result = recognizer.recognize(mp_image)

            gesture_text = "Gesto: sin mano"

            if recognition_result.gestures:
                top = recognition_result.gestures[0][0]
                nombre = top.category_name or "None"
                conf = top.score
                gesture_text = f"Gesto: {nombre} ({conf:.2f})"
                gesture_color = (0, 255, 0) if nombre in INTENCIONES_VISUALES else (0, 165, 255)

                gesto_util = nombre in INTENCIONES_VISUALES and conf >= GESTURE_CONFIDENCE_THRESHOLD

                if gesto_util:
                    if nombre == candidato_gesto:
                        frames_gesto += 1
                    else:
                        candidato_gesto = nombre
                        frames_gesto = 1
                else:
                    candidato_gesto = None
                    frames_gesto = 0
                    ultimo_gesto = None

                if gesto_util and frames_gesto >= GESTURE_STABLE_FRAMES and nombre != ultimo_gesto:
                    publicar_evento_visual(nombre, conf)
                    ultimo_gesto = nombre
            else:
                candidato_gesto = None
                frames_gesto = 0
                ultimo_gesto = None

        # ---- OVERLAY ----
        # Zona gaze con fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 5), (w - 10, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, gaze_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, gaze_color, 2, cv2.LINE_AA)

        # Gesto
        if gesture_text:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (10, 50), (w - 10, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, gesture_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                         0.7, gesture_color, 2, cv2.LINE_AA)

        # Indicador de zona en esquina
        if candidato_zona:
            zone_label = f"[{GAZE_ZONES[candidato_zona]['section'].upper()}]"
            cv2.putText(frame, zone_label, (w - 220, h - 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                         ZONE_COLORS.get(candidato_zona, (255, 255, 255)),
                         2, cv2.LINE_AA)

        cv2.imshow("ChefZeroWaste - Head Tracking + Gestos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()


# ------------------------------------------------------------------ #
#  Demo sin webcam                                                    #
# ------------------------------------------------------------------ #

def run_demo():
    """Simula eventos gaze sin webcam para probar la integracion."""
    print("Demo de Head/Gaze Tracking (sin webcam).")
    print("Simulando eventos de mirada hacia distintas zonas...")

    scenarios = [
        {"zone": "izquierda", "yaw": -22.5, "pitch": 3.0,
         "desc": "Usuario mira a la izquierda -> seccion ingredientes"},
        {"zone": "derecha", "yaw": 18.3, "pitch": 1.2,
         "desc": "Usuario mira a la derecha -> seccion pasos"},
        {"zone": "arriba", "yaw": 2.1, "pitch": -15.8,
         "desc": "Usuario mira arriba -> seccion titulo"},
        {"zone": "centro", "yaw": 1.5, "pitch": -2.0,
         "desc": "Usuario mira al centro -> seccion receta"},
    ]

    for scenario in scenarios:
        zone = scenario["zone"]
        yaw = scenario["yaw"]
        pitch = scenario["pitch"]
        confidence = compute_gaze_confidence(yaw, pitch, zone)
        print(f"\n  {scenario['desc']}")
        publicar_evento_gaze(zone, yaw, pitch, confidence)
        time.sleep(0.1)

    print("\nDemo de gaze completada.")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Head/Gaze Tracking + Reconocimiento de Gestos para ChefZeroWaste."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Simular eventos gaze sin webcam.")
    parser.add_argument("--no-gestures", action="store_true",
                        help="Solo head tracking, sin reconocimiento de gestos.")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        main_camera(enable_gestures=not args.no_gestures)


if __name__ == "__main__":
    main()
