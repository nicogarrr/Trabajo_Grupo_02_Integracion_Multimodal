from pathlib import Path
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"
EVENT_PATH = PROJECT_DIR / "event_vision.json"
GESTOS_VALIDOS = {
    "corte_cuchillo",
    "pizca_sal",
    "sustituir_ingrediente",
    "None",
}
INTENCIONES_VISUALES = {
    "corte_cuchillo": "aumentar_raciones",
    "pizca_sal": "iniciar_receta",
    "sustituir_ingrediente": "marcar_sustitucion",
}
CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES_REQUIRED = 6

sys.path.append(str(PROJECT_DIR))
from multimodal_utils import save_multimodal_event  # noqa: E402


def crear_reconocedor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se ha encontrado el modelo: {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def publicar_evento_visual(nombre_gesto, confianza):
    intencion_visual = INTENCIONES_VISUALES.get(nombre_gesto)
    if intencion_visual is None:
        return

    save_multimodal_event(
        EVENT_PATH,
        "vision",
        intencion_visual,
        gesture=nombre_gesto,
        confidence=round(float(confianza), 4),
    )
    print(f"Evento visual -> {intencion_visual} [{nombre_gesto}] ({confianza:.2f})")


def main():
    recognizer = crear_reconocedor()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    candidato_estable = None
    frames_estables = 0
    ultimo_gesto_publicado = None

    print(f"Modelo cargado: {MODEL_PATH.name}")
    print("Gestos esperados: {}".format(", ".join(sorted(GESTOS_VALIDOS))))
    print(f"Eventos visuales en: {EVENT_PATH}")
    print("Pulsa 'q' en la ventana de video para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la camara.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognition_result = recognizer.recognize(mp_image)

        texto_pantalla = "Gesto: sin mano detectada"
        color = (0, 0, 255)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            nombre_gesto = top_gesture.category_name or "None"
            confianza = top_gesture.score
            texto_pantalla = f"Gesto: {nombre_gesto} ({confianza:.2f})"
            color = (0, 255, 0) if nombre_gesto in GESTOS_VALIDOS else (0, 165, 255)

            gesto_util = (
                nombre_gesto in INTENCIONES_VISUALES
                and confianza >= CONFIDENCE_THRESHOLD
            )

            if gesto_util:
                if nombre_gesto == candidato_estable:
                    frames_estables += 1
                else:
                    candidato_estable = nombre_gesto
                    frames_estables = 1
            else:
                candidato_estable = None
                frames_estables = 0
                ultimo_gesto_publicado = None

            gesto_estable = frames_estables >= STABLE_FRAMES_REQUIRED
            gesto_nuevo = nombre_gesto != ultimo_gesto_publicado

            if gesto_util and gesto_estable and gesto_nuevo:
                publicar_evento_visual(nombre_gesto, confianza)
                ultimo_gesto_publicado = nombre_gesto
        else:
            candidato_estable = None
            frames_estables = 0
            ultimo_gesto_publicado = None

        cv2.putText(
            frame,
            texto_pantalla,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("ChefZeroWaste - Canal Visual", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
