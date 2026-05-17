from pathlib import Path

import cv2


GESTOS_RECOMENDADOS = [
    "corte_cuchillo",
    "pizca_sal",
    "sustituir_ingrediente",
    "None",
]
DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def main():
    print("Gestos de la entrega:")
    for gesto in GESTOS_RECOMENDADOS:
        print(f"- {gesto}")

    nombre_gesto = input("Introduce el nombre del gesto a capturar: ").strip()
    if not nombre_gesto:
        print("Nombre de gesto vacio. Cancelado.")
        return

    carpeta_destino = DATASET_DIR / nombre_gesto
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    contador = len(list(carpeta_destino.glob("muestra_*.jpg")))

    print("\n--- MODO CAPTURA ---")
    print(f"Guardando imagenes en: {carpeta_destino}")
    print("1. Ponte frente a la camara y haz el gesto.")
    print("2. Pulsa ESPACIO para tomar una foto.")
    print("3. Cambia angulo/luz entre fotos.")
    print("4. Pulsa 'q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la camara web.")
            break

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Gesto: {nombre_gesto}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Fotos capturadas: {contador}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            "ESPACIO: foto | Q: salir",
            (20, display_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Captura Dataset Gestos ChefZeroWaste", display_frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 32:
            nombre_archivo = carpeta_destino / f"muestra_{contador}.jpg"
            cv2.imwrite(str(nombre_archivo), frame)
            print(f"Imagen guardada: {nombre_archivo}")
            contador += 1
        elif tecla == ord("q"):
            print("Cerrando el capturador.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
