from pathlib import Path

from mediapipe_model_maker import gesture_recognizer


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
EXPORT_DIR = BASE_DIR / "modelo_exportado"


def main():
    negativas = DATASET_DIR / "negativas"
    ninguno = DATASET_DIR / "ninguno"
    none_dir = DATASET_DIR / "None"

    if negativas.exists() and not none_dir.exists():
        negativas.rename(none_dir)
        print("Carpeta 'negativas' renombrada a 'None'.")
    elif ninguno.exists() and not none_dir.exists():
        ninguno.rename(none_dir)
        print("Carpeta 'ninguno' renombrada a 'None'.")

    print(f"Cargando dataset desde: {DATASET_DIR}")
    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(DATASET_DIR),
        hparams=gesture_recognizer.HandDataPreprocessingParams(),
    )

    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    print(f"Tamano entrenamiento: {len(train_data)}")
    print(f"Tamano validacion: {len(validation_data)}")
    print(f"Tamano test: {len(test_data)}")

    hparams = gesture_recognizer.HParams(
        export_dir=str(EXPORT_DIR),
        epochs=10,
        batch_size=2,
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

    print("\nIniciando entrenamiento con transfer learning...")
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    print("\nEvaluando el modelo en test...")
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss}, Test accuracy: {acc}")

    print("\nExportando modelo...")
    model.export_model()
    print(f"Modelo exportado en: {EXPORT_DIR / 'gesture_recognizer.task'}")


if __name__ == "__main__":
    main()
