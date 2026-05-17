# Canal Visual - Gestos ChefZeroWaste

Esta carpeta contiene el canal visual de la practica final.

## Gestos

- `pizca_sal`: iniciar receta, como gesto de "vamos a cocinar".
- `corte_cuchillo`: partir la receta en mas raciones.
- `sustituir_ingrediente`: gesto de cruz para sustitucion de ingredientes.
- `None`: clase negativa.

## Archivos

- `dataset/`: imagenes capturadas por clase.
- `Entrenamiento_Gestos.ipynb`: entrenamiento con Model Maker en Colab.
- `gesture_recognizer.task`: modelo final exportado.
- `main.py`: prueba de webcam y publicacion de eventos JSON.
- `captura_datos.py`: captura local de nuevas muestras.
- `entrenar_gestos.py`: entrenamiento local equivalente al cuaderno.

## Probar el modelo

Desde la carpeta raiz del proyecto:

```powershell
python vision\main.py
```

El script escribe los gestos reconocidos en `../event_vision.json` cuando la confianza supera `0.60`.
