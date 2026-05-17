# Practica Final - Vision Artificial e Integracion Multimodal

Proyecto final de IMM basado en ChefZeroWaste. El sistema combina:

- Canal visual: reconocimiento de gestos personalizados con MediaPipe.
- Canal textual: clasificacion de intenciones PLN del agente ChefZeroWaste.
- Integrador: fusion semantica tardia mediante eventos JSON con timestamp.

## Diccionario visual

Modelo entrenado:

`vision/gesture_recognizer.task`

Dataset local:

`vision/dataset`

Clases del dataset:

- `pizca_sal`: gesto para indicar "vamos a cocinar" e iniciar una recomendacion de receta.
- `corte_cuchillo`: gesto para partir la receta en mas raciones.
- `sustituir_ingrediente`: gesto de cruz para marcar sustitucion de ingredientes.
- `None`: clase negativa.

El entrenamiento se hizo con transfer learning mediante Model Maker en:

`vision/Entrenamiento_Gestos.ipynb`

## Canal de texto

El PLN parte del agente conversacional ChefZeroWaste del trabajo anterior. Clasifica estas intenciones:

- `instrucciones`
- `anadir_ingredientes`
- `recomendar_receta`
- `sustituir_ingrediente`
- `ajustar_raciones`
- `lista_compra`

Cada intencion textual reconocida se guarda en:

`event_text.json`

Formato:

```json
{
  "source": "text",
  "intent": "recomendar_receta",
  "timestamp": 1778945739.28,
  "raw_text": "quiero una receta con tomate",
  "entities": {}
}
```

## Canal visual

El script `vision/main.py` abre la webcam, carga el modelo entrenado y guarda eventos en:

`event_vision.json`

Formato:

```json
{
  "source": "vision",
  "intent": "iniciar_receta",
  "gesture": "pizca_sal",
  "timestamp": 1778945740.10,
  "confidence": 0.91
}
```

## Fusion semantica tardia

`IntegradorMultimodal.py` mantiene una ventana temporal por defecto de 3000 ms. Si un evento textual y un evento visual caen dentro de esa ventana, solo ejecuta accion final si la combinacion esta en la tabla semantica.

Combinaciones validas:

- `anadir_ingredientes` + `iniciar_receta` -> recomendar receta usando los ingredientes indicados.
- `recomendar_receta` + `iniciar_receta` -> confirmar el inicio de la receta solicitada.
- `ajustar_raciones` + `aumentar_raciones` -> partir la receta en mas raciones.
- `sustituir_ingrediente` + `marcar_sustitucion` -> aplicar la sustitucion indicada por texto.

Ejemplo de combinacion rechazada:

- `lista_compra` + `iniciar_receta` no ejecuta accion final, aunque llegue dentro de la ventana.

La ultima fusion valida se escribe en:

`event_fusion.json`

## Ejecucion

Lanzador automatico de las tres terminales:

```powershell
powershell -ExecutionPolicy Bypass -File .\lanzar_app_final.ps1
```

Terminal 1: integrador.

```powershell
cd "C:\Users\Control Lunar\Desktop\IMM_Trabajos\Grupo\Trabajo_Grupo_02_Integracion_Multimodal"
python IntegradorMultimodal.py
```

Terminal 2: canal visual.

```powershell
cd "C:\Users\Control Lunar\Desktop\IMM_Trabajos\Grupo\Trabajo_Grupo_02_Integracion_Multimodal"
python vision\main.py
```

Terminal 3: canal de texto.

```powershell
cd "C:\Users\Control Lunar\Desktop\IMM_Trabajos\Grupo\Trabajo_Grupo_02_Integracion_Multimodal"
python main_chef_zero_waste.py
```

Cuando el chat pregunte si la categoria es correcta, pulsa `Enter` para aceptar o `n` para corregirla.

## Demo sin webcam

Para probar la fusion sin abrir la camara:

```powershell
python IntegradorMultimodal.py --demo
```

La demo muestra tres fusiones validas, una fusion ignorada y eventos asincronos.

## Ampliacion CLIP Zero-Shot

Esta ampliacion usa `openai/clip-vit-base-patch32` mediante `transformers`.
La primera ejecucion descarga el modelo en la cache local de Hugging Face.
En este equipo ya quedo cacheado en:

`C:\Users\Control Lunar\.cache\huggingface\hub\models--openai--clip-vit-base-patch32`

En este modo, el sistema proyecta:

- el recorte de la mano detectada en el fotograma de la camara,
- un comando textual libre escrito por el usuario,
- descripciones textuales de los tres gestos obligatorios,

en un espacio comun de embeddings CLIP. Despues calcula similitud coseno para emparejar el comando arbitrario con el gesto semanticamente mas cercano.
La deteccion de mano se usa solo para recortar la region relevante antes de pasarla a CLIP; no sustituye el emparejamiento por embeddings.
El modelo de recorte de mano esta en:

`models/hand_landmarker.task`

Autotest sin camara:

```powershell
python clip_zero_shot_multimodal.py --self-test
```

Salida esperada resumida:

```text
quiero empezar a cocinar con lo que tengo -> iniciar_receta / pizca_sal
divide la receta para mas personas -> aumentar_raciones / corte_cuchillo
cambia este ingrediente por una alternativa sin lactosa -> marcar_sustitucion / sustituir_ingrediente
```

Lanzador:

```powershell
powershell -ExecutionPolicy Bypass -File .\lanzar_ampliacion_clip.ps1
```

Uso:

1. Se abre el integrador y el script CLIP.
2. En CLIP escribe un comando libre, por ejemplo:

```text
quiero cocinar con tomate pan y queso
```

3. Haz el gesto correspondiente ante la camara.
4. Comprueba que aparece el rectangulo sobre la mano.
5. Pulsa `ESPACIO` o `ENTER` para analizar el recorte actual.
6. Si el texto y el recorte de mano emparejan con la misma intencion, CLIP publica `event_text.json` y `event_vision.json` para el integrador.

Ejemplos de comandos libres:

- `quiero empezar a cocinar con tomate pan y queso`
- `divide la receta para 4 personas`
- `cambia queso por una alternativa sin lactosa`

El modo CLIP extrae entidades basicas del comando libre: ingredientes conocidos, numero de raciones y restricciones como `sin_lactosa`, para que el integrador pueda imprimir una accion final concreta.

Controles:

- `ESPACIO` o `ENTER`: analiza el recorte de la mano actual y publica si texto + gesto coinciden.
- `T`: cambia el comando textual. El nuevo texto se escribe en la consola PowerShell, no dentro de la ventana de video.
- `Q` o `ESC`: sale del modo CLIP.

Nota: este modo usa la camara directamente con CLIP. No debe ejecutarse a la vez que `vision/main.py`.

## Ampliacion Voz ASR

Esta ampliacion elimina la escritura manual del canal textual. En su lugar usa el microfono y `faster-whisper` para transcribir audio real en castellano. La frase transcrita entra al mismo modulo PLN de ChefZeroWaste, que clasifica la intencion y publica `event_text.json`.

Lanzador:

```powershell
powershell -ExecutionPolicy Bypass -File .\lanzar_ampliacion_voz.ps1
```

Este modo abre:

- integrador con ventana temporal amplia de 15000 ms,
- canal visual `vision/main.py`,
- canal de voz `voz_asr_chef_zero_waste.py`.

Uso recomendado para demostrar la asincronia real:

1. Haz primero el gesto visual.
2. Despues di la frase por el microfono.
3. El canal ASR espera a que termine la frase, la transcribe y publica el evento textual.
4. El integrador fusiona el texto con el gesto que ya estaba guardado en memoria.

Ejemplos de frases habladas:

- `tengo tomate pan y queso`
- `divide la receta para cuatro personas`
- `cambia queso por una alternativa sin lactosa`

Comandos utiles:

```powershell
python voz_asr_chef_zero_waste.py --list-devices
python voz_asr_chef_zero_waste.py --demo-text "tengo tomate pan y queso"
python voz_asr_chef_zero_waste.py --input-device 1
```

## Ficheros principales

- `main_chef_zero_waste.py`: canal textual PLN.
- `CanalTextoChefZeroWaste.py`: clasificador textual y publicacion de eventos `event_text.json`.
- `voz_asr_chef_zero_waste.py`: ampliacion ASR, transcribe microfono y alimenta el canal PLN.
- `vision/main.py`: canal visual con MediaPipe.
- `IntegradorMultimodal.py`: fusion semantica tardia.
- `lanzar_app_final.ps1`: abre las tres terminales necesarias para la demo final.
- `clip_zero_shot_multimodal.py`: ampliacion con embeddings multimodales CLIP.
- `lanzar_ampliacion_clip.ps1`: abre integrador y demo CLIP zero-shot.
- `lanzar_ampliacion_voz.ps1`: abre integrador, vision y canal de voz ASR.
- `models/hand_landmarker.task`: detector de mano usado para recortar la region visual antes de CLIP.
- `multimodal_utils.py`: escritura comun de eventos JSON.
- `vision/captura_datos.py`: captura de imagenes para el dataset.
- `vision/entrenar_gestos.py`: entrenamiento local equivalente al cuaderno de Colab.
- `vision/Entrenamiento_Gestos.ipynb`: cuaderno usado para entrenar.
- `vision/dataset`: dataset de gestos personalizados.
- `vision/gesture_recognizer.task`: modelo entrenado final.
