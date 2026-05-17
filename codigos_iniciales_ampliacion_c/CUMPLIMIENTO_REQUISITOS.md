# Cumplimiento de requisitos - Practica Final IMM

## 1. Entrenamiento de Gestos Personalizados (Canal Visual)

Estado: **cumplido**.

### Diccionario visual

Se han definido tres gestos personalizados nuevos para ChefZeroWaste, mas una clase negativa:

| Gesto | Funcion semantica en la aplicacion |
| --- | --- |
| `pizca_sal` | Indica "vamos a cocinar": iniciar una recomendacion de receta. |
| `corte_cuchillo` | Indica partir la receta en mas raciones. |
| `sustituir_ingrediente` | Gesto de cruz para marcar sustitucion de ingredientes. |
| `None` | Clase negativa para cuando no hay gesto valido. |

La clase `None` no cuenta como gesto nuevo, pero mejora la robustez del clasificador al separar acciones reales de fondos o posiciones no intencionales.

### Dataset local

El dataset esta en:

`vision/dataset`

Distribucion de muestras:

| Clase | Muestras |
| --- | ---: |
| `corte_cuchillo` | 86 |
| `pizca_sal` | 119 |
| `sustituir_ingrediente` | 125 |
| `None` | 82 |

El script usado para capturar nuevas muestras es:

`vision/captura_datos.py`

### Transfer Learning con Model Maker

El entrenamiento se realizo con MediaPipe Model Maker en:

`vision/Entrenamiento_Gestos.ipynb`

Tambien se conserva un script equivalente para entrenamiento local:

`vision/entrenar_gestos.py`

Evidencias tecnicas:

- Importa `mediapipe_model_maker.gesture_recognizer`.
- Carga el dataset con `gesture_recognizer.Dataset.from_folder(...)`.
- Divide datos en `80%` entrenamiento, `10%` validacion y `10%` test.
- Configura `gesture_recognizer.HParams(...)`.
- Entrena con `gesture_recognizer.GestureRecognizer.create(...)`.
- Exporta con `model.export_model()`.

El cuaderno muestra que se reutilizan los modelos base de MediaPipe:

- `palm_detection_full.tflite`
- `hand_landmark_full.tflite`
- `gesture_embedder`

Y se entrena una capa final personalizada para 4 clases, incluyendo `None`.

### Modelo final y prediccion unimodal

El modelo final exportado esta en:

`vision/gesture_recognizer.task`

El canal visual de ejecucion esta en:

`vision/main.py`

Ese script carga el modelo `.task`, ejecuta inferencia con webcam y genera eventos unimodales visuales en JSON:

```json
{
  "source": "vision",
  "intent": "iniciar_receta",
  "gesture": "pizca_sal",
  "timestamp": 1778945740.10,
  "confidence": 0.91
}
```

Por tanto, el canal visual produce una prediccion unimodal empaquetada con:

- intencion visual semantica,
- etiqueta cruda del gesto reconocida por el modelo,
- confianza,
- marca temporal.

Este evento es la entrada visual que posteriormente consume el integrador multimodal.

## 2. Integracion con el Modulo de PLN

Estado: **cumplido**.

### Relacion con las practicas de PLN de la asignatura

Las carpetas:

- `S1TextClassificationBoW`
- `S2TextClassificationVectorsSinGloVeVectors`
- `S3TextClassificationMLMeasures`

son practicas de PLN/clasificacion textual de la asignatura:

- S1 implementa clasificacion de texto con `Bag of Words` mediante `CountVectorizer`.
- S2 implementa representaciones vectoriales de palabras y frases.
- S3 evalua modelos de clasificacion textual con metricas de ML como F1.

Por tanto, si el profesor habla del modulo de PLN o clasificacion de intenciones, esas carpetas son la base metodologica vista en clase. En la entrega final no se usa Reuters como dominio, sino una adaptacion propia al agente ChefZeroWaste.

### Modulo PLN usado en la practica final

La integracion parte del trabajo anterior:

`Trabajo_Grupo_01_Agente_Conversacional_ChefZeroWaste`

En esta carpeta final se ha incorporado el modulo de texto en:

- `Chat.py`: bucle general frase -> vector -> categoria -> agente.
- `BoWChat.py`: vectorizacion Bag of Words con `CountVectorizer`.
- `BowChatChefZeroWaste.py`: definicion de categorias/intenciones textuales.
- `BowChatChefZeroWaste_E1_Agente.py`: logica del agente.
- `BowChatChefZeroWaste_E2_STM.py`: memoria a corto plazo.
- `CanalTextoChefZeroWaste.py`: agente final que publica eventos JSON para el integrador.

Las intenciones textuales clasificadas son:

| Intencion textual | Significado |
| --- | --- |
| `instrucciones` | Pedir ayuda o ejemplos de uso. |
| `anadir_ingredientes` | Anadir ingredientes disponibles al contexto. |
| `recomendar_receta` | Solicitar una receta a partir de ingredientes, tiempo o raciones. |
| `sustituir_ingrediente` | Pedir una sustitucion por restriccion o preferencia. |
| `ajustar_raciones` | Cambiar numero de raciones de una receta previa. |
| `lista_compra` | Pedir ingredientes faltantes. |

Cada intencion textual se guarda en:

`event_text.json`

Ejemplo:

```json
{
  "source": "text",
  "intent": "recomendar_receta",
  "timestamp": 1778945740.10,
  "raw_text": "quiero una receta con tomate para 2 personas",
  "entities": {
    "ingrediente_principal": "tomate",
    "raciones": 2
  }
}
```

### Complementariedad texto-vision

Las intenciones de texto no duplican el gesto. El texto aporta los parametros concretos de la accion: ingredientes, restricciones o numero de raciones. El gesto visual aporta la intencion pragmatica: empezar a cocinar, partir en mas raciones o marcar una sustitucion.

| Texto PLN | Gesto/modelo visual | Intencion visual enviada | Complemento aportado |
| --- | --- | --- | --- |
| `anadir_ingredientes` | `pizca_sal` | `iniciar_receta` | El texto dice con que ingredientes; el gesto indica "vamos a cocinar". |
| `recomendar_receta` | `pizca_sal` | `iniciar_receta` | El texto aporta ingrediente, tiempo o raciones; el gesto confirma inicio. |
| `ajustar_raciones` | `corte_cuchillo` | `aumentar_raciones` | El texto indica el numero; el gesto expresa partir en mas raciones. |
| `sustituir_ingrediente` | `sustituir_ingrediente` | `marcar_sustitucion` | El texto indica que ingrediente y restriccion; el gesto de cruz marca sustitucion. |

Ejemplos no redundantes:

- Texto: "tengo tomate y queso" + gesto `pizca_sal` -> recomendar una receta con esos ingredientes.
- Texto: "ajusta a 4 raciones" + gesto `corte_cuchillo` -> partir la receta en mas raciones.
- Texto: "sustituye queso sin lactosa" + gesto `sustituir_ingrediente` -> aplicar sustitucion de ingrediente.

## 3. Fusion Semantica Tardia

Estado: **cumplido**.

### Integrador en Python

El integrador esta implementado en:

`IntegradorMultimodal.py`

Recibe eventos empaquetados desde dos archivos JSON:

| Canal | Archivo | Productor |
| --- | --- | --- |
| Texto | `event_text.json` | `CanalTextoChefZeroWaste.py` |
| Vision | `event_vision.json` | `vision/main.py` |

Los dos canales usan la funcion comun:

`multimodal_utils.py -> save_multimodal_event(...)`

Formato minimo compartido:

```json
{
  "source": "text",
  "intent": "recomendar_receta",
  "timestamp": 1778945740.10
}
```

Ejemplo de evento textual:

```json
{
  "source": "text",
  "intent": "recomendar_receta",
  "timestamp": 1778945740.10,
  "raw_text": "quiero una receta con tomate",
  "entities": {
    "ingrediente_principal": "tomate"
  }
}
```

Ejemplo de evento visual:

```json
{
  "source": "vision",
  "intent": "aumentar_raciones",
  "gesture": "corte_cuchillo",
  "timestamp": 1778945741.30,
  "confidence": 0.88
}
```

El integrador lee periodicamente esos archivos con `read_event_file(...)` y solo procesa eventos nuevos comparando su `timestamp`.

### Gestion de asincronia

La clase `MultimodalIntegrator` recibe un parametro:

```python
MultimodalIntegrator(time_window_ms=3000)
```

Esto crea una ventana de memoria de 3000 ms:

```python
self.time_window = timedelta(milliseconds=time_window_ms)
```

Los eventos no tienen que ocurrir en el mismo milisegundo. El emparejamiento se hace si:

```python
abs(text_event.timestamp - visual_event.timestamp) <= self.time_window
```

Ademas, el integrador mantiene listas internas de eventos recientes:

- `self.text_events`
- `self.visual_events`

Y descarta eventos antiguos mediante `_discard_old_events()`.

### Validacion semantica

La accion final solo se ejecuta si la pareja `(intencion_textual, intencion_visual)` esta en la tabla `VALID_FUSIONS`.

Combinaciones aceptadas:

| Intencion textual | Intencion visual | Accion final |
| --- | --- | --- |
| `anadir_ingredientes` | `iniciar_receta` | Recomendar receta con los ingredientes del texto. |
| `recomendar_receta` | `iniciar_receta` | Confirmar el inicio de la receta solicitada. |
| `ajustar_raciones` | `aumentar_raciones` | Partir la receta en mas raciones. |
| `sustituir_ingrediente` | `marcar_sustitucion` | Aplicar la sustitucion textual. |

Si la combinacion llega dentro de la ventana temporal pero no esta en `VALID_FUSIONS`, se ignora:

```text
Fusion ignorada: texto='lista_compra' + gesto='iniciar_receta' no es una combinacion valida.
```

Por tanto, no basta con recibir un evento textual y uno visual: la pareja debe ser temporalmente cercana y semanticamente valida.

### Salida de fusion

Cuando hay una fusion valida, el sistema genera:

`event_fusion.json`

Ejemplo:

```json
{
  "source": "fusion",
  "action": "recomendar_receta_con_ingredientes",
  "message": "Iniciar una recomendacion de receta usando los ingredientes indicados por texto.",
  "text_intent": "anadir_ingredientes",
  "visual_intent": "iniciar_receta",
  "delta_ms": 1200.0
}
```

### Prueba sin webcam

El integrador incluye una demo:

```powershell
python IntegradorMultimodal.py --demo
```

Salida esperada:

```text
FUSION VALIDA: texto='anadir_ingredientes' + gesto='iniciar_receta' (1200.0 ms)
Fusion ignorada: texto='lista_compra' + gesto='iniciar_receta' no es una combinacion valida.
FUSION VALIDA: texto='ajustar_raciones' + gesto='aumentar_raciones' (1100.0 ms)
FUSION VALIDA: texto='sustituir_ingrediente' + gesto='marcar_sustitucion' (1000.0 ms)
```

Esta prueba demuestra:

- eventos asincronos separados por mas de 1 segundo,
- rechazo de una combinacion no valida,
- ejecucion solo de acciones finales semanticamente validas.

## Ampliacion: Embeddings Multimodales Vision + Texto

Estado: **implementado como modo opcional**.

### Modelo usado

Se ha anadido:

`clip_zero_shot_multimodal.py`

Este script usa el modelo:

`openai/clip-vit-base-patch32`

mediante `transformers`.

CLIP proyecta imagenes y texto al mismo espacio vectorial. El sistema normaliza los embeddings y calcula similitud coseno mediante producto escalar. Para evitar que el fondo, la cara o la ropa dominen el embedding visual, se usa MediaPipe Hands solo como detector de region de interes: recorta la mano y ese recorte es la imagen que se proyecta con CLIP.

El detector de mano usado por MediaPipe Tasks queda guardado en:

`models/hand_landmarker.task`

El modelo se descarga automaticamente la primera vez que se ejecuta el script y queda guardado en la cache local de Hugging Face. En este equipo se verifico la cache en:

`C:\Users\Control Lunar\.cache\huggingface\hub\models--openai--clip-vit-base-patch32`

### Gestos obligatorios conservados

Se conservan los tres gestos obligatorios entrenados:

| Gesto | Intencion visual |
| --- | --- |
| `pizca_sal` | `iniciar_receta` |
| `corte_cuchillo` | `aumentar_raciones` |
| `sustituir_ingrediente` | `marcar_sustitucion` |

### Zero-shot con comandos arbitrarios

El usuario puede escribir un comando libre como:

- `quiero empezar a cocinar con tomate pan y queso`
- `divide la receta para 4 personas`
- `cambia queso por una alternativa sin lactosa`

El sistema compara ese texto con descripciones semanticas de los tres gestos, por ejemplo:

- `gesture meaning let's cook a recipe`
- `split the recipe into more servings`
- `cross gesture for ingredient substitution`

Tambien codifica el recorte de la mano del fotograma actual y lo compara contra esas mismas descripciones. Si texto y recorte de mano coinciden en la misma intencion visual, se publican los eventos JSON para el integrador.

Como las puntuaciones visuales de CLIP en gestos de mano son cercanas, el sistema admite un margen semantico pequeno: si la intencion textual esta casi empatada con la mejor prediccion visual, se acepta la intencion compatible con el texto. Esto sigue usando similitud coseno en el espacio compartido, pero evita que una diferencia minima entre prompts bloquee una fusion correcta.

Ademas, este modo extrae entidades simples del texto libre antes de publicar el evento: ingredientes conocidos, numero de raciones y restricciones como `sin_lactosa`. Asi el integrador puede producir una receta, una escala de raciones o una sustitucion concreta tambien desde la ampliacion.

### Verificacion del modelo sin camara

El script incluye un autotest para cargar CLIP, comprobar que los embeddings funcionan y validar el emparejamiento texto -> gesto sin depender de la webcam:

```powershell
python clip_zero_shot_multimodal.py --self-test
```

Resultado obtenido:

```text
quiero empezar a cocinar con lo que tengo -> iniciar_receta | gesto=pizca_sal
divide la receta para mas personas -> aumentar_raciones | gesto=corte_cuchillo
cambia este ingrediente por una alternativa sin lactosa -> marcar_sustitucion | gesto=sustituir_ingrediente
```

### Salida hacia el integrador

Cuando CLIP detecta compatibilidad semantica, publica:

`event_text.json`

```json
{
  "source": "text",
  "intent": "anadir_ingredientes",
  "raw_text": "quiero empezar a cocinar con tomate pan y queso",
  "entities": {
    "ingredientes": ["tomate", "pan", "queso"],
    "ingrediente_principal": "tomate"
  },
  "zero_shot": true,
  "clip_text_visual_intent": "iniciar_receta",
  "clip_text_score": 0.31
}
```

`event_vision.json`

```json
{
  "source": "vision",
  "intent": "iniciar_receta",
  "gesture": "pizca_sal",
  "zero_shot": true,
  "clip_image_score": 0.28,
  "clip_direct_image_text_score": 0.22
}
```

El integrador no necesita cambiar su contrato: sigue leyendo eventos JSON con `source`, `intent` y `timestamp`.

### Ejecucion

```powershell
powershell -ExecutionPolicy Bypass -File .\lanzar_ampliacion_clip.ps1
```

Este lanzador abre:

- integrador,
- demo CLIP zero-shot.

No debe abrirse a la vez que `vision/main.py`, porque ambos usan la camara.

## Ampliacion: Voz en Tiempo Real Speech-to-Text

Estado: **implementado como modo opcional**.

### Objetivo

La ampliacion B elimina la introduccion manual de texto por teclado en el canal PLN. En lugar de escribir frases en `main_chef_zero_waste.py`, el usuario habla por el microfono. El audio se transcribe con un motor ASR local y la transcripcion se envia al mismo modulo de clasificacion textual ChefZeroWaste.

### Motor ASR usado

Se ha anadido:

`voz_asr_chef_zero_waste.py`

Este script usa:

- `sounddevice` para capturar audio real del microfono.
- `faster-whisper` con el modelo `Systran/faster-whisper-medium`.
- Deteccion simple de voz/silencio por energia RMS.

El flujo del canal de voz es:

1. Calibra el ruido ambiente durante unos segundos.
2. Detecta comienzo de voz cuando la energia supera el umbral.
3. Graba hasta encontrar silencio final.
4. Transcribe la frase completa con Whisper.
5. Envia el texto transcrito al mismo PLN de ChefZeroWaste.
6. Publica `event_text.json` con `asr: true`.

Ejemplo de evento textual generado por voz:

```json
{
  "source": "text",
  "intent": "anadir_ingredientes",
  "raw_text": "tengo tomate pan y queso",
  "entities": {
    "ingredientes": ["tomate", "pan", "queso"],
    "ingrediente_principal": "tomate"
  },
  "asr": true,
  "asr_engine": "faster-whisper",
  "asr_language": "es"
}
```

### Gestion de la asincronia real

La ampliacion de voz se ejecuta con:

`lanzar_ampliacion_voz.ps1`

Este lanzador abre tres procesos:

- `IntegradorMultimodal.py --window-ms 15000`
- `vision/main.py`
- `voz_asr_chef_zero_waste.py`

La ventana temporal se amplia a 15000 ms porque en una interaccion natural el gesto suele ocurrir antes de que la frase hablada termine y sea transcrita. El integrador conserva el gesto en memoria a corto plazo mientras el canal ASR captura, espera silencio, transcribe y publica el evento textual.

Secuencia de demostracion:

1. El usuario hace `pizca_sal`.
2. `vision/main.py` publica `event_vision.json` con `iniciar_receta`.
3. El usuario dice "tengo tomate pan y queso".
4. Whisper termina la transcripcion unos segundos despues.
5. El canal de voz publica `event_text.json` con `anadir_ingredientes`.
6. El integrador fusiona ambos eventos porque siguen dentro de la ventana temporal.

### Ejecucion

```powershell
powershell -ExecutionPolicy Bypass -File .\lanzar_ampliacion_voz.ps1
```

Comandos de prueba:

```powershell
python voz_asr_chef_zero_waste.py --list-devices
python voz_asr_chef_zero_waste.py --demo-text "tengo tomate pan y queso"
```

Frases recomendadas:

- `tengo tomate pan y queso` + gesto `pizca_sal`.
- `divide la receta para cuatro personas` + gesto `corte_cuchillo`.
- `cambia queso por una alternativa sin lactosa` + gesto de sustitucion.
