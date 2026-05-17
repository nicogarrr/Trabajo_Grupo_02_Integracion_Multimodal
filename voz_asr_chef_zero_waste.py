#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from CanalTextoChefZeroWaste import CanalTextoChefZeroWaste


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datos"
DEFAULT_MODEL = "Systran/faster-whisper-medium"


class VoiceTextProcessor:
    def __init__(self, update_training=False):
        self.chat = CanalTextoChefZeroWaste(
            fileVectors=str(DATA_DIR / "ChefZeroWaste.vec"),
            fileVoc=str(DATA_DIR / "ChefZeroWaste.voc"),
        )
        self.chat._createFileVectors()
        self.previous_result = None
        self.update_training = update_training

    def process_sentence(self, sentence, asr_metadata=None):
        sentence = sentence.strip()
        if not sentence:
            return None

        model = self.chat.getModelFromFile()
        norm_sentence = self.chat.normalize(sentence)
        vector, entities = self.chat.vectorize(norm_sentence)
        predicted, details = self.chat.classify_intent(model, vector)
        self.chat._last_classification_confidence = details["confidence"]
        self.chat._last_classification_details = details

        print(
            "Categoria detectada por voz: {} [confianza {:.2f}]".format(
                self.chat.categories[predicted],
                details["confidence"],
            )
        )
        entities = self.chat.STMEntities(entities, predicted, self.previous_result)
        if self.update_training:
            self.chat.updateFileVectors(predicted, vector)

        self.chat._event_extra_fields = {
            "asr": True,
            "asr_engine": "faster-whisper",
            **(asr_metadata or {}),
        }
        self.previous_result = self.chat.agent(predicted, entities)
        return self.previous_result


class EnergyVoiceRecorder:
    def __init__(
        self,
        sample_rate=16000,
        block_seconds=0.2,
        calibration_seconds=1.5,
        min_phrase_seconds=0.8,
        max_phrase_seconds=8.0,
        end_silence_seconds=1.1,
        start_factor=3.0,
        stop_factor=1.6,
        min_start_threshold=0.010,
        min_stop_threshold=0.006,
        device=None,
    ):
        self.sample_rate = sample_rate
        self.block_seconds = block_seconds
        self.block_size = int(sample_rate * block_seconds)
        self.calibration_seconds = calibration_seconds
        self.min_phrase_seconds = min_phrase_seconds
        self.max_phrase_seconds = max_phrase_seconds
        self.end_silence_seconds = end_silence_seconds
        self.start_factor = start_factor
        self.stop_factor = stop_factor
        self.min_start_threshold = min_start_threshold
        self.min_stop_threshold = min_stop_threshold
        self.device = device
        self.audio_queue = queue.Queue()
        self.start_threshold = min_start_threshold
        self.stop_threshold = min_stop_threshold

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata[:, 0].copy())

    def open_stream(self):
        return sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self.callback,
        )

    def calibrate(self):
        print(f"Calibrando ruido ambiente durante {self.calibration_seconds:.1f} s...")
        chunks = []
        deadline = time.time() + self.calibration_seconds
        while time.time() < deadline:
            chunks.append(self.audio_queue.get())

        if chunks:
            audio = np.concatenate(chunks)
            noise_rms = rms(audio)
        else:
            noise_rms = 0.0

        self.start_threshold = max(self.min_start_threshold, noise_rms * self.start_factor)
        self.stop_threshold = max(self.min_stop_threshold, noise_rms * self.stop_factor)
        print(
            "Umbrales voz: inicio={:.4f}, silencio={:.4f}, ruido={:.4f}".format(
                self.start_threshold,
                self.stop_threshold,
                noise_rms,
            )
        )

    def listen_phrase(self):
        recording = False
        chunks = []
        silence_seconds = 0.0
        duration_seconds = 0.0

        while True:
            block = self.audio_queue.get()
            block_rms = rms(block)

            if not recording:
                if block_rms >= self.start_threshold:
                    recording = True
                    chunks = [block]
                    silence_seconds = 0.0
                    duration_seconds = self.block_seconds
                    print("Voz detectada. Habla ahora...")
                continue

            chunks.append(block)
            duration_seconds += self.block_seconds

            if block_rms < self.stop_threshold:
                silence_seconds += self.block_seconds
            else:
                silence_seconds = 0.0

            if (
                duration_seconds >= self.min_phrase_seconds
                and silence_seconds >= self.end_silence_seconds
            ):
                audio = np.concatenate(chunks)
                print(f"Fin de frase detectado ({duration_seconds:.1f} s). Transcribiendo...")
                return audio

            if duration_seconds >= self.max_phrase_seconds:
                audio = np.concatenate(chunks)
                print(f"Duracion maxima alcanzada ({duration_seconds:.1f} s). Transcribiendo...")
                return audio


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def transcribe_audio(model, audio, language):
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    metadata = {
        "asr_language": getattr(info, "language", language),
        "asr_language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
    }
    return text, metadata


def run_demo_text(args):
    processor = VoiceTextProcessor(update_training=args.update_training)
    processor.process_sentence(
        args.demo_text,
        {
            "asr_demo": True,
            "asr_language": args.language,
            "asr_language_probability": 1.0,
        },
    )


def run_microphone(args):
    print("ChefZeroWaste ASR iniciado.")
    print("No escribas texto: habla por el microfono.")
    print("Haz primero el gesto visual y despues di la frase.")
    print("Pulsa Ctrl+C para salir.")
    print("Modelo ASR:", args.model)

    whisper = load_whisper_model(args.model, args.device, args.compute_type)
    processor = VoiceTextProcessor(update_training=args.update_training)
    recorder = EnergyVoiceRecorder(
        sample_rate=args.sample_rate,
        block_seconds=args.block_seconds,
        calibration_seconds=args.calibration_seconds,
        min_phrase_seconds=args.min_phrase_seconds,
        max_phrase_seconds=args.max_phrase_seconds,
        end_silence_seconds=args.end_silence_seconds,
        start_factor=args.start_factor,
        stop_factor=args.stop_factor,
        min_start_threshold=args.min_start_threshold,
        min_stop_threshold=args.min_stop_threshold,
        device=args.input_device,
    )

    with recorder.open_stream():
        recorder.calibrate()
        while True:
            audio = recorder.listen_phrase()
            text, metadata = transcribe_audio(whisper, audio, args.language)
            if not text:
                print("No se reconocio texto claro.")
                continue

            print(f"ASR -> {text}")
            processor.process_sentence(text, metadata)


def load_whisper_model(model_name, device, compute_type):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def main():
    parser = argparse.ArgumentParser(description="Canal de voz ASR para ChefZeroWaste.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="es")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-seconds", type=float, default=0.2)
    parser.add_argument("--calibration-seconds", type=float, default=1.5)
    parser.add_argument("--min-phrase-seconds", type=float, default=0.8)
    parser.add_argument("--max-phrase-seconds", type=float, default=8.0)
    parser.add_argument("--end-silence-seconds", type=float, default=1.1)
    parser.add_argument("--start-factor", type=float, default=3.0)
    parser.add_argument("--stop-factor", type=float, default=1.6)
    parser.add_argument("--min-start-threshold", type=float, default=0.010)
    parser.add_argument("--min-stop-threshold", type=float, default=0.006)
    parser.add_argument("--update-training", action="store_true")
    parser.add_argument("--demo-text", default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.demo_text is not None:
        run_demo_text(args)
        return

    try:
        run_microphone(args)
    except KeyboardInterrupt:
        print("\nCanal de voz detenido.")


if __name__ == "__main__":
    main()
