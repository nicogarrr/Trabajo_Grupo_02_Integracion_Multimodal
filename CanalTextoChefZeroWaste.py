#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np

from BowChatChefZeroWaste_E2_STM import BowChatChefZeroWaste_E2_STM
from multimodal_utils import save_multimodal_event


class CanalTextoChefZeroWaste(BowChatChefZeroWaste_E2_STM):
    """
    Canal textual de la practica final.

    Clasifica la intencion con Bag of Words y publica el evento textual
    para el integrador multimodal. Se mantienen seis ceros iniciales para
    conservar la compatibilidad con el fichero de vectores del trabajo previo.
    """

    LEGACY_VISUAL_FEATURES = 6
    INTENT_RULES = [
        ("instrucciones", [r"\b(instrucciones|ayuda|como se usa)\b"]),
        ("sustituir_ingrediente", [r"\b(sustituir|sustituye|sustitucion|cambiar|cambia)\b"]),
        ("ajustar_raciones", [r"\b(ajustar|ajusta|racion|raciones|personas|platos)\b"]),
        ("lista_compra", [r"\b(lista|comprar|compra|faltan)\b"]),
        ("recomendar_receta", [r"\b(receta|recomendar|recomienda|cocinar|cocina)\b"]),
        ("anadir_ingredientes", [r"\b(tengo|anadir|anade|aniadir|aniade|agregar|agrega)\b"]),
    ]

    def evalConfirm(self, model, vector):
        predicted, details = self.classify_intent(model, vector)
        self._last_classification_confidence = details["confidence"]
        self._last_classification_details = details

        self.presentCategory(predicted)
        print(" [confianza {:.2f}]".format(details["confidence"]), end="")
        confirmed = self.askForRealCategory(predicted)
        if confirmed != predicted:
            details["user_corrected"] = True
            details["predicted_before_user_correction"] = self.categories[predicted]
            details["confidence_before_user_correction"] = details["confidence"]
            details["confidence"] = 1.0
            self._last_classification_confidence = 1.0
            self._last_classification_details = details
        return confirmed

    def agent(self, catIndex, entities):
        intent = self.categories[catIndex]
        event_path = Path(__file__).resolve().parent / "event_text.json"
        extra_event_fields = getattr(self, "_event_extra_fields", {})
        save_multimodal_event(
            event_path,
            "text",
            intent,
            raw_text=self._last_raw_sentence,
            entities=entities,
            confidence=round(float(getattr(self, "_last_classification_confidence", 0.5)), 4),
            classification=self._compact_classification_details(),
            **extra_event_fields,
        )
        self._event_extra_fields = {}
        return super().agent(catIndex, entities)

    def vectorize(self, normSen):
        bow_vector, entities = super().vectorize(normSen)
        vector = np.concatenate(
            [
                np.zeros(self.LEGACY_VISUAL_FEATURES, dtype=float),
                np.asarray(bow_vector, dtype=float),
            ]
        )
        print("Descripcion texto {} -> {}".format(normSen, self._plain_vector(vector)))
        return vector, entities

    def _plain_vector(self, vector):
        return [float(value) for value in vector]

    def classify_intent(self, model, vector):
        """
        Devuelve (categoria, detalles) con una confianza explicita.

        La practica base solo necesitaba una categoria. Para la ampliacion C el
        integrador necesita saber si el canal textual esta seguro o si la frase
        contiene ruido/ambiguedad y debe dejarse corregir por el canal visual.
        """
        model_index, model_confidence, model_ranking = self._model_intent_with_confidence(
            model,
            vector,
        )
        rule_details = self._rule_based_intent_details()

        if rule_details is None:
            return model_index, {
                "source": "model",
                "confidence": round(model_confidence, 4),
                "model_intent": self.categories[model_index],
                "model_ranking": model_ranking,
                "reason": "sin regla textual clara",
            }

        predicted = rule_details["predicted_index"]
        details = {
            "source": "rules",
            "confidence": rule_details["confidence"],
            "model_intent": self.categories[model_index],
            "model_confidence": round(model_confidence, 4),
            "model_ranking": model_ranking,
            "matched_rules": rule_details["matched_rules"],
            "reason": rule_details["reason"],
        }

        if len(rule_details["matched_rules"]) == 1 and predicted == model_index:
            details["confidence"] = max(details["confidence"], min(0.96, model_confidence))
            details["source"] = "rules+model"
            details["reason"] = "regla textual y clasificador coinciden"
        elif len(rule_details["matched_rules"]) == 1:
            details["confidence"] = min(details["confidence"], 0.72)
            details["reason"] = "regla textual clara, pero el clasificador discrepa"

        details["confidence"] = round(float(details["confidence"]), 4)
        return predicted, details

    def _model_intent_with_confidence(self, model, vector):
        predicted = int(model.predict([vector])[0])

        if getattr(model, "constantModel", None) is not None:
            return predicted, 0.55, [
                {"intent": self.categories[predicted], "score": 1.0}
            ]

        try:
            aligned_vector = self._align_vector_to_model(model, vector)
            scores = np.asarray(model.decision_function([aligned_vector]))[0]
            classes = [int(value) for value in getattr(model, "classes_", [])]
        except Exception:
            return predicted, 0.5, [
                {"intent": self.categories[predicted], "score": 0.5}
            ]

        if scores.ndim == 0:
            scores = np.asarray([float(scores)])

        if len(classes) == 2 and len(scores) == 1:
            raw = float(scores[0])
            scores = np.asarray([-raw, raw])

        if not classes or len(classes) != len(scores):
            classes = list(range(len(scores)))

        probabilities = self._softmax(scores)
        ranking = sorted(
            [
                {
                    "intent": self.categories[class_index],
                    "score": round(float(probabilities[position]), 4),
                }
                for position, class_index in enumerate(classes)
                if 0 <= class_index < len(self.categories)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        confidence = next(
            (item["score"] for item in ranking if item["intent"] == self.categories[predicted]),
            float(np.max(probabilities)) if len(probabilities) else 0.5,
        )
        return predicted, float(confidence), ranking[:3]

    def _align_vector_to_model(self, model, vector):
        row = list(vector)
        fit_n_att = getattr(model, "fitNAtt", len(row))
        if len(row) < fit_n_att:
            return row + [0] * (fit_n_att - len(row))
        return row[:fit_n_att]

    def _softmax(self, scores):
        scores = np.asarray(scores, dtype=float)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        total = np.sum(exp_scores)
        if total <= 0:
            return np.ones_like(scores) / len(scores)
        return exp_scores / total

    def _rule_based_intent(self):
        details = self._rule_based_intent_details()
        if details is None:
            return None
        return details["predicted_index"]

    def _rule_based_intent_details(self):
        text = self._strip_accents(self._last_raw_sentence.lower())
        text = text.replace("sin gluten", "sin_gluten")
        text = text.replace("sin lactosa", "sin_lactosa")

        matched_rules = []
        for category, patterns in self.INTENT_RULES:
            matched_patterns = [pattern for pattern in patterns if re.search(pattern, text)]
            if matched_patterns:
                matched_rules.append(
                    {
                        "intent": category,
                        "patterns": matched_patterns,
                    }
                )

        if not matched_rules:
            return None

        predicted_category = matched_rules[0]["intent"]
        confidence = 0.92
        reason = "regla textual unica"

        if len(matched_rules) > 1:
            confidence = 0.34
            reason = "frase ambigua: varias intenciones textuales activadas"

        return {
            "predicted_index": self.categories.index(predicted_category),
            "confidence": confidence,
            "matched_rules": matched_rules,
            "reason": reason,
        }

    def _compact_classification_details(self):
        details = getattr(self, "_last_classification_details", None)
        if not details:
            return {"source": "unknown", "confidence": 0.5}

        compact = {
            "source": details.get("source"),
            "reason": details.get("reason"),
            "model_intent": details.get("model_intent"),
            "model_confidence": details.get("model_confidence"),
            "matched_intents": [
                item["intent"] for item in details.get("matched_rules", [])
            ],
        }
        if details.get("user_corrected"):
            compact["user_corrected"] = True
            compact["predicted_before_user_correction"] = details.get(
                "predicted_before_user_correction"
            )
        return compact
