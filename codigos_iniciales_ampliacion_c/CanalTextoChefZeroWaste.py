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

    def evalConfirm(self, model, vector):
        predicted = self._rule_based_intent()
        if predicted is None:
            predicted = int(model.predict([vector])[0])

        self.presentCategory(predicted)
        return self.askForRealCategory(predicted)

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

    def _rule_based_intent(self):
        text = self._strip_accents(self._last_raw_sentence.lower())
        text = text.replace("sin gluten", "sin_gluten")
        text = text.replace("sin lactosa", "sin_lactosa")

        rules = [
            (r"\b(instrucciones|ayuda|como se usa)\b", "instrucciones"),
            (r"\b(sustituir|sustituye|sustitucion|cambiar|cambia)\b", "sustituir_ingrediente"),
            (r"\b(ajustar|ajusta|racion|raciones|personas|platos)\b", "ajustar_raciones"),
            (r"\b(lista|comprar|compra|faltan)\b", "lista_compra"),
            (r"\b(receta|recomendar|recomienda|cocinar|cocina)\b", "recomendar_receta"),
            (r"\b(tengo|anadir|anade|aniadir|aniade|agregar|agrega)\b", "anadir_ingredientes"),
        ]
        for pattern, category in rules:
            if re.search(pattern, text):
                return self.categories.index(category)
        return None
