#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat base del trabajo de grupo.
Equivale a BowChatCalculadora.py del ejemplo de clase:
solo define operadores/categorias, normalizacion y entidades.
"""

import re
import unicodedata

from BoWChat import BoWChat


class BowChatChefZeroWaste(BoWChat):
    """
    Agente conversacional de cocina de aprovechamiento.

    Operadores y aridades:
        instrucciones()                              -> aridad 0
        anadir_ingredientes(ingredientes...)         -> aridad variable
        recomendar_receta(ingrediente, tiempo, raciones) -> aridad 3
        sustituir_ingrediente(ingrediente, restriccion)  -> aridad 2
        ajustar_raciones(raciones)                  -> aridad 1
        lista_compra()                              -> aridad 0
    """

    ingredientes = {
        "tomate",
        "toamte",
        "lechuga",
        "zanahoria",
        "pan",
        "queso",
        "huevo",
        "pasta",
        "arroz",
        "patata",
        "pimiento",
        "calabacin",
        "cebolla",
        "ajo",
        "pollo",
        "atun",
        "lentejas",
        "garbanzos",
        "tofu",
        "leche",
        "yogur",
        "espinacas",
        "champinones",
        "setas",
        "harina",
        "aceituna",
        "jamon",
        "aceite",
        "albahaca",
    }

    restricciones = {
        "vegetariano",
        "vegetariana",
        "vegano",
        "vegana",
        "sin_gluten",
        "gluten",
        "celiaco",
        "sin_lactosa",
        "lactosa",
        "ligero",
        "rapido",
    }

    def __init__(self, fileVectors, fileVoc):
        categories = [
            "instrucciones",
            "anadir_ingredientes",
            "recomendar_receta",
            "sustituir_ingrediente",
            "ajustar_raciones",
            "lista_compra",
        ]
        BoWChat.__init__(self, categories, fileVectors, fileVoc)
        self._last_raw_sentence = ""

    def normalize(self, sentence):
        self._last_raw_sentence = sentence
        text = sentence.lower()
        image_paths = {}

        def keep_image_path(match):
            key = "__imagen{}__".format(len(image_paths))
            image_paths[key] = match.group(0)
            return " {} ".format(key)

        text = re.sub(
            r"[^\s]+\.(?:jpg|jpeg|png|bmp|webp)",
            keep_image_path,
            text,
            flags=re.IGNORECASE,
        )
        text = self._strip_accents(text)
        text = text.replace("sin gluten", "sin_gluten")
        text = text.replace("sin lactosa", "sin_lactosa")
        text = re.sub(r"\b(sustituir|sustituye|sustitucion)\b", "sustituye", text)
        text = re.sub(r"\b(cambiar|cambia|cambio)\b", "cambia", text)
        text = re.sub(r"\b(anadir|anade|aniadir|aniade|agregar|agrega)\b", "anade", text)
        text = re.sub(r"\b(recomendar|recomienda)\b", "recomienda", text)
        text = re.sub(r"\b(ajustar|ajusta)\b", "ajusta", text)
        text = re.sub(r"[,;:!?()\[\]{}]", " ", text)
        for key, image_path in image_paths.items():
            text = text.replace(key, image_path)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def isEntity(self, tok):
        """
        Como en la calculadora, las entidades no entran en la BoW.
        Aqui son ingredientes, restricciones, numeros y rutas de imagen.
        """
        token = self._clean_token(tok)
        return (
            token in self.ingredientes
            or token in self.restricciones
            or re.fullmatch(r"\d+", token) is not None
            or re.fullmatch(r"\d+(min|minutos|raciones|personas)", token) is not None
            or re.search(r"\.(jpg|jpeg|png|bmp|webp)$", token) is not None
        )

    def _clean_token(self, token):
        token = self._strip_accents(str(token).lower())
        return token.strip(".,;:!?()[]{}\"'")

    def _strip_accents(self, text):
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
