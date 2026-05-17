#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EJERCICIO 1: agente.
Equivale a BowChatCalculadora_E1_Agente.py del ejemplo de clase.
"""

import re

from BowChatChefZeroWaste import BowChatChefZeroWaste


class BowChatChefZeroWaste_E1_Agente(BowChatChefZeroWaste):

    ingredient_aliases = {
        "tomate": "tomate",
        "tomates": "tomate",
        "toamte": "tomate",
        "lechuga": "lechuga",
        "zanahoria": "zanahoria",
        "zanahorias": "zanahoria",
        "pan": "pan",
        "queso": "queso",
        "huevo": "huevo",
        "huevos": "huevo",
        "pasta": "pasta",
        "macarrones": "pasta",
        "espaguetis": "pasta",
        "arroz": "arroz",
        "patata": "patata",
        "patatas": "patata",
        "pimiento": "pimiento",
        "pimientos": "pimiento",
        "calabacin": "calabacin",
        "cebolla": "cebolla",
        "cebollas": "cebolla",
        "ajo": "ajo",
        "pollo": "pollo",
        "atun": "atun",
        "lentejas": "lentejas",
        "garbanzos": "garbanzos",
        "tofu": "tofu",
        "leche": "leche",
        "yogur": "yogur",
        "espinacas": "espinacas",
        "champinon": "champinon",
        "champinones": "champinon",
        "setas": "setas",
        "harina": "harina",
        "aceituna": "aceituna",
        "aceitunas": "aceituna",
        "oliva": "aceituna",
        "olivas": "aceituna",
        "jamon": "jamon",
        "aceite": "aceite",
        "albahaca": "albahaca",
    }

    restriction_aliases = {
        "vegetariano": "vegetariano",
        "vegetariana": "vegetariano",
        "vegano": "vegano",
        "vegana": "vegano",
        "sin_gluten": "sin_gluten",
        "gluten": "sin_gluten",
        "celiaco": "sin_gluten",
        "sin_lactosa": "sin_lactosa",
        "lactosa": "sin_lactosa",
        "ligero": "ligero",
        "rapido": "rapido",
    }

    recipe_library = {
        "tomate": {
            "nombre": "Tostas de tomate aprovechado",
            "extras": ["pan", "ajo", "aceite"],
            "pasos": [
                "Rallar el tomate con ajo y aceite.",
                "Tostar el pan.",
                "Cubrir el pan y servir con hojas verdes si quedan.",
            ],
        },
        "pasta": {
            "nombre": "Pasta salteada de nevera",
            "extras": ["ajo", "aceite", "queso"],
            "pasos": [
                "Cocer la pasta.",
                "Saltear los restos de verduras.",
                "Mezclar todo y terminar con queso si hay.",
            ],
        },
        "arroz": {
            "nombre": "Arroz rapido de aprovechamiento",
            "extras": ["cebolla", "aceite", "huevo"],
            "pasos": [
                "Sofreir cebolla o verdura picada.",
                "Anadir arroz cocido o rapido.",
                "Terminar con huevo revuelto.",
            ],
        },
        "huevo": {
            "nombre": "Tortilla flexible de restos",
            "extras": ["patata", "cebolla", "aceite"],
            "pasos": [
                "Batir los huevos.",
                "Anadir verduras o patata en trozos.",
                "Cuajar a fuego medio.",
            ],
        },
        "default": {
            "nombre": "Bol Chef Zero Waste",
            "extras": ["aceite", "ajo", "pan"],
            "pasos": [
                "Cortar los ingredientes disponibles.",
                "Saltear primero los ingredientes duros.",
                "Servir con pan, arroz o pasta si hace falta base.",
            ],
        },
    }

    def initialization(self):
        print("Chef Zero Waste iniciado.")
        print('Escribe "instrucciones" para ver ejemplos o "salir" para terminar.')

    def agent(self, catIndex, entities):
        """
        Implementa la operacion elegida y retorna un resultado.
        """
        operation = self.categories[catIndex]
        data = entities if isinstance(entities, dict) else self._parse_entities(entities)
        data.setdefault("stm_usada", [])

        if operation == "instrucciones":
            return self._agent_instrucciones()
        if operation == "anadir_ingredientes":
            return self._agent_anadir_ingredientes(data)
        if operation == "recomendar_receta":
            return self._agent_recomendar_receta(data)
        if operation == "sustituir_ingrediente":
            return self._agent_sustituir_ingrediente(data)
        if operation == "ajustar_raciones":
            return self._agent_ajustar_raciones(data)
        if operation == "lista_compra":
            return self._agent_lista_compra()

        print("Operacion no reconocida.")
        return {"operacion": operation, "resultado": None}

    def _agent_instrucciones(self):
        print("\nINSTRUCCIONES CHEF ZERO WASTE")
        print("- instrucciones(): muestra esta ayuda.")
        print("- anadir_ingredientes(ingredientes...): tengo tomate pan y queso.")
        print("- recomendar_receta(ingrediente, tiempo, raciones): receta con tomate en 15 minutos para 2 raciones.")
        print("- sustituir_ingrediente(ingrediente, restriccion): sustituye queso sin lactosa.")
        print("- ajustar_raciones(raciones): ajusta a 4 raciones.")
        print("- lista_compra(): que tengo que comprar.")
        print("- imagen: anade una ruta .png/.jpg/.webp; el vector final es [embedding_imagen + BoW].")
        print("- salir: termina la conversacion.\n")
        return {"operacion": "instrucciones", "resultado": "instrucciones impresas"}

    def _agent_anadir_ingredientes(self, data):
        ingredients = data.get("ingredientes", [])
        if len(ingredients) == 0:
            print("No he encontrado ingredientes.")
            return {"operacion": "anadir_ingredientes", "ingredientes": []}

        self._memorize(data)
        print("Ingredientes guardados: {}".format(", ".join(ingredients)))
        self._print_stm_notes(data)
        return {"operacion": "anadir_ingredientes", "ingredientes": ingredients}

    def _agent_recomendar_receta(self, data):
        ingredient = data.get("ingrediente_principal")
        if ingredient is None:
            print("Necesito un ingrediente para recomendar una receta.")
            return {"operacion": "recomendar_receta", "receta": None}

        time = data.get("tiempo") or 25
        servings = data.get("raciones") or 2
        available = self._unique(
            self._stm_value("STMingredientes", [])
            + data.get("ingredientes", [])
            + [ingredient]
        )
        recipe = self._build_recipe(ingredient, time, servings, available)

        self._memorize(data, recipe)

        print("\nReceta propuesta: {}".format(recipe["nombre"]))
        print("Ingrediente principal: {}".format(ingredient))
        print("Tiempo: {} minutos | Raciones: {}".format(time, servings))
        print("Ingredientes sugeridos: {}".format(", ".join(recipe["ingredientes"])))
        print("Pasos:")
        for i, step in enumerate(recipe["pasos"], start=1):
            print("  {}. {}".format(i, step))
        if len(recipe["faltan"]) > 0:
            print("Faltaria comprar: {}".format(", ".join(recipe["faltan"])))
        else:
            print("No faltan ingredientes basicos.")
        self._print_stm_notes(data)
        print()

        return {"operacion": "recomendar_receta", "receta": recipe}

    def _agent_sustituir_ingrediente(self, data):
        ingredient = data.get("ingrediente")
        restriction = data.get("restriccion")
        if ingredient is None:
            print("Necesito saber que ingrediente quieres sustituir.")
            return {"operacion": "sustituir_ingrediente", "resultado": None}
        if restriction is None:
            print("Necesito una restriccion: sin_gluten, sin_lactosa, vegetariano o vegano.")
            return {"operacion": "sustituir_ingrediente", "resultado": None}

        replacement = self._replacement_for(ingredient, restriction)
        self._memorize(data)
        print("Sustitucion propuesta: {} -> {} ({})".format(ingredient, replacement, restriction))
        self._print_stm_notes(data)
        return {
            "operacion": "sustituir_ingrediente",
            "ingrediente": ingredient,
            "restriccion": restriction,
            "sustituto": replacement,
        }

    def _agent_ajustar_raciones(self, data):
        servings = data.get("raciones")
        recipe = self._stm_value("STMreceta", None)
        if recipe is None:
            print("No hay receta previa. Primero pide una recomendacion.")
            return {"operacion": "ajustar_raciones", "receta": None}
        if servings is None:
            print("No he encontrado el numero de raciones.")
            return {"operacion": "ajustar_raciones", "receta": recipe}

        old_servings = recipe.get("raciones", 2) or 2
        adjusted = dict(recipe)
        adjusted["raciones"] = servings
        adjusted["nota_ajuste"] = "Multiplicar cantidades por {:.2f}".format(servings / old_servings)
        self._memorize(data, adjusted)

        print("Receta ajustada a {} raciones.".format(servings))
        print(adjusted["nota_ajuste"])
        self._print_stm_notes(data)
        return {"operacion": "ajustar_raciones", "receta": adjusted}

    def _agent_lista_compra(self):
        recipe = self._stm_value("STMreceta", None)
        if recipe is None:
            print("No hay receta previa. Pide antes una receta.")
            return {"operacion": "lista_compra", "faltan": []}

        missing = recipe.get("faltan", [])
        if len(missing) == 0:
            print("Lista de compra vacia.")
        else:
            print("Lista de compra: {}".format(", ".join(missing)))
        return {"operacion": "lista_compra", "faltan": missing}

    def _parse_entities(self, entities):
        raw = self._strip_accents(self._last_raw_sentence.lower())
        raw = raw.replace("sin gluten", "sin_gluten")
        raw = raw.replace("sin lactosa", "sin_lactosa")
        entity_tokens = [self._clean_token(tok) for tok in entities]
        all_words = re.findall(r"[\w_]+", raw)

        ingredients = []
        for token in entity_tokens + all_words:
            ingredient = self._canon_ingredient(token)
            if ingredient is not None:
                ingredients.append(ingredient)
        ingredients = self._unique(ingredients)

        restrictions = []
        for token in entity_tokens + all_words:
            restriction = self._canon_restriction(token)
            if restriction is not None:
                restrictions.append(restriction)
        restrictions = self._unique(restrictions)

        return {
            "ingredientes": ingredients,
            "ingrediente": ingredients[0] if len(ingredients) > 0 else None,
            "ingrediente_principal": ingredients[0] if len(ingredients) > 0 else None,
            "restriccion": restrictions[0] if len(restrictions) > 0 else None,
            "tiempo": self._extract_time(raw),
            "raciones": self._extract_servings(raw),
            "ruta_imagen": self._extract_image_path(self._last_raw_sentence),
            "stm_usada": [],
        }

    def _build_recipe(self, ingredient, time, servings, available):
        profile = self.recipe_library.get(ingredient, self.recipe_library["default"])
        ingredients = self._unique([ingredient] + profile["extras"] + available)
        missing = [item for item in profile["extras"] if item not in available]
        return {
            "nombre": profile["nombre"],
            "ingrediente_principal": ingredient,
            "tiempo": time,
            "raciones": servings,
            "ingredientes": ingredients,
            "faltan": missing,
            "pasos": profile["pasos"],
        }

    def _replacement_for(self, ingredient, restriction):
        replacements = {
            "sin_gluten": {
                "pan": "pan sin gluten o tortitas de maiz",
                "pasta": "pasta de maiz o arroz",
                "default": "arroz, quinoa o patata",
            },
            "sin_lactosa": {
                "queso": "queso sin lactosa o levadura nutricional",
                "leche": "bebida vegetal",
                "yogur": "yogur vegetal",
                "default": "version vegetal o sin lactosa",
            },
            "vegetariano": {
                "pollo": "tofu salteado",
                "atun": "garbanzos machacados con limon",
                "default": "legumbres o tofu",
            },
            "vegano": {
                "huevo": "harina de garbanzo con agua",
                "queso": "levadura nutricional",
                "pollo": "tofu o garbanzos",
                "default": "legumbres, tofu o frutos secos",
            },
            "ligero": {"default": "verduras salteadas y menos aceite"},
        }
        table = replacements.get(restriction, {})
        return table.get(ingredient, table.get("default", "otro ingrediente disponible"))

    def _extract_time(self, raw):
        for pattern in [r"(\d+)\s*(min|minuto|minutos)", r"en\s+(\d+)\s*$"]:
            match = re.search(pattern, raw)
            if match:
                return int(match.group(1))
        return None

    def _extract_servings(self, raw):
        patterns = [
            r"(\d+)\s*(racion|raciones|persona|personas|plato|platos)",
            r"para\s+(\d+)",
            r"a\s+(\d+)\s*(racion|raciones|persona|personas)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                return int(match.group(1))
        return None

    def _extract_image_path(self, sentence):
        patterns = [
            r'"([^"]+\.(?:jpg|jpeg|png|bmp|webp))"',
            r"'([^']+\.(?:jpg|jpeg|png|bmp|webp))'",
            r"(?:imagen|foto|ruta)\s+([^\s]+\.(?:jpg|jpeg|png|bmp|webp))",
            r"([^\s]+\.(?:jpg|jpeg|png|bmp|webp))",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _canon_ingredient(self, token):
        return self.ingredient_aliases.get(self._clean_token(token))

    def _canon_restriction(self, token):
        return self.restriction_aliases.get(self._clean_token(token))

    def _unique(self, values):
        result = []
        for value in values:
            if value is not None and value not in result:
                result.append(value)
        return result

    def _stm_value(self, name, default):
        return getattr(self, name, default)

    def _memorize(self, data, recipe=None):
        if not hasattr(self, "STMingredientes"):
            return
        ingredients = data.get("ingredientes", [])
        if len(ingredients) > 0:
            self.STMingredientes = self._unique(self.STMingredientes + ingredients)
            self.STMingredientePrincipal = data.get("ingrediente_principal") or ingredients[0]
        if data.get("tiempo") is not None:
            self.STMtiempo = data["tiempo"]
        if data.get("raciones") is not None:
            self.STMraciones = data["raciones"]
        if data.get("restriccion") is not None:
            self.STMrestriccion = data["restriccion"]
        if recipe is not None:
            self.STMreceta = recipe

    def _print_stm_notes(self, data):
        notes = data.get("stm_usada", [])
        if len(notes) > 0:
            print("STM usada: {}".format(", ".join(notes)))
