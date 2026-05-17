#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EJERCICIO 2: Short Term Memory.
Equivale a BowChatCalculadora_E2_STM.py del ejemplo de clase.
"""

from BowChatChefZeroWaste_E1_Agente import BowChatChefZeroWaste_E1_Agente


class BowChatChefZeroWaste_E2_STM(BowChatChefZeroWaste_E1_Agente):
    def __init__(self, fileVectors, fileVoc):
        BowChatChefZeroWaste_E1_Agente.__init__(self, fileVectors, fileVoc)

        self.STMingredientes = []
        self.STMingredientePrincipal = None
        self.STMtiempo = None
        self.STMraciones = None
        self.STMrestriccion = None
        self.STMreceta = None

    def STMEntities(self, entities, catIndex, prevResult):
        """
        Reglas STM para todos los operadores con parametros.

        anadir_ingredientes(ingredientes...)
            Si no hay ingredientes, usa los ingredientes anteriores.

        recomendar_receta(ingrediente, tiempo, raciones)
            Si falta ingrediente, tiempo o raciones, usa la STM.

        sustituir_ingrediente(ingrediente, restriccion)
            Si falta ingrediente o restriccion, usa la STM.

        ajustar_raciones(raciones)
            Si faltan raciones, usa las raciones anteriores.
            La receta previa tambien se recupera de STM.
        """
        operation = self.categories[catIndex]
        data = entities if isinstance(entities, dict) else self._parse_entities(entities)
        data["operacion"] = operation
        data.setdefault("stm_usada", [])

        if isinstance(prevResult, dict) and prevResult.get("receta") is not None:
            self.STMreceta = prevResult["receta"]

        if operation == "anadir_ingredientes":
            if len(data["ingredientes"]) == 0 and len(self.STMingredientes) > 0:
                data["ingredientes"] = list(self.STMingredientes)
                data["ingrediente"] = data["ingredientes"][0]
                data["ingrediente_principal"] = data["ingredientes"][0]
                data["stm_usada"].append("ingredientes anteriores")

        elif operation == "recomendar_receta":
            if data["ingrediente_principal"] is None:
                data["ingrediente_principal"] = self.STMingredientePrincipal
                data["ingrediente"] = self.STMingredientePrincipal
                if data["ingrediente_principal"] is not None:
                    data["stm_usada"].append("ingrediente anterior")

            if data["ingrediente_principal"] is None and len(self.STMingredientes) > 0:
                data["ingrediente_principal"] = self.STMingredientes[0]
                data["ingrediente"] = self.STMingredientes[0]
                data["stm_usada"].append("primer ingrediente guardado")

            if data["tiempo"] is None and self.STMtiempo is not None:
                data["tiempo"] = self.STMtiempo
                data["stm_usada"].append("tiempo anterior")

            if data["raciones"] is None and self.STMraciones is not None:
                data["raciones"] = self.STMraciones
                data["stm_usada"].append("raciones anteriores")

        elif operation == "sustituir_ingrediente":
            if data["ingrediente"] is None:
                data["ingrediente"] = self.STMingredientePrincipal
                data["ingrediente_principal"] = self.STMingredientePrincipal
                if data["ingrediente"] is not None:
                    data["stm_usada"].append("ingrediente anterior")

            if data["restriccion"] is None and self.STMrestriccion is not None:
                data["restriccion"] = self.STMrestriccion
                data["stm_usada"].append("restriccion anterior")

        elif operation == "ajustar_raciones":
            if data["raciones"] is None and self.STMraciones is not None:
                data["raciones"] = self.STMraciones
                data["stm_usada"].append("raciones anteriores")

        self._memorize(data)
        return data
