#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Chat:
    """
    Clase abstracta para un chat incremental.
    Sigue la estructura de la practica de clase:
    frase -> vector -> categoria -> STM -> agente.
    """

    def __init__(self, categories, fileVectors):
        self.categories = categories
        self.fileVectors = fileVectors

    def run(self):
        self._createFileVectors()
        self.initialization()
        exit_chat = False
        result = None

        while not exit_chat:
            model = self.getModelFromFile()
            sentence = self.prompt()

            if self.isExit(sentence):
                exit_chat = True
                continue

            normSen = self.normalize(sentence)
            vector, entities = self.vectorize(normSen)
            catIndex = self.evalConfirm(model, vector)
            entities = self.STMEntities(entities, catIndex, result)
            self.updateFileVectors(catIndex, vector)
            result = self.agent(catIndex, entities)

    def getModelFromFile(self):
        X = []
        Y = []

        with open(self.fileVectors, "rt", encoding="utf-8") as file:
            line = file.readline()
            while len(line) > 0:
                line = line.split("\n")[0]
                if "|" in line:
                    cat, vectorStr = line.split("|", 1)
                    X.append(self.vectorFromStr(vectorStr))
                    Y.append(int(cat))
                line = file.readline()

        model = self.multiClassLearner()
        model.fit(X, Y)
        return model

    def evalConfirm(self, model, vector):
        predicted = model.predict([vector])[0]
        self.presentCategory(predicted)
        return self.askForRealCategory(predicted)

    def updateFileVectors(self, catIndex, vector):
        with open(self.fileVectors, "at", encoding="utf-8") as file:
            file.write("{}|{}\n".format(catIndex, self.vectorToStr(vector)))

    def _createFileVectors(self):
        try:
            with open(self.fileVectors, "rt", encoding="utf-8"):
                pass
        except FileNotFoundError:
            open(self.fileVectors, "wt", encoding="utf-8").close()
            print('Fichero "{}" creado'.format(self.fileVectors))

    def initialization(self):
        print('Bienvenido al chat. Escribe "salir" para terminar.')

    def prompt(self):
        return input("> ")

    def isExit(self, sentence):
        return sentence.strip().lower() in {"exit", "salir", "quit", "fin"}

    def normalize(self, sentence):
        return sentence

    def presentCategory(self, predicted):
        print("Categoria detectada: {}".format(self.categories[predicted]), end="")

    def askForRealCategory(self, predicted):
        print(". Es correcta? (Si por defecto / No): ", end="")
        yes_no = input().strip().lower()
        if len(yes_no) == 0 or yes_no[0] != "n":
            return predicted

        while True:
            for icat, category in enumerate(self.categories):
                print("{} : {}".format(icat, category))
            value = input("Introduce el numero de la categoria correcta: ").strip()
            if value.isnumeric():
                real = int(value)
                if 0 <= real < len(self.categories):
                    return real
            print("Respuesta incorrecta.")

    def agent(self, catIndex, entities):
        print("Agente: Accion {} Datos {}".format(self.categories[catIndex], entities))
        return {"categoria": self.categories[catIndex], "datos": entities}

    def STMEntities(self, entities, catIndex, prevResult):
        return entities

    def multiClassLearner(self):
        raise NotImplementedError

    def vectorize(self, normSen):
        raise NotImplementedError

    def vectorToStr(self, vector):
        raise NotImplementedError

    def vectorFromStr(self, vectorStr):
        raise NotImplementedError
