#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer

from Chat import Chat
from RobustLinearSVC import RobustLinearSVC


class BoWChat(Chat):
    """
    Implementacion Bag of Words del chat.
    Las entidades se extraen y el resto de tokens alimentan el clasificador.
    """

    def __init__(self, categories, fileVectors, fileVoc):
        super().__init__(categories, fileVectors)
        self.Vec = CountVectorizer(token_pattern=r"\b[^ ]+\b")
        self.fileVoc = fileVoc
        self.Vec.vocabulary = self._readVoc(fileVoc)

    def multiClassLearner(self):
        return RobustLinearSVC(C=0.01, random_state=1, dual="auto")

    def vectorize(self, normSen):
        allTokens = self.Vec.build_tokenizer()(normSen)

        entities = []
        tokens = []
        for tok in allTokens:
            if self.isEntity(tok):
                entities.append(tok)
            else:
                tokens.append(tok)

        currVoc = self.Vec.vocabulary
        if currVoc is None:
            currVoc = {}

        newTok = False
        for tok in tokens:
            if tok not in currVoc:
                currVoc[tok] = len(currVoc)
                newTok = True
                if not getattr(self, "_quiet_vectorize", False):
                    print('Insertado en vocabulario "{}" -> {}'.format(tok, currVoc[tok]))

        if newTok:
            self.Vec = CountVectorizer(token_pattern=r"\b[^ ]+\b")
            self.Vec.vocabulary = currVoc
            self._writeVoc(currVoc, self.fileVoc)

        if len(currVoc) == 0:
            vector = []
        else:
            vector = self.Vec.transform([normSen])[0].toarray()[0].tolist()

        if not getattr(self, "_quiet_vectorize", False):
            print("{} -> {}".format(normSen, vector))
        return vector, entities

    def vectorToStr(self, vector):
        return " ".join([str(v) for v in vector])

    def vectorFromStr(self, vectorStr):
        return [float(v) for v in vectorStr.split(" ") if len(v) > 0]

    def _writeVoc(self, voc, fileVoc):
        with open(fileVoc, "wt", encoding="utf-8") as file:
            for token, value in voc.items():
                file.write("{} {}\n".format(token, value))

    def _readVoc(self, fileVoc):
        voc = {}
        try:
            with open(fileVoc, "rt", encoding="utf-8") as file:
                line = file.readline()
                while len(line) > 0:
                    row = line.split("\n")[0].split(" ")
                    if len(row) == 2 and row[1].isnumeric():
                        voc[row[0]] = int(row[1])
                    line = file.readline()
        except FileNotFoundError:
            return {}
        return voc

    def isEntity(self, tok):
        return False
