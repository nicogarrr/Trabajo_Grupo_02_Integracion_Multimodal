#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import LinearSVC


class RobustLinearSVC(LinearSVC):
    """
    LinearSVC robusto para el chat incremental:
    - Puede entrenar con 0 ejemplos.
    - Puede entrenar con una sola categoria.
    - Acepta vectores de distinta longitud, rellenando con ceros.
    """

    def __init__(
        self,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual="auto",
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super().__init__(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.constantModel = None
        self.fitNAtt = 0

    def fit(self, X, y, sample_weight=None):
        if len(X) == 0:
            self.constantModel = 0
            self.fitNAtt = 0
            return self

        classes = np.unique(y)
        self.fitNAtt = max([len(row) for row in X]) if len(X) > 0 else 0

        if len(classes) == 1 or self.fitNAtt == 0:
            values, counts = np.unique(y, return_counts=True)
            self.constantModel = values[np.argmax(counts)]
            return self

        self.constantModel = None
        Xe = []
        for x in X:
            row = list(x)
            Xe.append(row + [0] * (self.fitNAtt - len(row)))
        return super().fit(Xe, y, sample_weight)

    def predict(self, X):
        if self.constantModel is not None:
            return np.array([self.constantModel] * len(X))

        Xe = []
        for x in X:
            row = list(x)
            if len(row) < self.fitNAtt:
                row = row + [0] * (self.fitNAtt - len(row))
            else:
                row = row[: self.fitNAtt]
            Xe.append(row)
        return super().predict(Xe)
