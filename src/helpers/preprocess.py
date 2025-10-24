import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import CountEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class S21Pipelines:
    def __init__(self, Xs):
        self.X_train, self.X_test = Xs[0], Xs[1]


    def _preprocess_builder(self):
        X_train = self.X_train

        cat_cols = X_train.select_dtypes(include=["object","category"]).columns
        num_cols = X_train.select_dtypes(include=["number"]).columns

        num_block = Pipeline([
            ("impute", SimpleImputer(strategy="median")), # for NaN
            ("sc",     StandardScaler())
        ])

        cat_block = Pipeline([
            ("cnt", CountEncoder(cols=cat_cols, handle_unknown=0, handle_missing=0, normalize=True))
        ])

        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", num_block, num_cols),
                ("cat", cat_block, cat_cols),
            ]
        )

        return self