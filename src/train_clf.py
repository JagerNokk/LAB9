"""Этап 4 сквозного проекта: классификатор «квартира сдастся за 14 дней».

Запуск: python src/train_clf.py
Параметры читаются из params_clf.json, метрики пишутся в metrics_clf.json,
модель сохраняется в models/model_clf.joblib (в Git не хранится).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / "params_clf.json").read_text(encoding="utf-8"))
df = pd.read_csv(ROOT / "data" / "astana_rent_class.csv")

# Цель: rented_14 (1 - сдана за 14 дней). id и date в признаки не берём:
# номер объявления и дата не описывают квартиру (id - потенциальная утечка).
num_cols = ["rooms", "area", "floor", "floors", "year", "dist_center", "price", "photos"]
cat_cols = ["district", "furniture"]
X, y = df[num_cols + cat_cols], df["rented_14"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y)


def make_prep():
    return make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), cat_cols))


def make_classifier(name):
    cw = params["class_weight"]
    if name == "logreg":
        return LogisticRegression(C=params["C"], class_weight=cw)
    if name == "tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], class_weight=cw,
                                      random_state=params["random_state"])
    raise ValueError(f"неизвестная модель: {name}")


def cv_scores(model, threshold):
    """cv_f1 и cv_roc_auc на 5 стратифицированных фолдах обучающей выборки.

    F1 считается при пороге из params_clf.json, поэтому изменение порога
    видно в cv_f1 (cross_val_score с scoring='f1' всегда берёт порог 0.5).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params["random_state"])
    f1s, aucs = [], []
    for tr, va in skf.split(X_train, y_train):
        m = make_pipeline(make_prep(), make_classifier(params["model"])) if model == "pipeline" else model
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        proba = m.predict_proba(X_train.iloc[va])[:, 1]
        f1s.append(f1_score(y_train.iloc[va], (proba >= threshold).astype(int)))
        aucs.append(roc_auc_score(y_train.iloc[va], proba))
    return float(np.mean(f1s)), float(np.mean(aucs))


# Базовая константная модель: нижняя планка
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# Основная модель: предобработка внутри Pipeline, поэтому утечки из CV нет
cv_f1, cv_auc = cv_scores("pipeline", params["threshold"])
model = make_pipeline(make_prep(), make_classifier(params["model"]))
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= params["threshold"]).astype(int)

metrics = {
    "cv_f1": round(cv_f1, 4),
    "cv_roc_auc": round(cv_auc, 4),
    "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    "test_recall": round(float(recall_score(y_test, pred)), 4),
    "test_f1": round(float(f1_score(y_test, pred)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "dummy_test_accuracy": round(float(accuracy_score(y_test, dummy_pred)), 4),
}

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "model_clf.joblib")
(ROOT / "metrics_clf.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(params["model"], metrics)
