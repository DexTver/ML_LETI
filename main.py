# === ЛР2: Классификация на данных ЛР1 ===
# Требования: scikit-learn >= 1.2, pandas, numpy, matplotlib, seaborn, imbalanced-learn (опционально)
# Установки (если нужно): pip install scikit-learn imbalanced-learn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

# Опционально для борьбы с дисбалансом (если потребуется)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === 1) Загрузка данных (из ЛР1) ===
# Файл: data/amazon_products_sales_data_cleaned.csv (см. ЛР1)
DATA_PATH = "data/amazon_products_sales_data_cleaned.csv"
assert os.path.exists(DATA_PATH), f"Файл не найден: {DATA_PATH}"

data = pd.read_csv(DATA_PATH)

# Быстрая проверка структуры
print("Shape:", data.shape)
print("Columns:", list(data.columns))
print(data.head(3))

# === 2) Определение целевой и признаков ===
# По ЛР1: целевая переменная demand (классы: low/middle/high).
TARGET_COL = "demand"
assert TARGET_COL in data.columns, f"В данных нет колонки {TARGET_COL}"

# Базовый набор колонок по ЛР1 (скорректируйте, если названия отличаются):
# Числовые: reputation, totalprice, listedprice
# Категориальные: iscouponed (категории: 'no','10','10','25','25'), issponsored (0/1), buyboxavailability (0/1)
numeric_cols = []
for col in ["reputation", "totalprice", "listedprice"]:
    if col in data.columns:
        numeric_cols.append(col)

categorical_cols = []
for col in ["iscouponed", "issponsored", "buyboxavailability"]:
    if col in data.columns:
        categorical_cols.append(col)

# Защита от отсутствующих колонок
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# Фильтруем только нужные колонки + цель
used_cols = numeric_cols + categorical_cols + [TARGET_COL]
df = data[used_cols].copy()

# Удалим строки с пропусками в целевой
df = df[~df[TARGET_COL].isna()].copy()

# Убедимся, что целевая строковая
df[TARGET_COL] = df[TARGET_COL].astype(str)

# Стратифицированное разбиение train/val/test: 70/15/15
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE
)  # 0.1765*0.85 ~= 0.15, итого 70/15/15

print("Train size:", X_train.shape, "Val size:", X_val.shape, "Test size:", X_test.shape)
print("Train class dist:\n", y_train.value_counts(normalize=True))

# === 3) Препроцессинг через ColumnTransformer ===
# Масштабирование числовых важно для KNN/SVM; категориальные — One-Hot
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True))
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop"
)

# Если сильный дисбаланс, можно включить SMOTE внутри imbalanced pipeline
USE_SMOTE = False  # при необходимости переключите в True
if USE_SMOTE and IMB_AVAILABLE:
    BasePipeline = ImbPipeline
    sampler_step = [("smote", SMOTE(random_state=RANDOM_STATE))]
else:
    BasePipeline = Pipeline
    sampler_step = []

# === 4) Определяем модели и сетки гиперпараметров ===

# KNN
knn_clf = KNeighborsClassifier()
knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"]
}

knn_pipe = BasePipeline(steps=[("preprocess", preprocess)] + sampler_step + [("model", knn_clf)])

# SVM (SVC) — используем probability=True для ROC-AUC
svm_clf = SVC(probability=True, random_state=RANDOM_STATE)
svm_param_grid = [
    {
        "model__kernel": ["linear"],
        "model__C": np.logspace(-2, 2, 7)
    },
    {
        "model__kernel": ["rbf"],
        "model__C": np.logspace(-2, 2, 7),
        "model__gamma": np.logspace(-3, 1, 7)
    }
]
svm_pipe = BasePipeline(steps=[("preprocess", preprocess)] + sampler_step + [("model", svm_clf)])

# Decision Tree
tree_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_param_grid = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "model__max_depth": [None, 3, 5, 7, 10, 15, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
}
tree_pipe = BasePipeline(steps=[("preprocess", preprocess)] + sampler_step + [("model", tree_clf)])

# Random Forest
rf_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_param_grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", "log2", None]
}
rf_pipe = BasePipeline(steps=[("preprocess", preprocess)] + sampler_step + [("model", rf_clf)])

# === 5) Единый протокол валидации (StratifiedKFold) и метрика ===
# Для многоклассового ROC-AUC в GridSearchCV используем scoring="roc_auc_ovr_weighted" или "f1_macro"
# Рекомендуется основная метрика f1_macro (устойчивее к дисбалансу); AUC посчитаем отдельно.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = "f1_macro"

def run_grid_search(name, pipe, param_grid):
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"\n=== {name}: Best params ===")
    print(gs.best_params_)
    print(f"{name}: Best CV {SCORING} = {gs.best_score_:.4f}")
    return gs

grids = {}
grids["KNN"] = run_grid_search("KNN", knn_pipe, knn_param_grid)
grids["SVM"] = run_grid_search("SVM", svm_pipe, svm_param_grid)
grids["TREE"] = run_grid_search("DecisionTree", tree_pipe, tree_param_grid)
grids["RF"] = run_grid_search("RandomForest", rf_pipe, rf_param_grid)

# === 6) Оценка на валидации и тесте, сравнение моделей ===
def evaluate_model(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te, labels_order=None):
    # Предсказания
    y_tr_pred = model.predict(X_tr)
    y_v_pred  = model.predict(X_v)
    y_te_pred = model.predict(X_te)

    # Пробабилистические предсказания для ROC-AUC
    # Если predict_proba недоступен (например, SVC с probability=False), используем decision_function
    def get_proba(m, X_data):
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X_data)
        if hasattr(m, "decision_function"):
            dec = m.decision_function(X_data)
            # Привести к вероятностям через softmax для многокласса
            if dec.ndim == 1:
                # бинарный случай — приведем к двум столбцам
                dec = np.vstack([-dec, dec]).T
            expd = np.exp(dec - dec.max(axis=1, keepdims=True))
            return expd / expd.sum(axis=1, keepdims=True)
        # если ничего нет:
        # как fallback — one-hot у предсказаний (не идеально для AUC, но позволит считать macro AUC)
        preds = m.predict(X_data)
        classes_ = m.classes_
        proba = np.zeros((len(preds), len(classes_)), dtype=float)
        for i, p in enumerate(preds):
            proba[i, list(classes_).index(p)] = 1.0
        return proba

    y_tr_proba = get_proba(model, X_tr)
    y_v_proba  = get_proba(model, X_v)
    y_te_proba = get_proba(model, X_te)

    average = "macro"
    # Метрики
    metrics = {}
    for split_name, yt, yp, yp_proba in [
        ("train", y_tr, y_tr_pred, y_tr_proba),
        ("val", y_v, y_v_pred, y_v_proba),
        ("test", y_te, y_te_pred, y_te_proba),
    ]:
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average=average, zero_division=0)
        rec = recall_score(yt, yp, average=average, zero_division=0)
        f1 = f1_score(yt, yp, average=average, zero_division=0)
        # многоклассовый AUC OVR macro
        try:
            auc = roc_auc_score(yt, yp_proba, multi_class="ovr", average=average, labels=labels_order)
        except Exception:
            auc = np.nan

        metrics[split_name] = {
            "accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "roc_auc_ovr_macro": auc
        }

    print(f"\n=== {name}: classification report (val) ===")
    print(classification_report(y_v, y_v_pred, zero_division=0))
    print(f"{name}: Metrics summary:")
    for split in ["train","val","test"]:
        m = metrics[split]
        print(f"{split}: acc={m['accuracy']:.4f} | prec={m['precision_macro']:.4f} | rec={m['recall_macro']:.4f} | f1={m['f1_macro']:.4f} | auc={m['roc_auc_ovr_macro']:.4f}")

    return metrics

results = {}
labels_order = sorted(y.unique().tolist())  # фиксируем порядок классов
for name, gs in grids.items():
    best_model = gs.best_estimator_
    metrics = evaluate_model(
        name, best_model,
        X_train, y_train, X_val, y_val, X_test, y_test,
        labels_order=labels_order
    )
    results[name] = {"best_params": gs.best_params_, "cv_best_score": gs.best_score_, "metrics": metrics, "model": best_model}

# === 7) Выбор лучшей модели по валид. F1_macro и итоговая оценка на тесте ===
def pick_best_by_val_f1(results_dict):
    best_name, best_score = None, -np.inf
    for name, info in results_dict.items():
        f1_val = info["metrics"]["val"]["f1_macro"]
        if f1_val > best_score:
            best_score = f1_val
            best_name = name
    return best_name, results_dict[best_name]

best_name, best_info = pick_best_by_val_f1(results)
print("\n=== Лучшее по f1_macro (val) ===")
print("Model:", best_name)
print("Params:", best_info["best_params"])
print("Val f1_macro:", best_info["metrics"]["val"]["f1_macro"])
print("Test f1_macro:", best_info["metrics"]["test"]["f1_macro"])

best_model = best_info["model"]

# === 8) ROC-кривые для лучшей модели (one-vs-rest по классам) ===
# Строим ROC для каждого класса на валидации и тесте
def plot_multiclass_roc(model, X_set, y_set, title, labels):
    # вероятности
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_set)
    else:
        # decision_function -> softmax (см. выше)
        if hasattr(model, "decision_function"):
            dec = model.decision_function(X_set)
            if dec.ndim == 1:
                dec = np.vstack([-dec, dec]).T
            expd = np.exp(dec - dec.max(axis=1, keepdims=True))
            proba = expd / expd.sum(axis=1, keepdims=True)
        else:
            # fallback one-hot
            preds = model.predict(X_set)
            classes_ = model.classes_
            proba = np.zeros((len(preds), len(classes_)), dtype=float)
            for i, p in enumerate(preds):
                proba[i, list(classes_).index(p)] = 1.0

    # бинализуем y_set по классам
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_set, classes=labels)

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(labels):
        RocCurveDisplay.from_predictions(
            y_true=y_bin[:, i],
            y_pred=proba[:, i],
            name=f"Class {cls}"
        )
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    fname = f"images/{title.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"Saved ROC plot -> {fname}")

plot_multiclass_roc(best_model, X_val, y_val, "ROC OVR (Validation)", labels_order)
plot_multiclass_roc(best_model, X_test, y_test, "ROC OVR (Test)", labels_order)

# === 9) Интерпретация: дерево или важности признаков у леса ===
# Построим и сохраним визуализацию, если выбрано дерево; если лес — отобразим важности.

def get_feature_names(preprocess: ColumnTransformer):
    # Возвращает имена признаков после трансформации ColumnTransformer
    output_features = []
    for name, trans, cols in preprocess.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            if isinstance(cols, list):
                f_names = trans.get_feature_names_out(cols)
            else:
                f_names = trans.get_feature_names_out()
            output_features.extend(list(f_names))
        else:
            # если трансформер без метода, используем исходные имена
            if isinstance(cols, list):
                output_features.extend(cols)
            else:
                output_features.append(cols)
    return output_features

# Извлекаем внутреннюю модель из пайплайна
final_model = best_model.named_steps["model"]
final_preprocess = best_model.named_steps["preprocess"]

feature_names = get_feature_names(final_preprocess)

if isinstance(final_model, DecisionTreeClassifier):
    plt.figure(figsize=(18, 10))
    plot_tree(
        final_model,
        feature_names=feature_names,
        class_names=labels_order,
        filled=True,
        rounded=True,
        max_depth=3  # для наглядности ограничим глубину визуализации
    )
    plt.title("Decision Tree (top levels)")
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/decision_tree_top.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved tree visualization -> images/decision_tree_top.png")

elif isinstance(final_model, RandomForestClassifier):
    importances = final_model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_k = min(20, len(importances))
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[order][:top_k], y=np.array(feature_names)[order][:top_k], orient="h", palette="viridis")
    plt.title("RandomForest - Feature Importances (Top 20)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/rf_feature_importances_top20.png", dpi=200)
    plt.show()
    print("Saved feature importance -> images/rf_feature_importances_top20.png")
else:
    print("Лучшая модель не дерево и не лес — визуализация структуры/важностей пропущена.")

# === 10) Конф. матрица для лучшей модели на тесте ===
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=labels_order, cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {best_name} (Test)")
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig(f"images/confmat_{best_name.lower()}_test.png", dpi=200)
plt.show()

# === 11) Сводка результатов по моделям ===
summary = []
for name, info in results.items():
    row = {
        "model": name,
        "cv_best_f1_macro": info["cv_best_score"],
        "val_f1_macro": info["metrics"]["val"]["f1_macro"],
        "val_auc_macro_ovr": info["metrics"]["val"]["roc_auc_ovr_macro"],
        "test_f1_macro": info["metrics"]["test"]["f1_macro"],
        "test_auc_macro_ovr": info["metrics"]["test"]["roc_auc_ovr_macro"],
        "best_params": info["best_params"]
    }
    summary.append(row)

summary_df = pd.DataFrame(summary).sort_values(by="val_f1_macro", ascending=False)
print("\n=== Summary ===")
print(summary_df)

os.makedirs("results", exist_ok=True)
summary_df.to_csv("results/lab2_model_summary.csv", index=False)
print("Saved results -> results/lab2_model_summary.csv")
