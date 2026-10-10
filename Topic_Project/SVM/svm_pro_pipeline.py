"""
SVM from start to finish, done like a pro.
Dataset: Breast Cancer Wisconsin (built into scikit-learn, no download needed).
"""
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# ------------------------------------------------------------------ 1. LOAD
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target            # y: 0 = malignant, 1 = benign
print("Shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Class balance:\n", y.value_counts(normalize=True).round(3))
print("Feature range: min of maxes = %.3g, max of maxes = %.3g"
      % (X.max().min(), X.max().max()))

# ------------------------------------------------------------------ 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------------------------------------------------ 3. CV + METRICS
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

# ------------------------------------------------------------------ 4. BASELINES
def cv_f1(model):
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()

baselines = {
    "Dummy (majority class)":       DummyClassifier(strategy="most_frequent"),
    "SVC default, NO scaling":      SVC(),
    "SVC default + scaling":        Pipeline([("scale", StandardScaler()), ("svm", SVC())]),
    "Linear SVC + scaling":         Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="linear"))]),
}
base_scores = {name: cv_f1(m) for name, m in baselines.items()}
for k, v in base_scores.items():
    print(f"{k:28s} CV F1 = {v:.4f}")

# ------------------------------------------------------------------ 5. PIPELINE
pipe = Pipeline([("scale", StandardScaler()),
                 ("svm", SVC(class_weight="balanced", random_state=RANDOM_STATE))])

# ------------------------------------------------------------------ 6. COARSE SEARCH
coarse_grid = [
    {"svm__kernel": ["linear"],
     "svm__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    {"svm__kernel": ["rbf"],
     "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
     "svm__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
    {"svm__kernel": ["poly"],
     "svm__C": [0.1, 1, 10],
     "svm__degree": [2, 3],
     "svm__coef0": [0.0, 1.0],
     "svm__gamma": ["scale"]},
]
coarse = GridSearchCV(pipe, coarse_grid, cv=cv, scoring=scoring, refit="f1",
                      n_jobs=-1, return_train_score=True)
t0 = time.time()
coarse.fit(X_train, y_train)
print("Coarse search took %.1f s" % (time.time() - t0))
print("\nCOARSE best:", coarse.best_params_, " CV F1 = %.4f" % coarse.best_score_)
n_fits = len(coarse.cv_results_["params"]) * cv.get_n_splits()
print("Coarse candidates:", len(coarse.cv_results_["params"]), " fits:", n_fits)

# ------------------------------------------------------------------ 6b. READ THE COARSE RESULTS
res = pd.DataFrame(coarse.cv_results_)
top = (res.sort_values("rank_test_f1")
          [["param_svm__kernel", "param_svm__C", "param_svm__gamma",
            "mean_test_f1", "std_test_f1", "mean_train_f1"]].head(8))
print(top.to_string(index=False))

# ------------------------------------------------------------------ 7. FINE SEARCH
best = coarse.best_params_
if best["svm__kernel"] == "rbf":
    C0, g0 = best["svm__C"], best["svm__gamma"]
    fine_grid = {"svm__kernel": ["rbf"],
                 "svm__C": np.round(np.geomspace(C0 / 4, C0 * 4, 7), 6).tolist(),
                 "svm__gamma": np.round(np.geomspace(g0 / 4, g0 * 4, 7), 8).tolist()}
    fine = GridSearchCV(pipe, fine_grid, cv=cv, scoring=scoring, refit="f1",
                        n_jobs=-1, return_train_score=True)
    t0 = time.time()
    fine.fit(X_train, y_train)
    print("Fine search took %.1f s" % (time.time() - t0))
    print("\nFINE best:", fine.best_params_, " CV F1 = %.4f" % fine.best_score_)
    final_search = fine
else:
    final_search = coarse

# ------------------------------------------------------------------ 7b. BREAK TIES ON PURPOSE
def simplest_within_one_se(cv_results):
    """Among candidates within 1 standard error of the best F1,
    pick the smoothest model: smallest C, then smallest gamma."""
    mean = np.asarray(cv_results["mean_test_f1"])
    se = np.asarray(cv_results["std_test_f1"]) / np.sqrt(cv.get_n_splits())
    ok = np.flatnonzero(mean >= mean.max() - se[mean.argmax()])
    C = np.asarray(cv_results["param_svm__C"], dtype=float)[ok]
    g = np.asarray(cv_results["param_svm__gamma"], dtype=float)[ok]
    return int(ok[np.lexsort((g, C))[0]])          # sort by C, then gamma

if best["svm__kernel"] == "rbf":
    simple = GridSearchCV(pipe, fine_grid, cv=cv, scoring=scoring,
                          refit=simplest_within_one_se, n_jobs=-1,
                          return_train_score=True)
    simple.fit(X_train, y_train)
    print("Simplest-within-1-SE:", simple.best_params_)
    n_tied = int((np.asarray(simple.cv_results_["mean_test_f1"]) >= fine.best_score_ - 1e-12).sum())
    print("Candidates tied with the best score:", n_tied, "of", len(simple.cv_results_["params"]))
    final_search = simple
final_model = final_search.best_estimator_

# ------------------------------------------------------------------ 8. TEST ONCE
y_pred = final_model.predict(X_test)
y_score = final_model.decision_function(X_test)
print("\nTEST results")
print(classification_report(y_test, y_pred, target_names=["malignant", "benign"], digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC = %.4f" % roc_auc_score(y_test, y_score))

# ------------------------------------------------------------------ 9. HEALTH CHECK
svm = final_model.named_steps["svm"]
n_sv = svm.n_support_.sum()
print("Support vectors: %d of %d (%.0f%%)" % (n_sv, len(X_train), 100 * n_sv / len(X_train)))
r = pd.DataFrame(final_search.cv_results_).iloc[final_search.best_index_]
print("Chosen cell: train F1 = %.4f  vs  CV F1 = %.4f" % (r["mean_train_f1"], r["mean_test_f1"]))

# ------------------------------------------------------------------ 10. SAVE
joblib.dump(final_model, "svm_breast_cancer_v1.joblib")
loaded = joblib.load("svm_breast_cancer_v1.joblib")
print("Reloaded model agrees on test set:", bool((loaded.predict(X_test) == y_pred).all()))

# ------------------------------------------------------------------ 11. PREDICT ON NEW DATA
new_patient = X_test.iloc[[0]]                     # raw, unscaled numbers: the pipeline scales them
label = int(loaded.predict(new_patient)[0])
print("Prediction:", data.target_names[label],
      "| decision score = %.2f" % loaded.decision_function(new_patient)[0],
      "| truth:", data.target_names[int(y_test.iloc[0])])

# ------------------------------------------------------------------ BONUS: nested CV
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
small_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [1e-3, 1e-2, 1e-1, 1]}
nested_search = GridSearchCV(pipe, small_grid, cv=inner, scoring="f1", n_jobs=-1)
nested = cross_val_score(nested_search, X, y, cv=outer, scoring="f1")
print("\nNested CV F1: %.4f +/- %.4f" % (nested.mean(), nested.std()))
