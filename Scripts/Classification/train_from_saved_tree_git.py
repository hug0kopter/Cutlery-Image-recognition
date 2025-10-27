from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

# ---------- PORTABLE PATHS ----------
# This file lives in: Cutlery_Image_Recognition/Scripts/Classification/
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts_shared"
# -----------------------------------

if __name__ == "__main__":
    artifacts = ARTIFACTS
    ds_path = artifacts / "cutlery_dataset.npz"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n  {ds_path}\nRun explore_and_save_fixed.py first."
        )

    data = np.load(ds_path, allow_pickle=True)
    X_train = data["X_train"]          # scaling not required for trees
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    class_names = data["class_names"].tolist()
    feature_names = data["feature_names"].tolist()

    print(f"Loaded dataset: {ds_path}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Classes: {class_names}")

    # ---- Decision Tree classifier ----
    clf = DecisionTreeClassifier(
        criterion="gini",     # or "entropy", "log_loss"
        max_depth=10,         # try None, 5–20 if needed
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    y_pred = clf.predict(X_test)
    print("\nDecision Tree classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    print(f"\nCV accuracy (train, 5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std()*2:.4f}")

    model_out = artifacts / "cutlery_tree.joblib"
    joblib.dump(
        {"model": clf, "class_names": class_names, "feature_names": feature_names},
        model_out
    )
    print(f"[OK] Saved model to: {model_out.resolve()}")
