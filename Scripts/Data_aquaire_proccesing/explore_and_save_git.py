import itertools, json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cutlery_data import fetch_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------- PORTABLE DIRECTORIES ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data_and_features" / "Data_proccesed"
ARTIFACTS = REPO_ROOT / "artifacts_shared"
# ------------------------------------------
FEATURE_NAMES = ["aspect_ratio", "area_perimeter", "extent"]
ALLOWED_BASE = ["fork", "spoon", "knife", "ignore"]

def normalize_label(lbl: str) -> str:
    s = str(lbl).strip().lower()
    mapping = {
        "fork": "fork", "forks": "fork",
        "spoon": "spoon", "spoons": "spoon",
        "knife": "knife", "knives": "knife",
        "ignore": "ignore", "ignored": "ignore", "background": "ignore",
    }
    return mapping.get(s, s)

if __name__ == "__main__":
    data_path = Path(DATA)
    out_dir = Path(ARTIFACTS); out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(), f"Dataset folder not found: {data_path}"

    ds = fetch_data(str(data_path))          # expects ds.data (N,3), ds.target (N,)
    X_all = np.asarray(ds.data)
    y_raw = [normalize_label(t) for t in ds.target]

    present = [c for c in ALLOWED_BASE if c in set(y_raw)]
    if not present:
        raise RuntimeError("No allowed classes found in processed data.")

    keep_mask = np.isin(y_raw, present)
    X = X_all[keep_mask]
    y = np.array([t for t, k in zip(y_raw, keep_mask) if k], dtype=object)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    np.savez(out_dir / "cutlery_dataset.npz",
             X_train=X_train_scaled, y_train=y_train,
             X_test=X_test_scaled,   y_test=y_test,
             class_names=np.array(le.classes_, dtype=object),
             feature_names=np.array(FEATURE_NAMES, dtype=object))

    meta = {
        "data_path": str(data_path),
        "present_classes": present,
        "random_state": 42, "test_size": 0.25,
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "note": "X_* are scaled.",
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # (plots unchanged)
    sns.set_context("talk")
    inv_y = np.array(le.inverse_transform(y_train))

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure()
    ax = sns.countplot(x=inv_y, order=sorted(set(inv_y)))
    ax.set_title("Cutlery class distribution (train)")
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(plots_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
