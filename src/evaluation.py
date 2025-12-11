import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys
import mlflow
from pathlib import Path
import dagshub
import os
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.data_loader import get_test_dataset
from src.config import CLASSES
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
dagshub.init(repo_name="Emotion-Recognition", repo_owner="Omemo7", mlflow=True)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "models/vgg16_emotion_final3.h5"
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Loaded model from: {MODEL_PATH}")

# ==========================
# LOAD TEST DATA
# ==========================
test_ds = get_test_dataset()  # <-- your existing function


# ==========================
# PREDICTION COLLECTION
# ==========================
def collect_predictions(model, test_ds):
    y_true = []
    y_pred = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.append(batch_y.numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


# ==========================
# EVALUATION
# ==========================
def evaluate_saved_model(model, test_ds, class_names=None):
    y_true, y_pred = collect_predictions(model, test_ds)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    p   = precision_score(y_true, y_pred, average="macro")
    r   = recall_score(y_true, y_pred, average="macro")

    print("\n===== TEST METRICS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}  (macro)")
    print(f"Precision: {p:.4f}  (macro)")
    print(f"Recall:    {r:.4f}  (macro)")

    # ---- Detailed per-class scores ----
    print("\n===== CLASS REPORT =====")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc, f1, p, r, cm


def evaluate_and_log_model(model_path, test_ds, class_names, run_name="evaluate_saved_model"):
    # ===== Load model =====
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # ===== Start MLflow run =====
    with mlflow.start_run(run_name=run_name):
        # optional: log that this is an evaluation-only run
        mlflow.log_param("eval_model_path", model_path)

        # ===== Get predictions =====
        y_true, y_pred = collect_predictions(model, test_ds)

        # ===== Metrics =====
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        p   = precision_score(y_true, y_pred, average="macro")
        r   = recall_score(y_true, y_pred, average="macro")

        print("\n===== TEST METRICS =====")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}  (macro)")
        print(f"Precision: {p:.4f}  (macro)")
        print(f"Recall:    {r:.4f}  (macro)")

        # ---- Log metrics to MLflow ----
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1)
        mlflow.log_metric("test_precision_macro", p)
        mlflow.log_metric("test_recall_macro", r)

        # ===== Classification report =====
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        print("\n===== CLASS REPORT =====")
        print(report_str)

        # Save report as text artifact
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path)

        # ===== Confusion Matrix =====
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        cm_path = "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)

        # Log confusion matrix image
        mlflow.log_artifact(cm_path)

        print("\nEvaluation logged to MLflow.")

        # Optionally return stuff if you want to use it in code
        return acc, f1, p, r, cm



# ==========================
# CLASS NAMES
# ==========================
# Just customize the order to your label encoding


# ==========================
# RUN EVALUATION
# ==========================
#evaluate_and_log_model(MODEL_PATH, test_ds, CLASSES)

