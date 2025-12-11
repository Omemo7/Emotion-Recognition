import os
import sys
from pathlib import Path

from keras.config import set_max_steps_per_epoch

from evaluation import collect_predictions
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import dagshub 
import mlflow
from src.data_loader import get_datasets, get_test_dataset
from src.model import build_model
from src.config import EPOCHS, LEARNING_RATE, BATCH_SIZE, MODELS_DIR,NUM_CLASSES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys
import mlflow
from pathlib import Path
import dagshub
import os
from src.data_loader import get_test_dataset
from src.config import CLASSES
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)



dagshub.init(repo_name="Emotion-Recognition", repo_owner="Omemo7", mlflow=True)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average="macro", name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average

        # Confusion matrix accumulator
        self.cm = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch,), int labels
        # y_pred: (batch, num_classes), logits or probabilities
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_labels = tf.cast(tf.reshape(y_pred_labels, [-1]), tf.int32)

        cm = tf.math.confusion_matrix(
            y_true,
            y_pred_labels,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )

        self.cm.assign_add(cm)

    def result(self):
        tp = tf.linalg.diag_part(self.cm)
        fp = tf.reduce_sum(self.cm, axis=0) - tp
        fn = tf.reduce_sum(self.cm, axis=1) - tp

        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)

        if self.average == "macro":
            return tf.reduce_mean(f1)
        elif self.average == "micro":
            tp_sum = tf.reduce_sum(tp)
            fp_sum = tf.reduce_sum(fp)
            fn_sum = tf.reduce_sum(fn)
            precision_micro = tf.math.divide_no_nan(tp_sum, tp_sum + fp_sum)
            recall_micro = tf.math.divide_no_nan(tp_sum, tp_sum + fn_sum)
            return tf.math.divide_no_nan(
                2.0 * precision_micro * recall_micro,
                precision_micro + recall_micro,
            )
        else:
            # return per-class F1
            return f1

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# --- CUSTOM CALLBACK (The Fix) ---
# This replaces autolog() to safely log metrics epoch-by-epoch
class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log training & validation metrics safely
        mlflow.log_metric("accuracy", logs.get("accuracy"), step=epoch)
        mlflow.log_metric("loss", logs.get("loss"), step=epoch)
        mlflow.log_metric("val_accuracy", logs.get("val_accuracy"), step=epoch)
        mlflow.log_metric("val_loss", logs.get("val_loss"), step=epoch)
         # New: F1 metrics
        if "f1" in logs:
            mlflow.log_metric("f1", float(logs["f1"]), step=epoch)
        if "val_f1" in logs:
            mlflow.log_metric("val_f1", float(logs["val_f1"]), step=epoch)
        

def train():
    print("Loading Data...")
    train_ds, val_ds = get_datasets()

    print("Building Model...")
    model = build_model()
    f1_metric = F1Score(num_classes=NUM_CLASSES, average="macro", name="f1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', f1_metric]
    )

    # Define params to log manually
    params = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": "VGG16"
    }

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_f1",
    mode="max",       # higher is better
    patience=5,
    restore_best_weights=True
)

    print("Starting Training...")
    with mlflow.start_run():
        # 1. Log Params (Manually)
        mlflow.log_params(params)
        STEPS_PER_EPOCH = 25 #196 happy(majority) * .8 the train only count = 157 then * 5 = 785 then divide by batch /32 = 25 steps
        # 2. Train (with our custom callback)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=[MLflowLogger(), early_stop] # <-- Connects the logger here
        )
       
        test_ds = get_test_dataset()
        
        # Evaluate the trained model on the test set
        print("Starting Final Test Evaluation...")
        test_results = model.evaluate(test_ds, verbose=0)
        # order: [loss, accuracy, f1]
        test_loss, test_accuracy, test_f1 = test_results

        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.log_metric("final_test_f1", test_f1)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        y_true, y_pred = collect_predictions(model, test_ds)
            # ===== Classification report =====
        report_str = classification_report(y_true, y_pred, target_names=CLASSES)
        print("\n===== CLASS REPORT =====")
        print(report_str)

        # Save report as text artifact
        report_path = "last_run_eval/classification_report.txt"
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
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        cm_path = "last_run_eval/confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)

        # Log confusion matrix image
        mlflow.log_artifact(cm_path)

        print("\nEvaluation logged to MLflow.")

        # 3. Save Model Locally
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, "vgg16_emotion_final.h5")
        model.save(model_path)
        
        # 4. Upload Model as Artifact (Safe Method)
        # We upload it as a plain file, avoiding the "Registry" error
        mlflow.log_artifact(model_path)
        
        print(f"Training Finished. Model saved to {model_path} and uploaded to DagsHub.")

if __name__ == "__main__":
    train()