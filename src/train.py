import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import dagshub 
import mlflow
from src.data_loader import get_datasets, get_test_dataset
from src.model import build_model
from src.config import EPOCHS, LEARNING_RATE, BATCH_SIZE, MODELS_DIR



dagshub.init(repo_name="Emotion-Recognition", repo_owner="Omemo7", mlflow=True)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

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
        

def train():
    print("Loading Data...")
    train_ds, val_ds = get_datasets()

    print("Building Model...")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define params to log manually
    params = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": "VGG16"
    }

    print("Starting Training...")
    with mlflow.start_run():
        # 1. Log Params (Manually)
        mlflow.log_params(params)
        
        # 2. Train (with our custom callback)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[MLflowLogger()] # <-- Connects the logger here
        )
        is_test_split = False #Todo: remove this after splitting the test set
        #temporary until splitting the test set
        if is_test_split:
            test_ds = get_test_dataset()
            
            # Evaluate the trained model on the test set
            print("Starting Final Test Evaluation...")
            test_results = model.evaluate(test_ds, verbose=0)
            
            test_loss = test_results[0]
            test_accuracy = test_results[1]
            
            # Log the critical final metrics to MLflow
            mlflow.log_metric("final_test_loss", test_loss)
            mlflow.log_metric("final_test_accuracy", test_accuracy)
            
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # 3. Save Model Locally
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        model_path = os.path.join(MODELS_DIR, "vgg16_emotion_final3.h5")
        model.save(model_path)
        
        # 4. Upload Model as Artifact (Safe Method)
        # We upload it as a plain file, avoiding the "Registry" error
        mlflow.log_artifact(model_path)
        
        print(f"Training Finished. Model saved to {model_path} and uploaded to DagsHub.")

if __name__ == "__main__":
    train()