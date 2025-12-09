import os

# --- PATHS ---
# Relies on the structure we created: data/raw_images/class_name/img.jpg
DATA_DIR = os.path.join("data", "working_set")
TEST_DIR = os.path.join("data", "final_test_set")
MODELS_DIR = "models"

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Slower LR is better for fine-tuning VGG
EPOCHS =20
IMAGE_SIZE = (224, 224) # VGG standard input size
VALIDATION_SPLIT = 0.2
SEED = 42

# --- MODEL CONFIG ---
CLASSES = ['angry', 'fear', 'happy','sad','surprised']
NUM_CLASSES = len(CLASSES)