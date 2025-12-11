import tensorflow as tf
from src.config import DATA_DIR,TEST_DIR, BATCH_SIZE, IMAGE_SIZE, SEED, VALIDATION_SPLIT

def get_datasets():
    """
    Loads images from the data directory and splits them into Train and Validation.
    """
    
    # 1. Load Training Data (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int' # Returns integer labels (0, 1, 2)
    )

    # 2. Load Validation Data (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    # 3. Optimize for Performance (Prefetching)
    # This stops the GPU from waiting for data loading
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds



def get_test_dataset(dir=TEST_DIR):
    """
    Loads the final, held-out test dataset without splitting or shuffling.
    """
    # Load the test data (100% of the files in TEST_DIR)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        shuffle=False, # Critical: Do not shuffle the test set
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Optimize for Performance
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_ds