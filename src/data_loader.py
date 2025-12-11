import tensorflow as tf
from src.config import DATA_DIR,TEST_DIR, BATCH_SIZE, IMAGE_SIZE, SEED, VALIDATION_SPLIT

import tensorflow as tf

import tensorflow as tf

# Augmentation layers (Applied to Training)
augment_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.2),
])

def get_datasets(data_dir=DATA_DIR):
    # ---------------------------------------------------------
    # 1. Load Training First (To detect classes)
    # ---------------------------------------------------------
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
        # Implicit: No class_names argument. 
        # It reads folders A-Z automatically.
    )

    # DYNAMICALLY detect classes
    detected_classes = raw_train_ds.class_names
    num_classes = len(detected_classes)
    print(f"Found {num_classes} classes: {detected_classes}")
    
    # ---------------------------------------------------------
    # 2. Load Validation
    # ---------------------------------------------------------
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # ---------------------------------------------------------
    # 3. Balance & Augment
    # ---------------------------------------------------------
    # Unbatch to process single images
    ds_unbatched = raw_train_ds.unbatch()
    
    datasets = []
    
    # Loop 0 to N-1 based on what we found
    for i in range(num_classes):
        # Filter for class 'i'
        class_ds = ds_unbatched.filter(lambda x, y: y == i)
        
        # Apply augmentation to everyone (Safe for small data)
        # If you really want to skip the Majority (Happy), check the print log 
        # to see which index 'happy' got (likely 2), and add: if i != 2:
        class_ds = class_ds.map(lambda x, y: (augment_layers(x, training=True), y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
        class_ds = class_ds.repeat()
        datasets.append(class_ds)

    # Calculate even weights (e.g., if 5 classes -> [0.2, 0.2, 0.2, 0.2, 0.2])
    # This forces every class to represent an equal slice of the pie.
    equal_weights = [1.0 / num_classes] * num_classes

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        datasets, 
        weights=equal_weights
    )

    # Shuffle & Batch
    train_ds = balanced_ds.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

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