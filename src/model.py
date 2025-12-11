import tensorflow as tf
from tensorflow import keras
from src.config import IMAGE_SIZE, CLASSES
from keras import regularizers




def build_model():
    """
    Builds a VGG16 model adapted for Emotion Recognition.
    """
    input_shape = IMAGE_SIZE + (3,) # Result: (224, 224, 3)
    num_classes = len(CLASSES)

    # 1. Load the Pre-trained Base
    # include_top=False removes the 1000-class ImageNet layer at the end
    base_model = keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet' 
    )

    # 2. Freeze the Base
    # We don't want to break the pre-trained patterns initially
    base_model.trainable = False 

    # 3. Add Custom Classification Head
    model = keras.Sequential([
        # Data Augmentation (Optional but recommended)
        keras.layers.RandomFlip("horizontal", input_shape=input_shape),
        keras.layers.RandomRotation(0.1),
        keras.layers.Rescaling(1./255), # VGG expects pixels 0-1 or normalized
        
        base_model,
        
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5), # Reduces overfitting
        keras.layers.Dense(num_classes, activation='softmax') # Final output
    ])

    return model