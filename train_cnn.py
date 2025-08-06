import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight


tf.random.set_seed(42)
np.random.seed(42)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0,
    brightness_range=[0.95, 1.05]
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'dataset/train_augmented',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    'dataset/valid',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary'
)


class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))


def build_model():
    base_model = DenseNet121(
        input_shape=(256, 256, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

model = build_model()


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_focal_crossentropy',
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')  
    ]
)


callbacks = [
    EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', monitor='val_auc', save_best_only=True)
]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=callbacks,
    class_weight=class_weights
)


for layer in model.layers[1].layers[-50:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_focal_crossentropy',
    metrics=['accuracy', Precision(), Recall(), AUC()]
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=callbacks,
    class_weight=class_weights
)


def plot_history(history, history_fine=None):
    metrics = ['accuracy', 'loss', 'auc'] 
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history[f'val_{metric}'], label='Validation')
        if history_fine:
            plt.plot(history_fine.history[metric], label='Fine-tune Train')
            plt.plot(history_fine.history[f'val_{metric}'], label='Fine-tune Val')
        plt.title(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history, history_fine)


os.makedirs('models', exist_ok=True)
model.save('models/lung_cancer_densenet_final.keras')
print("Model saved successfully!")