import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
from augmentation import get_augmenter
from encoder import get_encoder

#Dataset Loader
def load_contrastive_datasets(batch_size=64):
    train_images = np.load("D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/train_slices.npy").astype(np.float32)
    train_labels = np.load("D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/train_labels_2d.npy").astype(np.int32)
    test_images = np.load("D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/test_slices.npy").astype(np.float32)
    test_labels = np.load("D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/test_labels_2d.npy").astype(np.int32)

    train_images = np.clip(train_images, 0, 255) / 255.0
    test_images = np.clip(test_images, 0, 255) / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    min_len = min(train_images.shape[0], train_labels.shape[0])
    train_images = train_images[:min_len]
    train_labels = train_labels[:min_len]

    if min_len % 2 != 0:
        min_len -= 1
        train_images = train_images[:min_len]
        train_labels = train_labels[:min_len]

    split_index = min_len // 2
    unlabeled_images = train_images[:split_index]
    dummy_labels = np.zeros(split_index, dtype=np.int32)
    labeled_images = train_images[split_index:]
    labeled_labels = train_labels[split_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((unlabeled_images, dummy_labels), (labeled_images, labeled_labels))
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

# ───────────────────────────────

width = 128
num_epochs = 20
temperature = 0.5
contrastive_augmentation = {"min_area": 0.5}
classification_augmentation = {"min_area": 0.8}

class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = get_encoder()

        self.projection_head = keras.Sequential([
            keras.Input(shape=(width,)),
            layers.Dense(width, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(width, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(width, kernel_initializer='he_normal'),
            layers.BatchNormalization()
        ], name="projection_head")
        '''
        self.projection_head = keras.Sequential([
            keras.Input(shape=(width,)),
            layers.Dense(width),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(width),
            layers.BatchNormalization(),
        ], name="projection_head")
        '''
        self.linear_probe = keras.Sequential([
            layers.Input(shape=(width,)),
            layers.BatchNormalization(),
            layers.Dense(2),
        ], name="linear_probe")

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # Normalize the projections
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        batch_size = tf.shape(projections_1)[0]

        # Compute similarities with higher temperature at start
        similarities = tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature

        # Create labels for the positive pairs
        labels = tf.range(batch_size)

        # Compute loss for both directions
        loss_1_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, similarities)
        loss_2_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, tf.transpose(similarities))

        # Update accuracy metric
        self.contrastive_accuracy.update_state(labels, similarities)
        self.contrastive_accuracy.update_state(labels, tf.transpose(similarities))

        return tf.reduce_mean(loss_1_2 + loss_2_1) / 2
        '''
        projections_1 = ops.normalize(projections_1, axis=1)
        projections_2 = ops.normalize(projections_2, axis=1)
        similarities = ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        batch_size = ops.shape(projections_1)[0]
        contrastive_labels = ops.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, ops.transpose(similarities))
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, ops.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2
        '''

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        images = ops.concatenate((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(contrastive_loss, self.encoder.trainable_weights + self.projection_head.trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_weights + self.projection_head.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        preprocessed_images = self.classification_augmenter(labeled_images, training=True)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)

        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(zip(gradients, self.linear_probe.trainable_weights))
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data
        preprocessed_images = self.classification_augmenter(labeled_images, training=False)
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        return {m.name: m.result() for m in self.metrics[2:]}

train_dataset, test_dataset = load_contrastive_datasets(batch_size=64)
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)
pretraining_history = pretraining_model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=test_dataset,
)
save_path = "/content/drive/MyDrive/Brain_Tumor_Data/contrastive_model.h5"
pretraining_model.save(save_path)

print("Maximal validation accuracy: {:.2f}%".format(
    max(pretraining_history.history["val_p_acc"]) * 100
))
