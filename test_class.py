import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
eps = 1e-3

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0





class ConvBnRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBnRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.99,)

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        return inputs


class CicleLoss(tf.keras.Model):
    def __init__(self, category=10, margin=0., reweight=5.):
        super(CicleLoss, self).__init__()
        self.reweight = reweight
        self.pos_margin = 1. - margin
        self.neg_margin = margin
        self.category = category
        self.op = 1. + self.pos_margin
        self.on = - self.neg_margin
        """需要使用归一化产生最优正分和最优负分吗？"""
        self.dense = tf.keras.layers.Dense(units=category, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        if training:
            inputs, label = inputs
            score = self.dense(inputs)
            label = tf.one_hot(label, depth=self.category, dtype=tf.float32)
            neg = tf.exp(self.reweight*tf.nn.relu(score-self.on)*(score-self.neg_margin))
            pos = tf.exp(-self.reweight*tf.nn.relu(self.op-score)*(score-self.pos_margin))
            pos = k.sum(pos*label, axis=-1)
            neg = k.sum(neg*(1.-label), axis=-1)
            loss = tf.math.log1p(pos * neg)
            return k.mean(loss)
        else:
            score = self.dense(inputs)
            return tf.argmax(score, axis=-1)


class classfier(tf.keras.Model):
    def __init__(self):
        super(classfier, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")
        self.conv1 = ConvBnRelu(filters=64, kernel_size=3, strides=1)
        self.conv2 = ConvBnRelu(filters=128, kernel_size=3, strides=1)
        self.conv3 = ConvBnRelu(filters=256, kernel_size=3, strides=2)
        self.conv4 = ConvBnRelu(filters=256, kernel_size=3, strides=1)
        self.conv5 = ConvBnRelu(filters=256, kernel_size=3, strides=1)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.conv0(inputs))
        inputs = self.conv1(inputs, training=training)
        inputs = self.conv2(inputs, training=training)
        inputs = self.conv3(inputs, training=training)
        inputs = self.conv4(inputs, training=training)
        inputs = self.conv5(inputs, training=training)
        inputs = self.pool(inputs)
        inputs = self.latten_compress(inputs)
        return inputs


if __name__ == '__main__':
    traindataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    testdataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    prediction_matrix = CicleLoss()
    optimizer = tf.keras.optimizers.Adam(5e-4)
    # Training
    model = classfier()
    traindataset = traindataset.cache().shuffle(10000, reshuffle_each_iteration=True).batch(256, drop_remainder=True)
    testdataset = testdataset.batch(256, drop_remainder=True)
    for epoch in range(50):
        for syncdata in traindataset:
            images, labels = syncdata
            images = 2 * (tf.cast(images[..., tf.newaxis], tf.float32) - .5)
            labels = tf.cast(labels, tf.int64)
            with tf.GradientTape() as tape:
                embedding = model(images, training=True)
                loss = prediction_matrix([embedding, labels], training=True)
                # loss = k.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                #                                                              labels=labels[:, 0]),
                #               axis=-1)
                grd = tape.gradient(loss, model.trainable_variables + prediction_matrix.trainable_variables)
                optimizer.apply_gradients(zip(grd, model.trainable_variables + prediction_matrix.trainable_variables))
        mean_acc = []
        for syncdata in testdataset:
            images, labels = syncdata
            images = 2 * (tf.cast(images[..., tf.newaxis], tf.float32) - .5)
            labels = tf.cast(labels, tf.int64)
            embedding = model(images, training=False)
            embedding = embedding.numpy()
            prediction = prediction_matrix(embedding, training=False)
            acc = np.mean(labels.numpy() == prediction.numpy())
            mean_acc.append(acc)
        print(np.mean(acc))
