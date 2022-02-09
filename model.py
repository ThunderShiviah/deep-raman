import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import skimage
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from utils import *
import metrics

import streamlit as st

loss_options = {'mean absolute error': keras.losses.mean_absolute_error,
    'peak signal to noise ratio': metrics.psnr_loss,
    'mean squared error': keras.losses.mean_squared_error}

NUM_EPOCHS = st.selectbox('Number of epochs', [10**i for i in range(0,3)])
loss_choice = st.selectbox('Loss function', loss_options.keys())

LOSS = loss_options[loss_choice]

x = np.linspace(-200,200,1024)

X, y = generate_training_set(x, num_base_examples=64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

X_test, y_test = generate_training_set(x, num_base_examples=15)

x_train = np.array(X_train).reshape(-1, 1024, 1) 
x_test = np.array(X_test).reshape(-1, 1024, 1) 

y_train = np.array(y_train).reshape(-1, 1024, 1) 
y_test = np.array(y_test).reshape(-1, 1024, 1)  

inputs = keras.Input(shape=(32*32, 1))
x = layers.BatchNormalization(axis=-1)(inputs)

x = layers.Conv1D(16, 16, input_shape=(32*32, 1))(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(16, 16, 16)(x)
x = layers.MaxPooling1D(3)(x)

x = layers.Conv1D(64, 10)(x)
outputs = layers.Conv1DTranspose(1, 1024)(x)


model = models.Model(inputs=inputs, outputs=outputs, name="cnn_model")

model.compile(
    loss=LOSS,
    optimizer=keras.optimizers.Nadam(learning_rate=3e-3),
    metrics=["mae", "mape"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2,)

test_scores = model.evaluate(x_test, y_test, verbose=2)
st.write("Test loss:", test_scores[0])
st.write("Test accuracy:", test_scores[1])

sample_input, sample_prediction_, sample_target_ = x_train[0:1], model.predict(x_train[0:1]), y_train[0:1]

fig = plt.figure(figsize=(12,8))

plt.subplot(311)
plt.title("Sample Input")
plt.plot(sample_input.ravel())


plt.subplot(312)
plt.title("Sample Prediction")
plt.plot(sample_prediction_.ravel())

plt.subplot(313)
plt.title("Sample Target")
plt.plot(sample_target_.ravel())
fig.tight_layout()

fig

# TODO: Visualize difference between train loss and test loss - something like tensorboard?


