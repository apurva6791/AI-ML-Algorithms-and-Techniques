import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Encoder
def build_encoder():
    input_img = layers.Input(shape=(784,))
    x = layers.Dense(128, activation="relu")(input_img)
    encoded = layers.Dense(64, activation="relu")(x)
    return models.Model(input_img, encoded)

# Decoder
def build_decoder():
    encoded_input = layers.Input(shape=(64,))
    x = layers.Dense(128, activation="relu")(encoded_input)
    decoded = layers.Dense(784, activation="sigmoid")(x)
    return models.Model(encoded_input, decoded)

# Load data
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Build models
encoder = build_encoder()
decoder = build_decoder()

input_img = layers.Input(shape=(784,))
encoded = encoder(input_img)
decoded = decoder(encoded)

autoencoder = models.Model(input_img, decoded)

# Compile & train
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test, X_test)
)

