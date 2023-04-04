import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model, Sequential

def build_autoencoder(input_dim, encoding_dim, features):
    input_layer = Input(shape=(input_dim,))
    encoder = Sequential([Dense(encoding_dim, activation='relu') for _ in range(features)])
    decoder = Sequential([Dense(input_dim, activation='sigmoid') for _ in range(features)])

    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder_model = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder_model

input_dim = 784 # Example input dimension for MNIST dataset
encoding_dim = 128 # Example encoding dimension
features = 4

autoencoder, encoder = build_autoencoder(input_dim, encoding_dim, features)

# Train the autoencoder using your data
# autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Encode and decode the data
# encoded_data = encoder.predict(x_test)
# decoded_data = autoencoder.predict(x_test)

# To implement the autoencoder into another neural network, you can use the 'encoder' model as a feature extractor.
# Use 'encoded_data' as input for your neural network.