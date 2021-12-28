# CJD @ Tempest Bioinstruments
# From tutorial https://keras.io/examples/timeseries/timeseries_transformer_classification/


# IMPORTING DATA
import numpy as np

def readUCR(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:,0]
    x = data[:,1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readUCR(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readUCR(root_url + "FordA_TEST.tsv")


# STANDARDIZING THE DATA
print("Before reshaping, x_train has shape ",end='')
print(x_train.shape)
print("Before reshaping, x_test has shape ",end='')
print(x_test.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
print("After reshaping, x_train has shape ",end='')
print(x_train.shape)
print("After reshaping, x_test has shape ",end='')
print(x_test.shape)


# SHUFFLE
n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# BUILD THE MODEL
from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads,
                                  dropout=dropout)(x,x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)    
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs,outputs)


    

