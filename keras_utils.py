import tensorflow.keras as keras
import numpy as np

def truncate_model(full_model):
    return keras.Model(full_model.input, full_model.layers[-2].output)

def get_model_output_features(model, X):
    # Assuming that the input has been normalized using:
    # X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)
    X = np.asarray(X).astype(np.float32)
    return model.predict(X)