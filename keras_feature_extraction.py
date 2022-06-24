from time import time

from move_dataset import MoveDataset

from torch.utils.data import DataLoader
from keras_fcn import build_fcn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn import metrics

dataset = MoveDataset(pickle_file='move_data.pkl')
loader = DataLoader(dataset, batch_size=2, shuffle=True)
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

X, y = dataset.train_data()
X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)

cb, model = build_fcn((128, 2), dataset.unique_user_count(), 128, tensorboard)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11235)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=11235)

mini_batch_size = int(min(X_train.shape[0] / 10, 512))
X_train = np.asarray(X_train).astype(np.float32)
X_val = np.asarray(X_val).astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = train_ds.shuffle(100).batch(512)
val_ds = val_ds.batch(512)

hist = model.fit(train_ds,
                 epochs=200,
                 verbose=True,
                 validation_data=val_ds,
                 callbacks=cb)

X_test = np.asarray(X_test).astype(np.float32)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = metrics.accuracy_score(y_true, y_pred)
print("Test accuracy: " + str(accuracy))
