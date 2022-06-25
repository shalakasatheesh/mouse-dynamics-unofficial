from time import time
from move_dataset import MoveDataset
import argparse
from keras_fcn import build_fcn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard


def main():
    # Extract arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="move_data.pkl")
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--tensorboard", type=bool, default=True)
    args = parser.parse_args()

    dataset = MoveDataset(pickle_file=args.dataset)
    if args.tensorboard:
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    else:
        tensorboard = None

    X, y = dataset.train_data()
    X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)

    cb, model = build_fcn((128, 2), dataset.unique_user_count(), 128, tb=tensorboard,
                          learning_rate=args.learning_rate,
                          model_path=args.model, verbose=args.verbose)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11235)

    mini_batch_size = int(min(X_train.shape[0] / 10, args.batch))
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_ds = train_ds.shuffle(2 * mini_batch_size).batch(mini_batch_size)
    val_ds = val_ds.batch(mini_batch_size)

    model.fit(train_ds,
              epochs=args.epochs,
              verbose=args.verbose,
              validation_data=val_ds,
              callbacks=cb)

    model.save('final_' + args.model)


if __name__ == "__main__":
    main()
