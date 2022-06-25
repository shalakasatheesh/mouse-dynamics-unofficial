import tensorflow.keras as keras


def build_fcn(input_shape, nb_classes, num_filters=128, learning_rate=0.0001, model_path='model.h5', verbose=True, tb=None):
    input_layer = keras.layers.Input(input_shape)

    # conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.Conv1D(filters=num_filters, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    # conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.Conv1D(filters=2 * num_filters, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    # conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.Conv1D(num_filters, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=50,
                                                  min_lr=learning_rate)

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_categorical_accuracy',
                                                       save_best_only=True,
                                                       verbose=verbose)

    callbacks = [reduce_lr, model_checkpoint]
    if tb is not None:
        callbacks.append(tb)

    return callbacks, model
