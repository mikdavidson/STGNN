import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping


physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class temporalcn:

    def __init__(self, x_train, y_train, x_val, y_val, n_lag, n_features, n_ahead, epochs, batch_size,
                 act_func, loss, learning_rate, batch_norm, layer_norm, weight_norm, kernel,
                 filters, dilations, padding, dropout, patience, save):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.n_lag = n_lag
        self.n_features = n_features
        self.n_ahead = n_ahead
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_func = act_func
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_norm = weight_norm
        self.kernel = kernel
        self.filters = filters
        self.dilations = dilations
        self.padding = padding
        self.dropout = dropout
        self.patience = patience
        self.save = save

    def temperature_model(self):
        model = Sequential()
        model.add(TCN(
            input_shape=(self.n_lag, self.n_features),
            activation=self.act_func,
            nb_filters=self.filters,
            kernel_size=self.kernel,
            dilations=self.dilations,
            dropout_rate=self.dropout,
            use_batch_norm=self.batch_norm,
            use_weight_norm=self.weight_norm,
            use_layer_norm=self.layer_norm,
            return_sequences=True,
            padding=self.padding
        ))
        model.add(TCN(
            input_shape=(self.n_lag, self.n_features),
            activation=self.act_func,
            nb_filters=self.filters,
            kernel_size=self.kernel,
            dilations=self.dilations,
            dropout_rate=self.dropout,
            use_batch_norm=self.batch_norm,
            use_weight_norm=self.weight_norm,
            use_layer_norm=self.layer_norm,
            return_sequences=False,
            padding=self.padding
        ))
        model.add(Dense(self.n_ahead, activation="linear"))

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate, decay=1e-2/self.epochs)

        model.compile(loss=self.loss,
                      optimizer=opt,
                      metrics=[self.loss, 'mape'])

        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        checkpoint = ModelCheckpoint(self.save, save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     mode='min', save_freq='epoch')
        callback = [early_stop, checkpoint]

        history = model.fit(self.x_train, self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callback)
        return model, history
