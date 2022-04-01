from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping


class lstm:

    def __init__(self, x_train, y_train, x_val, y_val, n_lag, n_features, n_ahead, epochs, batch_size,
                 act_func, loss, units, learning_rate, dropout, patience, l1, l2, bias_reg, save):
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
        self.units = units
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.patience = patience
        self.l_one = l1
        self.l_two = l2
        self.bias_regularizer = bias_reg
        self.save = save

    def temperature_model(self):
        model = Sequential()
        model.add(LSTM(units=self.units,
                       activation=self.act_func,
                       input_shape=(self.n_lag, self.n_features),
                       recurrent_activation='sigmoid',
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       recurrent_initializer='orthogonal',
                       bias_initializer='zeros',
                       unit_forget_bias=False,
                       kernel_regularizer=regularizers.l1_l2(l1=self.l_one, l2=self.l_two),
                       bias_regularizer=regularizers.l1(self.bias_regularizer),
                       activity_regularizer=None,
                       kernel_constraint=None,
                       recurrent_constraint=None,
                       bias_constraint=None,
                       dropout=self.dropout,
                       recurrent_dropout=0.0,
                       return_sequences=False,
                       return_state=False,
                       go_backwards=False,
                       stateful=False,
                       time_major=False,
                       unroll=False,
                       ))

        model.add(Dense(self.n_ahead, activation="linear"))

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate, decay=1e-2 / self.epochs)

        model.compile(loss=self.loss,
                      optimizer=opt,
                      metrics=[self.loss, 'mape', 'mae'])

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
