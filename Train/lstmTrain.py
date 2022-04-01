import Utils.baselineUtils as utils
import pandas as pd
import Models.lstm as lstm
from keras.models import load_model
import tensorflow as tf

def train(stations, increment):
    """
    Trains the final LSTM models for each weather station across all forecasting horizons
    using walk-forward validation across 27 splits. Ideal parameters are read in from a text file. The parameters are
    then converted to a list. The train, validation, and test sets are normalised using MinMax scaler, the normalised
    sets are then processed into sliding-window input-output pairs. The LSTM model is the instantiated, trained on the
    train set and tested on the test set. The predictions, targets, and losses are written to .csv files for each
    individual weather station across all the forecasting horizons.

    Parameters:
        stations -  List of weather stations.
        increment - Walk-forward validation split points.
    """

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    forecasting_horizons = [3, 6, 9, 12, 24]

    for forecast_len in forecasting_horizons:
        configFile = open("Train/Best Configurations/lstm_params.txt", "r")

        for station in stations:
            # printing out which station we are forecasting
            print('Forecasting at station: {0} '.format(station))
            # pulling in weather station data
            weatherData = 'Data/Weather Station Data/' + station + '.csv'
            ts = utils.create_dataset(weatherData)

            # reading in the parameters from the text file
            params = configFile.readline()
            cfg = utils.stringtoCfgLSTM(params)
            # setting parameters for lstm model
            units = cfg[0]
            batch = cfg[1]
            epoch = cfg[2]
            lag_length = cfg[3]
            dropout = cfg[4]
            lr = cfg[5]
            l1 = cfg[6]
            l2 = cfg[7]
            bias_reg = cfg[8]
            activation = 'tanh'
            patience = 5
            loss_metric = 'MSE'
            n_ahead_length = forecast_len

            lossDF = pd.DataFrame()
            resultsDF = pd.DataFrame()
            targetDF = pd.DataFrame()

            targetFile = 'Results/LSTM/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
                         'target.csv'
            resultsFile = 'Results/LSTM/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                          'result.csv'
            lossFile = 'Results/LSTM/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                       'loss.csv'

            num_splits = 27
            for k in range(num_splits):
                print('LSTM Model on split {0}/27 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                      forecast_len))

                saveFile = 'Garage/Final Models/LSTM/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                           + str(n_ahead_length) + '_walk_' + str(k) + '.h5'

                split = [increment[k], increment[k + 1], increment[k + 2]]
                pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)

                # Scaling the data
                train, validation, test = utils.min_max(pre_standardize_train,
                                                        pre_standardize_validation,
                                                        pre_standardize_test)

                # Defining input shape
                n_ft = train.shape[1]

                # Creating the X and Y for forecasting
                X_train, Y_train = utils.create_X_Y(train, lag_length, n_ahead_length)

                # Creating the X and Y for validation set
                X_val, Y_val = utils.create_X_Y(validation, lag_length, n_ahead_length)

                # Get the X feature set for training
                X_test, Y_test = utils.create_X_Y(test, lag_length, n_ahead_length)

                # Creating the lstm model for temperature prediction
                lstm_model = lstm.lstm(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                       n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                       epochs=epoch, batch_size=batch, act_func=activation, loss=loss_metric,
                                       learning_rate=lr, dropout=dropout, patience=patience, units=units,
                                       l1=l1, l2=l2, bias_reg=bias_reg, save=saveFile)

                # Training the model
                model, history = lstm_model.temperature_model()

                # validation and train loss to dataframe
                lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

                # load best model
                model = load_model(saveFile)
                # Test the model and write to file
                yhat = model.predict(X_test)
                # predictions to dataframe
                resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])
                # Targets to dataframe
                targetDF = pd.concat([targetDF, pd.Series(Y_test.reshape(-1, ))])

            resultsDF.to_csv(resultsFile)
            lossDF.to_csv(lossFile)
            targetDF.to_csv(targetFile)

        configFile.close()
