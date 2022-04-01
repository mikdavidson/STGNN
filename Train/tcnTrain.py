import tensorflow as tf
import Models.tcnone as tcn_one
import Models.tcntwo as tcn_two
import Utils.baselineUtils as utils
import pandas as pd
from keras.models import load_model
from tcn import TCN
import tensorflow as tf


def train(stations, increment):
    """
    Trains the final TCN models for each weather station across all forecasting horizons
    using walk-forward validation across 27 splits. Ideal parameters are read in from a text file. The parameters are
    then converted to a list. The train, validation, and test sets are normalised using MinMax scaler, the normalised
    sets are then processed into sliding-window input-output pairs. The TCN model is the instantiated, trained on the
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
        configFile = open("Train/Best Configurations/tcn_params.txt", "r")

        for station in stations:
            # printing out which station we are forecasting
            print('Forecasting at station ', station)

            # pulling in weather station data
            weatherData = 'Data/Weather Station Data/' + station + '.csv'
            ts = utils.create_dataset(weatherData)

            # reading in the parameters from the text file
            params = configFile.readline()
            cfg = utils.stringtoCfgTCN(params)

            # setting parameters for tcn model
            layers = int(cfg[0])
            filters = int(cfg[1])
            lag_length = int(cfg[2])
            batch = int(cfg[3])
            dropout = float(cfg[4])
            activation = cfg[5]
            epoch = 40
            lr = 0.01
            patience = 5
            loss_metric = 'MSE'
            kernels = 2
            dilations = [1, 2, 4, 8, 16, 32, 64]
            batch_norm = False
            weight_norm = False
            layer_norm = True
            padding = 'causal'
            n_ahead_length = forecast_len

            lossDF = pd.DataFrame()
            resultsDF = pd.DataFrame()
            targetDF = pd.DataFrame()

            targetFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
                         'target.csv'
            resultsFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                          'result.csv'
            lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                       'loss.csv'

            num_splits = 27
            for k in range(num_splits):
                print('TCN Model on split {0}/27 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                     forecast_len))

                saveFile = 'Garage/Final Models/TCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
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

                # Creating the tcn model for temperature prediction
                if layers == 1:
                    tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                   n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                   epochs=epoch, batch_size=batch, act_func=activation,
                                                   loss=loss_metric,
                                                   learning_rate=lr, batch_norm=batch_norm, layer_norm=layer_norm,
                                                   weight_norm=weight_norm, kernel=kernels, filters=filters,
                                                   dilations=dilations, padding=padding, dropout=dropout,
                                                   patience=patience, save=saveFile)

                    # Training the model
                    model, history = tcn_model.temperature_model()

                    # validation and train loss to dataframe
                    lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

                    # load best model
                    model = load_model(saveFile, custom_objects={'TCN': TCN})
                    # Test the model and write to file
                    yhat = model.predict(X_test)
                    # predictions to dataframe
                    resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])

                else:
                    tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                   n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                   epochs=epoch, batch_size=batch, act_func=activation,
                                                   loss=loss_metric,
                                                   learning_rate=lr, batch_norm=batch_norm, layer_norm=layer_norm,
                                                   weight_norm=weight_norm, kernel=kernels, filters=filters,
                                                   dilations=dilations, padding=padding, dropout=dropout,
                                                   patience=patience, save=saveFile)

                    # Training the model
                    model, history = tcn_model.temperature_model()

                    # validation and train loss to dataframe
                    lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

                    # load best model
                    model = load_model(saveFile, custom_objects={'TCN': TCN})
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
