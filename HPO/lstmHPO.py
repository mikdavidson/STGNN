import numpy as np
import Utils.metrics as metrics
import Utils.baselineUtils as utils
import Models.lstm as lstm
from keras.models import load_model


def hpo(stations, increment, args):
    """
    Performs random search HPO on the LSTM model. Trains a group of LSTM models for each weather station with different
    hyper-parameters on a train set and then tests the models' performance on the validation set. The configuration with
    the lowest MSE for that station is then written to a file.

    Parameters:
        stations - List of weather stations.
        args -  Parser of parameters.
        increment - Walk-forward validation split points.
    """

    num_splits = 2
    for station in stations:
        # printing out which station we are forecasting
        print('Performing LSTM random search HPO at station: ', station)
        # pulling in weather station data
        weatherData = 'Data/Weather Station Data/' + station + '.csv'
        ts = utils.create_dataset(weatherData)

        textFile = 'HPO/Best Parameters/LSTM/' + station + '_configurations.txt'
        f = open(textFile, 'w')
        best_mse = 999999
        best_cfg = []

        for i in range(args.num_configs):
            # generate random parameters
            cfg = utils.generateRandomLSTMParameters()

            # dataframes of results and targets
            resultsArray = np.array([])
            targetArray = np.array([])
            print('This is the configuration: ', cfg)

            for k in range(num_splits):
                saveFile = 'Garage/HPO Models/LSTM/' + station + '/Best_Model_24' + '_walk_' + str(k) + '.h5'
                split = [increment[k], increment[k + 1], increment[k + 2]]
                pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)

                # Scaling the data
                train, validation, test = utils.min_max(pre_standardize_train,
                                                        pre_standardize_validation,
                                                        pre_standardize_test)

                # Defining input shape
                n_ft = train.shape[1]

                # Creating the X and Y for forecasting
                X_train, Y_train = utils.create_X_Y(train, cfg['Lag'], cfg['Forecast Horizon'])

                # Creating the X and Y for validation set
                X_val, Y_val = utils.create_X_Y(validation, cfg['Lag'], cfg['Forecast Horizon'])

                # Creating the lstm model for temperature prediction
                lstm_model = lstm.lstm(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                       n_lag=cfg['Lag'], n_features=n_ft, n_ahead=cfg['Forecast Horizon'],
                                       epochs=cfg['Epochs'], batch_size=cfg['Batch Size'], act_func=cfg['Activation'],
                                       loss=cfg['Loss'], learning_rate=cfg['lr'], dropout=cfg['Dropout'],
                                       patience=cfg['Patience'], units=cfg['Units'], l1=cfg['l1'], l2=cfg['l2'],
                                       bias_reg=cfg['Bias'], save=saveFile)

                # Training the model
                model, history = lstm_model.temperature_model()
                # load best model
                model = load_model(saveFile)
                # Test the model and write to file
                yhat = model.predict(X_val)
                # predictions to dataframe
                resultsArray = np.concatenate((resultsArray, np.array(yhat.reshape(-1, ))))
                # Targets to dataframe
                targetArray = np.concatenate((targetArray, np.array(Y_val.reshape(-1, ))))

            # calculate metrics
            ave_mse = metrics.mse(targetArray, resultsArray)
            print('This is the MSE over 2 splits: ', ave_mse)
            f.write('Configuration parameters at station ' + station + ': ' + str(cfg) + ' with MSE =' +
                    str(ave_mse) + '\n')

            if ave_mse < best_mse:
                best_mse = ave_mse
                best_cfg = cfg

        f.write('Best parameters found at station ' + station + ': ' + str(best_cfg) + ' with MSE =' + str(best_mse) +
                '\n')
        f.close()
