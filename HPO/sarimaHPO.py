from statsmodels.tsa.statespace.sarimax import SARIMAX
import Utils.metrics as utils
from numpy import random
import warnings
import numpy as np
import pandas as pd


def train(train_set, configuration):
    """
    Uses a SARIMA model to output a set of predictions as long as the validation set.

    Parameters:

        train - Train data set.
        configuration -  Configuration of SARIMA model.

    Returns:
        model - Returns a SARIMA model trained on the train data set.
    """

    order, sorder = configuration
    model = SARIMAX(train_set, order=order, seasonal_order=sorder)
    model = model.fit(disp=False, low_memory=True)
    return model


def validate(validation, sarimax_model):
    """
    Uses a SARIMA model to output a set of predictions as long as the validation set.

    Parameters:
        validation - Validation data set.
        sarimax_model -  SARIMA model used to predict temperature.

    Returns:
        yhat - List of predictions.
    """

    yhat = sarimax_model.predict(0, len(validation) - 1)
    return yhat


def generateRandomParamters():
    """
    Generates a random configuration of hyper-parameters from the pre-defined search space.

    Returns:
        configuration_list - Returns randomly selected configuration of LSTM hyper-parameters.
    """

    d = 0
    m = 24
    p_params = [0, 1, 2, 3]
    q_params = [0, 1, 2, 3]
    P_params = [0, 1, 2, 3]
    D_params = [0, 1]
    Q_params = [0, 1, 2, 3]
    p = random.randint(len(p_params))
    q = random.randint(len(q_params))
    P = random.randint(len(P_params))
    D = random.randint(len(D_params))
    Q = random.randint(len(Q_params))
    cfg = [(p, d, q), (P, D, Q, m)]
    return cfg


def hpo(stations, increment, num_configs):
    """
    Performs random search HPO on the SARIMA model. Trains a group of SARIMA models for each weather station with
    different hyper-parameters on a train set and then tests the models' performance on the validation set. The
    configuration with the lowest MSE for that station is then written to a file.

    Parameters:
        stations - List of weather satations.
        args -  Parser of parameters.
        increment - Walk-forward validation split points.
    """

    warnings.filterwarnings("error")

    for station in stations:

        print('Performing SARIMA random search HPO at station:', station)
        weatherData = 'Data/Weather Station Data/' + station + '.csv'
        textFile = 'HPO/Best Parameters/SARIMA/' + station + '_configurations.txt'
        f = open(textFile, 'w')
        df = pd.read_csv(weatherData)
        temp = df['Temperature']
        best_mse = 999999
        best_config = []

        for i in range(num_configs):

            config = generateRandomParamters()
            target = []
            results = []
            valid_model = True
            print('This is the configuration: ', config)

            for k in range(2):

                split = [increment[k], increment[k + 1]]
                train_data = temp[:split[0]]
                validation_data = temp[split[0]:split[1]]
                target.append(validation_data.values)

                try:
                    sarima_model = train(train_data, config)
                    predictions = validate(validation_data, sarima_model)
                    results.append(predictions)

                except Warning:
                    print('This configuration does not converge.')
                    valid_model = False
                    break

            if valid_model:
                targets = np.concatenate([i for i in target])
                preds = np.concatenate([i for i in results])
                ave_mse = utils.mse(targets, preds)
                print('This is the MSE over 2 splits: ', ave_mse)
                f.write('Configuration parameters at station ' + station + ': ' + str(config) + 'with MSE =' +
                        str(ave_mse) + '\n')
                if ave_mse < best_mse:
                    best_mse = ave_mse
                    best_config = config

        f.write('Best parameters found at station ' + station + ': ' + str(best_config) + 'with MSE =' + str(best_mse) +
                '\n')
        f.close()
