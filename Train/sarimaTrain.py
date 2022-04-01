from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


# one-step sarima forecast
def sarima_forecast(history, config):
    """
    Fits and trains a SARIMA model on the train data set using the optimal configuration found through random search
    HPO.

    Parameters:
        history - train data set.
        config - optimal SARIMA configuration for each weather station

    Returns:
        model_fit - Returns model fit on train data.
    """
    order, sorder = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder)
    # fit model
    model_fit = model.fit(disp=False, low_memory=True)
    return model_fit


def train_model(sets, data, cfg, resultsDF):
    """
    Uses a trained SARIMA model to predict a number of values that is the same length as the test set.

    Parameters:
        sets - set of data points of train, validation and test set.
        data - train, validation, and test data
        cfg - optimal SARIMA configuration for each weather station
        resultsDF - DataFrame to write predictions to.

    Returns:
        resDF -  DataFrame of predictions.
    """
    model = sarima_forecast(data[:sets[0]], cfg)
    resDF = test(model, sets, resultsDF)
    return resDF


def test(model, sets, resDF):
    """
    Uses a trained SARIMA model to predict a number of values that is the same length as the test set.

    Parameters:
        model -  SARIMA model used to make the predictions.
        sets - set of data points of train, validation and test set.
        resDF - DataFrame to write predictions to.

    Returns:
        resDF -  DataFrame of predictions.
    """
    yhat = model.predict(sets[1], sets[2]-1)
    resDF = pd.concat([resDF, yhat.to_frame()])
    return resDF


def stringtoCfg(params):
    """
    Creates a configuration of the optimal hyper-parameters for each weather station.

    Parameters:
        params -  String of optimal hyper-parameters.

    Returns:
        config -  List of hyper-parameters for the SARIMA model.
    """

    string = params.split(" ")
    config = [(int(string[0]), int(string[1]), int(string[2])),
              (int(string[3]), int(string[4]), int(string[5]), int(string[6]))]
    return config


def train(stations, increment):
    """
    Trains and tests SARIMA models for each station using walk-forward validation across 27 splits using the optimal
    parameters found through random search HPO.

    Parameters:
        stations -  List of weather stations.
        increment - Walk-forward validation split points.
    """

    # configFile = open("SARIMA/Hyperparameter/params.txt", "r")
    configFile = open("Train/Best Configurations/sarima_params.txt", "r")
    for station in stations:
        print('SARIMA training and predicting at station: ', station)
        file = "Results/SARIMA/Predictions/" + station + '.csv'
        resultsDF = pd.DataFrame()
        params = configFile.readline()
        cfg = stringtoCfg(params)
        weatherData = 'Data/Weather Station Data/' + station + '.csv'
        df = pd.read_csv(weatherData)
        temp = df['Temperature']
        for k in range(27):
            print('Training on split ', k)
            split = [increment[k], increment[k + 1], increment[k + 2]]
            resultsDF = train_model(split, temp, cfg, resultsDF)

        resultsDF.to_csv(file)
