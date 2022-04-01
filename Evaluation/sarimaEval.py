import numpy as np
import pandas as pd
import Utils.metrics as metrics


def eval(stations, increment):
    """
     Calculates SARIMA models' performance on the test set. The predictions are read from the results file for each
     station. The targets are compiled from each weather station's data. The MSE, MAE, RMSE, and SMAPE metrics are all
     calculated and the metrics are then written to the outputFile for each weather station.

     Parameters:
         stations - List of the weather stations.
         increment - Walk-forward validation split points.
     """

    for station in stations:
        outputFile = 'Results/SARIMA/Metrics/' + station + '/metrics.txt'
        targetFile = 'Data/Weather Station Data/' + station + '.csv'
        resultsFile = 'Results/SARIMA/Predictions/' + station + '.csv'

        file = open(outputFile, 'w')

        real = pd.read_csv(targetFile)
        realy = real['Temperature']
        targetList = []
        for k in range(27):
            split = [increment[k], increment[k + 1], increment[k + 2]]
            targetList.append(realy[split[1]:split[2]].values)
        targets = np.concatenate(targetList)

        yhat = pd.read_csv(resultsFile)
        yhat = yhat.drop(['Unnamed: 0'], axis=1)
        preds = yhat['predicted_mean']

        mse = metrics.mse(targets, preds)
        mae = metrics.mae(targets, preds)
        rmse = metrics.rmse(targets, preds)
        SMAPE = metrics.smape(targets, preds)

        print('This is the MSE at the ', station, ' weather station: ', mse)
        print('This is the MAE at the ', station, ' weather station: ', mae)
        print('This is the RMSE at the ', station, ' weather station: ', rmse)
        print('This is the SMAPE at the ', station, ' weather station: ', SMAPE)
        print(' ')

        file.write('This is the MSE ' + str(mse) + '\n')
        file.write('This is the MAE ' + str(mae) + '\n')
        file.write('This is the RMSE ' + str(rmse) + '\n')
        file.write('This is the SMAPE ' + str(SMAPE) + '\n')

        file.close()

