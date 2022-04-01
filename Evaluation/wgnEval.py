import Utils.gwnUtils as utils
import Utils.metrics as metrics
import numpy as np


def eval(stations, args):
    """
     Calculates the WGN model's performance on the test set across all forecasting horizons[3, 6, 9, 12, 24] for each
     individual station. The predictions are read from the results file for each split of the walk-forward validation
     method. The predictions from each split are appended into one long list of predictions. The targets are pulled from
     targets file in the GWN results directory. The MSE, MAE, RMSE, and SMAPE metrics are then calculated and written
     to the metric files.

     Parameters:
         stations - List of the weather stations.
         args - Parser of parameter arguments.
     """
    num_splits = 27
    num_stations = 21
    for station in range(num_stations):

        for horizon in [3, 6, 9, 12, 24]:

            pred = []
            real = []

            for split in range(num_splits):
                resultsFile = 'Results/WGN/' + str(horizon) + ' Hour Forecast/Predictions/outputs_' + str(
                    split) + '.pkl'
                targetsFile = 'Results/WGN/' + str(horizon) + ' Hour Forecast/Targets/targets_' + str(split) + '.pkl'
                yhat = utils.load_pickle(resultsFile)
                target = utils.load_pickle(targetsFile)
                pred.extend(np.array(yhat).flatten())
                real.extend(np.array(target).flatten())

            pred = np.array(pred).reshape((int(len(real) / (args.n_stations * (args.nsteps - args.mask_len))),
                                           args.nsteps - args.mask_len, args.n_stations))
            real = np.array(real).reshape((int(len(real) / (args.n_stations * (args.nsteps - args.mask_len))),
                                           args.nsteps - args.mask_len, args.n_stations))

            metricFile = 'Results/WGN/Metrics/' + stations[station] + '/metrics_' + str(horizon)
            file = open(metricFile, 'w')

            preds = pred[:, :, station]
            real_values = real[:, :, station]

            root = metrics.rmse(real_values, preds)
            square = metrics.mse(real_values, preds)
            abs = metrics.mae(real_values, preds)
            ape = metrics.smape(real_values, preds)

            print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station, horizon))
            print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station, horizon))
            print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs, station, horizon))
            print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station, horizon))
            print(' ')

            file.write('This is the MSE ' + str(square) + '\n')
            file.write('This is the MAE ' + str(abs) + '\n')
            file.write('This is the RMSE ' + str(root) + '\n')
            file.write('This is the SMAPE ' + str(ape) + '\n')

            file.close()
