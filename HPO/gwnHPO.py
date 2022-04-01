import pandas as pd
import Utils.gwnUtils as util
import torch
import time
from Engine.gwnEngine import trainer
import numpy as np
import Utils.metrics as metrics
import warnings
warnings.filterwarnings("error")


def train_model(args, data, split, supports, adj_init, model_file):
    """
    Trains a GWN model and calculates MSE on validation set for random search HPO. Train and validation sets scaled
    using MinMax normalization. Scaled train and validation data then processed into sliding window input-output pairs.
    Scaled sliding-window data then fed into DataLoaders. GWN model then trained on training data and tested on
    validation data.

    Parameters:
        data_split - Split of data within walk-forward validation.
        args - Parser of parameters.
        train_data - Training data used to train GWN model.
        validate_data - Validation data on which the GWN model is tested.
        lag - length of the input sequence
        forecast - forecasting horizon(length of output), set to 24 hours.
        model_file - File of the best model on the validation set.

    Returns:
        predictions - Returns predictions made by the GWN model on the validation set.
        targets - Returns the target set.
    """

    train_data = data[0]
    validate_data = data[1]

    scaler = util.NormScaler(train_data.min(), train_data.max())

    engine = trainer(scaler, supports, adj_init, args)

    x_train, y_train = util.sliding_window(scaler.transform(train_data), args.lag_length, args.seq_length, split, 0)
    x_validation, y_validation = util.sliding_window(scaler.transform(validate_data), args.lag_length, args.seq_length,
                                                     split, 1)

    trainLoader = util.DataLoader(x_train, y_train, args.batch_size)
    validationLoader = util.DataLoader(x_validation, y_validation, args.batch_size)

    min_val_loss = np.inf
    trainLossArray = []
    validationLossArray = []

    for epoch in range(args.epochs):
        patience = 0
        trainStart = time.time()
        train_loss = engine.train(trainLoader, args)
        trainLossArray.append(train_loss)
        trainTime = time.time() - trainStart

        validationStart = time.time()
        validation_loss = engine.validate(validationLoader, args)
        validationLossArray.append(validation_loss)
        validationTime = time.time() - validationStart

        print(
            'Epoch {:2d} | Train Time: {:4.2f}s | Train Loss: {:5.4f} | Validation Time: {:5.4f} | Validation Loss: '
            '{:5.4f} '.format(epoch + 1, trainTime, train_loss, validationTime, validation_loss))

        if min_val_loss > validation_loss:
            min_val_loss = validation_loss
            patience = 0
            util.save_model(engine.model, model_file)

        else:
            patience += 1

        if patience == args.patience:
            break

    engine.model = util.load_model(model_file)
    testStart = time.time()
    validation_test_loss, predictions, targets = engine.test(validationLoader, args)
    testTime = time.time() - testStart

    print('Inference Time: {:4.2f}s | Loss on validation set: {:5.4f} '.format(
        testTime, validation_test_loss))

    return predictions, targets


def hpo(increment, args):
    """
    Performs random search HPO on the GWN model. Trains a group of GWN models with different hyper-parameters on a train
    set and then tests the models' performance on the validation set. The configuration with the lowest MSE is then
    written to a file.
    Parameters:
        args -  Parser of parameters.
        increment - Walk-forward validation split points.
    """
    data = pd.read_csv(args.data)
    data = data.drop(['StasName', 'DateT'], axis=1)

    textFile = 'HPO/Best Parameters/GWN/configurations.txt'
    f = open(textFile, 'w')

    best_mse = np.inf
    best_cfg = []

    num_splits = 2
    for i in range(args.num_configs):
        config = util.generateRandomParameters(args)
        valid_config = True
        targets = []
        preds = []

        for k in range(num_splits):
            modelFile = 'Garage/HPO Models/GWN/model_split_' + str(k)

            split = [increment[k] * args.n_stations, increment[k + 1] * args.n_stations,
                     increment[k + 2] * args.n_stations]
            data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]

            adj_matrix = util.load_adj(adjFile=args.adjdata, adjtype=args.adjtype)
            supports = [torch.tensor(i).to(args.device) for i in adj_matrix]

            if args.randomadj:
                adjinit = None
            else:
                adjinit = supports[0]
            if args.aptonly:
                supports = None

            torch.manual_seed(0)

            try:
                print('This is the HPO configuration: \n',
                      'Dropout - ', args.dropout, '\n',
                      'Lag_length - ', args.lag_length, '\n',
                      'Hidden Units - ', args.nhid, '\n',
                      'Layers - ', args.num_layers, '\n',
                      'Batch Size - ', args.batch_size, '\n')
                output, real = train_model(args, data_sets, split, supports, adjinit, modelFile)

            except Warning:
                valid_config = False
                break

            targets.append(np.array(real).flatten())
            preds.append(np.array(output).flatten())

        if valid_config:
            mse = metrics.mse(np.concatenate(np.array(targets, dtype=object)),
                              np.concatenate(np.array(preds, dtype=object)))
            if mse < best_mse:
                best_cfg = config
                best_mse = mse

    f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
    f.close()
