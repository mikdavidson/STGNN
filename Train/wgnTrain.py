import time
import Utils.wgnUtils as util
import numpy as np
import Models.wgc as gconv
import tensorflow as tf
import pickle


def train_model(sess, args, forecast, split, preds_file, target_file, matrix_file,
                model_file):
    """
    Trains a WGN model and calculates MSE on test set. Train, validation, and test sets scaled using
    MinMax normalization. Scaled train, validation, and test data then shifted into time-series input-output pairs.
    WGN model instantiated, then trained on nbatches of train data, then best model tested on test data. Predictions,
    targets, and adjacency matrices for each of the 27 splits are then written to files.

    Parameters:
        sess - Tensorflow session.
        args - Parser of parameters.
        split - Split of data within walk-forward validation.
        preds_file - File to write predictions of GWN model to.
        target_file - File to write targets of test set to.
        model_file - File of the best model on the validation set.
    """

    seed = 123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.compat.v1.disable_eager_execution()
    adj, x_train, y_train, x_val, y_val, x_test, y_test = util.load_data(forecast, split,
                                                                         args.adjdata)

    # Build model
    model = gconv.GraphConvLSTM(adj=adj, n_stations=args.n_stations, n_features=args.features,
                                num_layers=args.nlayers, n_steps=args.nsteps,
                                n_hidden=args.nhidden,
                                adj_trainable=args.adjTrainable,
                                use_sparse=args.use_sparse, mask_len=args.mask_len,
                                learning_rate=args.learningRate, rank=args.rank)

    init = tf.compat.v1.global_variables_initializer()

    # Initialize tensorflow variables
    sess.run(init)

    train_mse = 0
    best_val = 99999999.
    best_batch = 0
    last_lr_update = 0

    learningRate = sess.run(model.learning_rate_variable)

    cost_val = []

    denom = 0.
    batches_complete = sess.run(model.global_step)

    while batches_complete < args.nbatches:
        x_train_b, y_train_b = util.get_random_batch(x_train, y_train, args.nsteps, args.mask_len, args.nbatch_size)
        t = time.time()
        # Construct feed dictionary
        feed_dict = util.construct_feed_dict(x_train_b, y_train_b, model)
        feed_dict[model.learning_rate_variable] = learningRate
        # Training step
        _, batch_mse, batches_complete = sess.run([model.opt_op, model.mse, model.global_step], feed_dict=feed_dict)
        train_mse += batch_mse

        batch_time = time.time() - t
        denom += 1

        # Periodically compute validation and test loss
        if batches_complete % args.display_step == 0 or batches_complete == args.nbatches:
            # Validation
            val_mse, output, target, duration, val_adj = util.evaluate(x_val, y_val, model, sess, args.nsteps,
                                                                       args.mask_len)
            cost_val.append(val_mse)

            # Print results
            print(
                "Batch Number:%04d" % batches_complete,
                "train_mse={:.5f}".format(train_mse / denom),
                "val_mse={:.5f}".format(val_mse),
                "time={:.5f}".format(batch_time),
                "lr={:.8f}".format(learningRate))
            train_mse = 0
            denom = 0.

            # Check if val loss is the best encountered so far
            if val_mse < best_val:
                best_val = val_mse
                best_batch = batches_complete - 1
                patience = 0
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, model_file)

            else:
                patience += 1

            if patience == args.patience:
                break

            if (batches_complete - best_batch > args.between_lr_updates) and (
                    batches_complete - last_lr_update > args.between_lr_updates):
                learningRate = learningRate * args.lr_factor
                last_lr_update = batches_complete

    # load the model here
    new_saver = tf.compat.v1.train.import_meta_graph(model_file + '.meta')
    new_saver.restore(sess, model_file)
    test_mse, output, target, duration, test_adj = util.evaluate(x_test, y_test, model, sess, args.nsteps,
                                                                 args.mask_len)

    outputPkl = open(preds_file, 'wb')
    pickle.dump(output, outputPkl)
    outputPkl.close()

    targetPkl = open(target_file, 'wb')
    pickle.dump(target, targetPkl)
    targetPkl.close()

    adjPkl = open(matrix_file, 'wb')
    pickle.dump(test_adj, adjPkl)
    adjPkl.close()

    print('best val mse: {0}, test mse: {1}'.format(best_val, test_mse))
    tf.compat.v1.reset_default_graph()


def train(increment, args):
    """
    Trains and tests the final WGN model through walk-forward validation across all 5 foreacasting horizons.
    A WGN model, with the same parameters, is trained across 27 splits. The predictions, targets, losses and adjacency
    matrices for each split are written to files for evaluation.

    Parameters:
        args -  Parser of parameters.
        increment - Walk-forward validation split points.
    """
    forecast_horizons = [3, 6, 9, 12, 24]

    for i in forecast_horizons:
        forecast_horizon = i
        print('Training WGN models through walk-forward validation on a forecasting horizon of: {0}'.format(forecast_horizon))
        for k in range(args.n_split):
            print('Training WGN model on split: {0}'.format(k))
            job_ppn = 4
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=job_ppn,
                                              inter_op_parallelism_threads=job_ppn - 2,
                                              allow_soft_placement=True, device_count={'GPU': 1})
            sess = tf.compat.v1.Session(config=config)

            predFile = 'Results/WGN/' + str(i) + ' Hour Forecast/Predictions/outputs_' + str(k) + '.pkl'
            targetFile = 'Results/WGN/' + str(i) + ' Hour Forecast/Targets/' + 'targets_' + str(k) + '.pkl'
            matrixFile = 'Results/WGN/' + str(i) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv'
            modelFile = 'Garage/Final Models/WGN/' + str(i) + ' Hour Models/model_split_' + str(k)

            split = [increment[k] * args.n_stations, increment[k + 1] * args.n_stations,
                     increment[k + 2] * args.n_stations]
            train_model(sess, args, forecast_horizon, split, predFile, targetFile, matrixFile, modelFile)
