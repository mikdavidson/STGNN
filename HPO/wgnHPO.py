import time
import Utils.wgnUtils as util
import numpy as np
import Models.wgc as gconv
import tensorflow as tf
from numpy import random
import Utils.metrics as metrics


def train_model(sess, args, split):
    """
    Trains a WGN model and calculates MSE on validation set for random search HPO. Train and validation sets scaled using
    MinMax normalization. Scaled train and validation data then processed into sliding window input-output pairs. Scaled
    sliding-window data then used to train and test WGN model. WGN model then trained on training data and tested on
    validation data.

    Parameters:
        sess - Tensorflow session.
        args - Parser of parameters.
        split - Walk-forward validation split points in data sets.
        model_file - File that models are saved to.
    Returns:
        output - Returns predictions made by the GWN model on the validation set.
        targets - Returns the target set.
    """
    seed = 123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.compat.v1.disable_eager_execution()
    adj, x_train, y_train, x_val, y_val, x_test, y_test = util.load_data(args.forecast, split,
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

    denom = 0.
    batches_complete = sess.run(model.global_step)
    best_output = []
    best_target = []

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

        # Periodically compute validation loss
        if batches_complete % args.display_step == 0 or batches_complete == args.nbatches:
            # Validation
            val_mse, output, target, duration, val_adj = util.evaluate(x_val, y_val, model, sess, args.nsteps,
                                                                       args.mask_len)

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
                best_output = output
                best_target = target

            if (batches_complete - best_batch > args.between_lr_updates) and (
                    batches_complete - last_lr_update > args.between_lr_updates):
                learningRate = learningRate * args.lr_factor
                last_lr_update = batches_complete

    tf.compat.v1.reset_default_graph()

    return best_target, best_output


def hpo(increment, args):
    """
    Performs random search HPO on the GWN model. Trains a group of GWN models with different hyper-parameters on a train
    set and then tests the models' performance on the validation set. The configuration with the lowest MSE is then
    written to a file.
    Parameters:
        args -  Parser of parameters.
        increment - Walk-forward validation split points.
    """

    textFile = 'HPO/Best Parameters/WGN/configurations.txt'
    f = open(textFile, 'w')

    best_mse = np.inf
    best_cfg = []
    num_splits = 2
    for i in range(args.num_configs):
        cfg = util.generateRandomParameters(args)

        print('This is the HPO configuration: \n',
              'Batch Size - ', args.nbatch_size, '\n',
              'Hidden Units - ', args.nhidden, '\n',
              'Lag Length - ', args.nsteps, '\n',
              'Mask Length - ', args.mask_len, '\n',
              'Layers - ', args.nlayers, '\n',
              'Num Batches to train on - ', args.nbatches, '\n')
        targets = []
        preds = []
        for k in range(num_splits):
            job_ppn = 4
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=job_ppn,
                                              inter_op_parallelism_threads=job_ppn - 2,
                                              allow_soft_placement=True, device_count={'GPU': 1})
            sess = tf.compat.v1.Session(config=config)

            split = [increment[k] * args.n_stations, increment[k + 1] * args.n_stations,
                     increment[k + 2] * args.n_stations]

            real, output = train_model(sess, args, split)

            targets.append(np.array(real).flatten())
            preds.append(np.array(output).flatten())

        targets = np.concatenate([targets[0], targets[1]])
        preds = np.concatenate([preds[0], preds[1]])
        mse = metrics.mse(np.array(targets), np.array(preds))

        print('This is the mse over ', num_splits, ' splits ', mse)

        if mse < best_mse:
            best_cfg = cfg
            best_mse = mse

    f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
    f.close()
