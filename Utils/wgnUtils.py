import pandas as pd
import numpy as np
import time
import pickle
from sklearn.utils import shuffle
from numpy import random
import tensorflow as tf

'''
Implementation of several matrix operations that are useful for computing graph convolution
The purpose of many of these functions is to easily support both sparse and dense tensors.
'''


def is_sparse(tensor):
    '''
    check if a given tensor is sparse
    '''
    return isinstance(tensor, tf.SparseTensor)


def is_dense(tensor):
    '''
    check if a given tensor is dense
    '''
    return isinstance(tensor, tf.Tensor)


def matmul(a, b, name=None, transpose_a=False, transpose_b=False):
    '''
    implementation of matrix multiplication that supports dense-dense and dense-sparse multiplication
    '''
    if is_sparse(a) and is_sparse(b):
        raise ValueError('Only a single sparse argument to matmul is supported')
    if is_sparse(a):
        out = tf.compat.v1.sparse_tensor_dense_matmul(a, b, name=name, adjoint_a=transpose_a, adjoint_b=transpose_b)
    elif is_sparse(b):
        out = transpose(tf.compat.v1.sparse_tensor_dense_matmul(b, a, name=name, adjoint_a=(not transpose_b),
                                                                adjoint_b=(not transpose_a)))
    else:
        out = tf.matmul(a, b, name=name, transpose_a=transpose_a, transpose_b=transpose_b)
    return out


def reshape(tensor, shape, name=None):
    '''
    reshapes a tensor. can handle sparse and dense tensors
    '''
    if is_sparse(tensor):
        if name is None:
            name = 'sparse_reshape'
        out = tf.compat.v1.sparse_reshape(tensor, shape, name=name)
    elif is_dense(tensor):
        if name is None:
            name = 'dense_reshape'
        out = tf.reshape(tensor, shape, name=name)
    else:
        raise ValueError('Passed object with invalid type %s' % str(type(tensor)))
    return out


def transpose(tensor, perm=None, name='transpose'):
    '''
    can support sparse and dense transposition
    '''
    if is_sparse(tensor):
        if name is None:
            name = 'sparse_transpose'
        out = tf.compat.v1.sparse_transpose(tensor, perm=perm, name=name)
    elif is_dense(tensor):
        if name is None:
            name = 'dense_transpose'
        out = tf.transpose(tensor, perm=perm, name=name)
    else:
        raise ValueError('Passed object with invalid type %s' % str(type(tensor)))
    return out


def batch_matmul(x, y, batchr=False, batchl=False, shape_dim=3):
    '''
    implementation of batched matrix multiplication that supports dense-dense and dense-sparse multiplication
    '''
    batch_size = tf.compat.v1.get_default_graph().get_tensor_by_name('batch_size:0')
    n_locations = tf.compat.v1.get_default_graph().get_tensor_by_name('n_locations:0')

    if batchr:
        with tf.name_scope('batch_right_matmul'):
            with tf.name_scope('pre_matmul'):
                n_features = tf.gather(tf.shape(y), [2])
                perm = list(range(shape_dim)[1:]) + [0]
                y = transpose(y, perm=perm)
                new_shape = tf.concat([n_locations, n_features * batch_size], axis=0)

                y = reshape(y, new_shape)
            with tf.name_scope('matmul'):
                if isinstance(x, list):
                    out = matmul(x[0], matmul(x[1], y))
                else:
                    out = matmul(x, y)

            with tf.name_scope('post_matmul'):
                new_shape = tf.concat([n_locations, n_features, batch_size], axis=0)
                out = reshape(out, new_shape)

                perm = [shape_dim - 1] + list(range(shape_dim)[:-1])

                out = tf.transpose(out, perm=perm)
    elif batchl:
        with tf.name_scope('batch_left_matmul'):
            with tf.name_scope('pre_matmul'):
                n_features = tf.gather(tf.shape(x), [2])
                new_shape = tf.concat([batch_size * n_locations, n_features], 0)
                x = reshape(x, new_shape)
            with tf.name_scope('matmul'):
                out = matmul(x, y)
            with tf.name_scope('post_matmul'):
                new_n_features = tf.gather(tf.shape(y), [1])
                out_shape = tf.concat([batch_size, n_locations, new_n_features], axis=0)
                out = tf.reshape(out, out_shape)
    return out


def gconv(adj, x, w):
    '''
    computes graph convolution
    '''
    out = batch_matmul(adj, x, batchr=True)
    out = batch_matmul(out, w, batchl=True)
    return out


def construct_feed_dict(x, y, model):
    """
    Construct feed dictionary.
    Used to provide input to the tensorflow model
        x - a batch of predictors
        y - a batch of target
        model - an instance of GraphConvLSTM
    """
    feed_dict = dict()
    feed_dict.update({model.inputs_placeholder: x})
    feed_dict.update({model.outputs_placeholder: y})
    return feed_dict


def get_batches(x, y, n_steps, mask_len, shuffleData=False):
    """
    Given a dataset returns a batch of sequential samples from that dataset using sliding-window technique.
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
        batch_size - batch size

    Returns:
        x_batch - of shape batch_size * n_steps * S * F
        y_batch - of shape batch_size * (n_steps - mask_len) * S
    """
    x_batches = list()
    y_batches = list()
    for i in range(len(x) - n_steps + 1):
        x_batches.append(np.expand_dims(x[i:i + n_steps, :, :], 0))
        y_batches.append(np.expand_dims(y[i:i + n_steps, :], 0))

    if shuffleData:
        x_batches, y_batches = shuffle(x_batches, y_batches, random_state=0)

    x_batch = np.concatenate(x_batches, axis=0)
    y_batch = np.concatenate(y_batches, axis=0)[:, mask_len:, :]

    return x_batch, y_batch, x_batch.shape[0]


def get_random_batch(x, y, n_steps, mask_len, batch_size):
    """
    Given a dataset returns a batch of random samples from that dataset.
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
        batch_size - batch size

    Returns:
        x_batch - of shape batch_size * n_steps * S * F
        y_batch - of shape batch_size * (n_steps - mask_len) * S
    """
    x_batches = list()
    y_batches = list()

    for i in range(batch_size):
        start_ind = np.random.randint(0, x.shape[0] - n_steps - 1)
        x_batches.append(np.expand_dims(x[start_ind:start_ind + n_steps, :, :], 0))
        y_batches.append(np.expand_dims(y[start_ind:start_ind + n_steps, :], 0))
    x_batch = np.concatenate(x_batches, axis=0)
    y_batch = np.concatenate(y_batches, axis=0)[:, mask_len:, :]

    return x_batch, y_batch


def evaluate(x, y, model, sess, n_steps, mask_len):
    """
    Computes the loss on a data set. Can be used to compute the MSE on the validation and test sets
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        model - an instance of GraphConvLSTM
        sess - a tensorflow session
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
    Returns:
        batch_mse - overall batch loss including regularization loss.
        output - list of predictions
        target - list of targets/real values
        adj - adjacency matrix of WGN model
    """
    t_test = time.time()

    x_b, y_b, batches = get_batches(x, y, n_steps, mask_len)

    feed_dict_val = construct_feed_dict(x_b, y_b, model)

    fetches = [model.mse, model.preds, model.outputs_placeholder, model.adj_tensor]

    batch_mse, output, target, adj = sess.run(fetches, feed_dict=feed_dict_val)

    return batch_mse, output, target, time.time() - t_test, adj


def stationFrames():
    """
    Creates a list of DataFrames of each weather station
    Returns:
        frames - list of DataFrames of each weather station.
    """

    stations = ['Atlantis.csv', 'Calvinia WO.csv', 'Cape Columbine.csv', 'Cape Point.csv',
                'Cape Town - Royal Yacht Club.csv', 'Cape Town Slangkop.csv', 'Excelsior Ceres.csv', 'Hermanus.csv',
                'Jonkershoek.csv', 'Kirstenbosch.csv', 'Ladismith.csv', 'Molteno Resevoir.csv', 'Paarl.csv',
                'Porterville.csv', 'Robben Island.csv', 'Robertson.csv', 'SA Astronomical Observatory.csv',
                'Struisbaai.csv', 'Tygerhoek.csv', 'Wellington.csv', 'Worcester AWS.csv']
    frames = []
    for s in stations:
        s = pd.read_csv('Data/Weather Station Data/' + s)
        frames.append(s)
    return frames


def shift(df, forecast):
    """
    Shifts the original weather data up by the length of the output sequence(horecast horizon) to convert it into
    time-series data

    Parameters:
        df - DataFrame of weather station data
        forecast - forecasting horizon (length of output sequence)
    Returns:
        df - dataframe with output feature shifted up by forecasting horizon.
    """

    temp = df['Temperature'].shift(-forecast)
    df['Target'] = temp
    df = df.dropna()
    return df


def prepWGNGraphData(forecast):
    """
    Iterates through list of DataFrames of weather stations, shifts the target temperature values up by the forecasting
    horizon(length of output sequence). Then concatenates all the DataFrames into one DataFrame, and orders them
    chronologically. The final frame that is returned is of the shape (N * T * S * F), where N=number of samples,
    T=number_of_time_steps, S=number_of_stations, F=features.

    Parameters:
        forecast - length of output sequence(forecasting horizon)

    Returns:
        cat - graph weather data for WGN model
    """

    weather_stations = stationFrames()
    adjustedStations = []
    for station in weather_stations:
        adjustedStations.append(shift(station, forecast))
    cat = pd.concat(adjustedStations)
    cat = cat.drop(['Latitude', 'Longitude'], axis=1)
    cat = cat.sort_values(['DateT', 'StasName'], ascending=[True, True])
    cat.to_csv('Data/Graph Neural Network Data/Graph Station Data/Graph_wgn.csv', index=False)
    pd.to_pickle(cat, 'Data/Graph Neural Network Data/Graph Station Data/GraphWGN.pkl')
    return cat

def load_data(forecast, split, adjFile):
    """
    Preprocess and loads the train, validation, and test sets for the WGN model. Adjacency file is read from pickle
    file, the train, validation, and test sets are scaled using MinMax scaling with min=train.min() and max=train.max().
    The train, validation, and test sets are then reshaped into input data of shape (N * S * F) and output data of shape
    (N * S) where N=number_of_samples, S=num_stations and F=num_features

    Parameters:
        forecast - length of output sequence(forecasting horizon)

    Returns:
        adj - adjacency matrix
        X_train - x train set
        Y_train - y train set
        X_val - x validation set
        Y_val - y validation set
        X_test - x test set
        Y_test - y test set
    """

    graphData = prepWGNGraphData(forecast)
    graphData = graphData.drop(['DateT', 'StasName'], axis=1)
    graphData = graphData[['Temperature', 'Rain', 'WindSpeed', 'WindDir', 'Pressure', 'Humidity', 'Target']]

    with open(adjFile, 'rb') as f:
        data = pickle.load(f)

    adj = data

    train_len = split[0]
    val_len = split[1]
    test_len = split[2]

    train = graphData[0:train_len]

    min = train.min()
    max = train.max()

    x_train = (train - min) / (max - min)
    y_train = x_train['Target']
    x_train = x_train.drop(['Target'], axis=1)
    X_train = np.reshape(x_train.values, (int(train_len / 21), 21, 6))
    Y_train = np.reshape(y_train.values, (int(train_len / 21), 21))

    validation = graphData[train_len:val_len]
    x_val = (validation - min) / (max - min)
    y_val = x_val['Target']
    x_val = x_val.drop(['Target'], axis=1)
    X_val = np.reshape(x_val.values, (int((val_len - train_len) / 21), 21, 6))
    Y_val = np.reshape(y_val.values, (int((val_len - train_len) / 21), 21))

    test = graphData[val_len:test_len]
    x_test = (test - min) / (max - min)
    y_test = x_test['Target']
    x_test = x_test.drop(['Target'], axis=1)
    X_test = np.reshape(x_test.values, (int((test_len - val_len) / 21), 21, 6))
    Y_test = np.reshape(y_test.values, (int((test_len - val_len) / 21), 21))

    return adj, X_train, Y_train, X_val, Y_val, X_test, Y_test


def generateRandomParameters(args):
    """
    Generates a random configuration of hyper-parameters from the pre-defined search space.

    Returns:
        configuration_list - Returns randomly selected configuration of GWN hyper-parameters.
    """

    batch_size = [32, 64, 125]
    layers = [1, 2, 3, 4]
    hidden_units = [8, 10, 15, 20]
    batches = [10000, 15000, 20000]
    lag_length = [24, 48, 72]
    mask_len = [0, 20, 30]

    batch = batch_size[random.randint(len(batch_size))]
    num_layers = layers[random.randint(len(layers))]
    units = hidden_units[random.randint(len(hidden_units))]
    numbatches = batches[random.randint(len(batches))]
    maskPos = random.randint(len(lag_length))
    lag = lag_length[maskPos]
    mask = mask_len[maskPos]

    args.nbatch_size = batch
    args.nbatches = numbatches
    args.nlayers = num_layers
    args.nhidden = units
    args.nsteps = lag
    args.mask_len = mask

    return [batch, numbatches, num_layers, units, lag, num_layers]
