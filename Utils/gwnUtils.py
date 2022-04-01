import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import pandas as pd
from Models.graphWaveNet import *
from numpy import random


class DataLoader(object):
    """
    Creates a dataloader class that provides methods to shuffle and iterate through the data.
    """

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs: input
        :param ys: output
        :param batch_size: size of batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Creates a a scaler object that performs z-score normalisation on a data set
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NormScaler:
    """
    Creates a scaler object that performs MinMax scaling on a data set
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """

    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_adj(adjFile, adjtype):
    """
    Loads adjacency matrix and calculates the type of matrix depending on adjtype

    Parameters:
        adjFile - file to read adjacency matrix from
        adjtype - type of adjacency matrix to use

    Returns:
        adj - returns adjacency matrix
    """

    adj_mx = load_pickle(adjFile)
    adj_mx = adj_mx.values

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


def load_pickle(pickle_file):
    """
    Loads data from a pickle file

    Parameters:
        pickle_file - file to read
    Returns:
        pickle_data - returns data read from pickle file
    """

    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_adj_matrix(model):
    """
    Gets adjacency matrix from GWN model

    Parameters:
        model - trained GWN model

    Returns:
        df - dataframe of adjacency matrix
    """

    adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    device = torch.device('cpu')
    adp.to(device)
    adp = adp.cpu().detach().numpy()
    adp = adp * (1 / np.max(adp))
    df = pd.DataFrame(adp)
    return df


def save_model(model, file_name):
    """
    Saves a pytoch GWN model

    Parameters:
        model - trained GWN model
        file_name - name of saved model file
    """

    torch.save(model, file_name)


def load_model(file_name):
    """
    Loads a pytorch GWN model

    Parameters:
        file_name - file from which to load trained GWN model

    Returns:
        model - returns loaded model from file
    """

    model = torch.load(file_name)
    return model



def sliding_window(df, lag, forecast, split, set):
    """
    Converts array to times-series input-output sliding-window pairs.
    Parameters:
        df - DataFrame of weather station data
        lag - length of input sequence
        forecast - length of output sequence(forecasting horizon)
        split - points at which to split data into train, validation, and test sets
        set - indicates if df is train, validation, or test set
    Returns:
        x, y - returns x input and y output
    """
    if set == 0:
        samples = int(split[0] / 21)
    if set == 1:
        samples = int(split[1] / 21 - split[0] / 21)
    if set == 2:
        samples = int(split[2] / 21 - split[1] / 21)


    dfy = df.drop(['Rain', 'Humidity', 'Pressure', 'WindSpeed', 'WindDir'], axis=1)
    stations = 21
    features = 6

    df = df.values.reshape(samples, stations, features)
    dfy = dfy.values.reshape(samples, stations, 1)

    x_offsets = np.sort(np.concatenate((np.arange(-(lag - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (forecast + 1), 1))

    data = np.expand_dims(df, axis=-1)
    data = data.reshape(samples, 1, stations, features)
    data = np.concatenate([data], axis=-1)

    datay = np.expand_dims(dfy, axis=-1)
    datay = datay.reshape(samples, 1, stations, 1)
    datay = np.concatenate([datay], axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))  # Exclusive

    # t is the index of the last observation.
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(datay[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x = np.squeeze(x)
    y = np.squeeze(y, axis=2)
    return x, y


def generateRandomParameters(args):
    """
    Generates a random configuration of hyper-parameters from the pre-defined search space.

    Parameters:
        args -  Parser of parameters.

    Returns:
        config - list of HPO parameters
    """

    batch_size = [32, 64]
    hidden_units = [22, 32, 42]
    lag_length = [12, 24]
    dropout = [0.1, 0.2, 0.3]
    layers = [2, 3, 4]
    epochs = [30, 40, 50, 60]

    batch = batch_size[random.randint(len(batch_size))]
    units = hidden_units[random.randint(len(hidden_units))]
    lag = lag_length[random.randint(len(lag_length))]
    dropout = dropout[random.randint(len(dropout))]
    num_layers = layers[random.randint(len(layers))]
    epoch = epochs[random.randint(len(epochs))]

    args.batch_size = batch
    args.dropout = dropout
    args.nhid = units
    args.lag_length = lag
    args.num_layers = num_layers
    args.epochs = epoch

    return [batch, dropout, units, lag, num_layers, epoch]
