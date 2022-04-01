import argparse
import HPO.sarimaHPO as sarimaHPO
import HPO.lstmHPO as lstmHPO
import HPO.tcnHPO as tcnHPO
import HPO.gwnHPO as gwnHPO
import HPO.wgnHPO as wgnHPO

import Train.sarimaTrain as sarimaTrain
import Train.lstmTrain as lstmTrain
import Train.tcnTrain as tcnTrain
import Train.gwnTrain as gwnTrain
import Train.wgnTrain as wgnTrain

import Evaluation.sarimaEval as sarimaEval
import Evaluation.baselineEval as baselineEval
import Evaluation.gwnEval as gwnEval
import Evaluation.wgnEval as wgnEval

parser = argparse.ArgumentParser()

# Random Search HPO arguments
parser.add_argument('--num_configs', type=int, default=30, help='number of random configurations to search through')
parser.add_argument('--tune_lstm', type=bool, help='whether to perform random search HPO on LSTM models')
parser.add_argument('--tune_sarima', type=bool, help='whether to perform random search HPO on SARIMA models')
parser.add_argument('--tune_tcn', type=bool, help='whether to perform random search HPO on TCN models')
parser.add_argument('--tune_gwn', type=bool, help='whether to perform random search HPO on GWN model')
parser.add_argument('--tune_wgn', type=bool, help='whether to perform random search HPO on WGN model')

# Train final baseline and GNN model arguments
parser.add_argument('--train_sarima', type=bool, help='whether to train final sarima models')
parser.add_argument('--train_lstm', type=bool, help='whether to train final LSTM models')
parser.add_argument('--train_tcn', type=bool, help='whether to train final TCN models')
parser.add_argument('--train_gwn', type=bool, help='whether to train final GWN model')
parser.add_argument('--train_wgn', type=bool, help='whether to train final WGN model')

# Calculate metrics of final models' results arguments
parser.add_argument('--eval_sarima', type=bool, help='whether to report final sarima metrics')
parser.add_argument('--eval_lstm', type=bool, help='whether to report final lstm metrics')
parser.add_argument('--eval_tcn', type=bool, help='whether to report final tcn metrics')
parser.add_argument('--eval_gwn', type=bool, help='whether to report final gwn metrics')
parser.add_argument('--eval_wgn', type=bool, help='whether to report final wgn metrics')

parser.add_argument('--n_stations', type=int, default=21, help='number of weather stations')
parser.add_argument('--n_split', type=int, default=27, help='number of splits in walk-forward validation')

# Graph WaveNet arguments, default arguments are optimal hyper-parameters
parser.add_argument('--device', type=str, default='cuda', help='device to place model on')
parser.add_argument('--data', type=str, default='Data/Graph Neural Network Data/Graph Station Data/Graph.csv',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='Data/Graph Neural Network Data/Adjacency Matrix/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly', type=bool, default=True, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=bool, default=True, help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=24, help='length of output sequence')
parser.add_argument('--lag_length', type=int, default=12, help='length of input sequence')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=6, help='number of features in input')
parser.add_argument('--num_nodes', type=int, default=21, help='number of nodes in graph(num weather stations)')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--patience', type=int, default=9, help='patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=30, help='')
parser.add_argument('--save', type=str, default='Garage/Final Models/GWN/', help='save path')

# Low Rank-Weighted GNN arguments, default arguments are optimal hyper-parameters
parser.add_argument('--adjTrainable', type=bool, default=True, help='adjacency matrix a trainable parameter')
parser.add_argument('--use_sparse', type=bool, default=False, help='Use sparse adajcency matrix')
parser.add_argument('--nlayers', type=int, default=2, help='number of gconv layers')
parser.add_argument('--features', type=int, default=6, help='number of features')
parser.add_argument('--nbatch_size', type=int, default=32, help='batch size')
parser.add_argument('--nbatches', type=int, default=15000, help='number of batches to train on')
parser.add_argument('--nhidden', type=int, default=8, help='')
parser.add_argument('--nsteps', type=int, default=24, help='length of input sequence')
parser.add_argument('--mask_len', type=int, default=0, help='length of input to be used as context')
parser.add_argument('--forecast', type=int, default=24, help='how many steps into future to forecast')
parser.add_argument('--rank', type=int, default=None, help='rank of adjacency matrix')
parser.add_argument('--display_step', type=int, default=500, help='display metrics after number of batches')
parser.add_argument('--between_lr_updates', type=int, default=500, help='display metrics after number of batches')
parser.add_argument('--learningRate', type=float, default=0.01, help='display metrics after number of batches')
parser.add_argument('--lr_factor', type=float, default=0.9, help='display metrics after number of batches')
args = parser.parse_args()

if __name__ == '__main__':

    # list of all weather stations
    stations = ['Atlantis', 'Calvinia WO', 'Cape Columbine', 'Cape Point',
                'Cape Town - Royal Yacht Club', 'Cape Town Slangkop', 'Excelsior Ceres', 'Hermanus',
                'Jonkershoek', 'Kirstenbosch', 'Ladismith', 'Molteno Resevoir', 'Paarl',
                'Porterville', 'Robben Island', 'Robertson', 'SA Astronomical Observatory',
                'Struisbaai', 'Tygerhoek', 'Wellington', 'Worcester AWS']

    """
    List of points to split data in train, validation, and test sets for walk-forward validation. The first marker, 
    8784 is one year's worth of data, the next step is 3 months of data, and the following step is also 3 months of 
    data, resulting in rolling walk-forward validation where the train size increases each increment, with the 
    validation and test sets each being 3 months' worth of data.
    """
    increment = [8784, 10944, 13128, 15336, 17544, 19704, 21888,
                 24096, 26304, 28464, 30648, 32856, 35064, 37248,
                 39432, 41640, 43848, 46008, 48192, 50400, 52608,
                 54768, 56952, 59160, 61368, 63528, 65712, 67920, 70128]

    """
    Adjusted increment list seen above for WGN model. A number of steps are removed when shifting the time-series up
    to process the data into input-output pairs. args.forecast = forecast length(24 hours).
    """
    wgn_increment = [8784, 10944, 13128, 15336, 17544, 19704, 21888,
                     24096, 26304, 28464, 30648, 32856, 35064, 37248,
                     39432, 41640, 43848, 46008, 48192, 50400, 52608,
                     54768, 56952, 59160, 61368, 63528, 65712, 67920, 70128 - args.forecast]

    # Random search SARIMA
    if args.tune_sarima:
        sarimaHPO.hpo(stations, increment, args.num_configs)

    # Random search LSTM
    if args.tune_lstm:
        lstmHPO.hpo(stations, increment, args)

    # Random search TCN
    if args.tune_tcn:
        tcnHPO.hpo(stations, increment, args)

    # Random search GWN
    if args.tune_gwn:
        gwnHPO.hpo(increment, args)

    # Random search GWN
    if args.tune_wgn:
        wgnHPO.hpo(increment, args)

    # Train final SARIMA models
    if args.train_sarima:
        sarimaTrain.train(stations, increment)

    # Train final LSTM models
    if args.train_lstm:
        lstmTrain.train(stations, increment)

    # Train final TCN models
    if args.train_tcn:
        tcnTrain.train(stations, increment)

    # Train final GWN models
    if args.train_gwn:
        gwnTrain.train(increment, args)

    # Train final WGN models
    if args.train_wgn:
        wgnTrain.train(wgn_increment, args)

    # Record metrics for final SARIMA models
    if args.eval_sarima:
        sarimaEval.eval(stations, increment)

    # Record metrics for final LSTM models
    if args.eval_lstm:
        baselineEval.eval(stations, 'LSTM')

    # Record metrics for final TCN models
    if args.eval_tcn:
        baselineEval.eval(stations, 'TCN')

    # Record metrics for final GWN models
    if args.eval_gwn:
        gwnEval.eval(stations, args)

    # Record metrics for final WGN models
    if args.eval_wgn:
        wgnEval.eval(stations, args)

