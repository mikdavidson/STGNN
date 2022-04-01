# Weather-DL 

*Deep Learning pipeline for weather prediction using temporal and spatio-temporal deep learning models.* 


| Member     | Student Number |
| ----------- | ----------- |
| Mikhail Davidson      | DVDMOH003       |

| Staff Member     | Department |
| ----------- | ----------- |
| A/Prof Deshendran Moodley | Computer Science       |

# Requirements

python3

See `requirements.txt`


# Installation

`virtualenv -p /usr/bin/python3.8 venv`

`source venv/bin/activate`

`pip3 install -r requirements.txt`


# Experiments

## Random-Search Hyper-Parameter Optimisation(HPO)

Baseline HPO across 21 weather stations on 24 hour forecasting horizon:

`python3 main.py --tune_sarima=True`

`python3 main.py --tune_lstm=True`

`python3 main.py --tune_tcn=True`

WGN GNN HPO on 24 hour forecasting horizon:

`python3 main.py --tune_wgn=True`

GWN GNN HPO on 24 hour forecasting horizon:

`python3 main.py --tune_gwn=True`


## Training Models Using Optimal Hyper-Parameters

Baseline HPO across 21 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon:

`python3 main.py --train_sarima=True`

`python3 main.py --train_lstm=True`

`python3 main.py --train_tcn=True`

WGN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon:

`python3 main.py --train_wgn=True`

GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon:

`python3 main.py --train_gwn=True`


## Evaluate Models' Performance(MSE, RMSE, MAE, SMAPE)


 Evaluation across 21 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon:

`python3 main.py --eval_sarima=True`

`python3 main.py --eval_lstm=True`

`python3 main.py --eval_tcn=True`


WGN GNN evaluation on [3, 6, 9, 12, 24] hour forecasting horizon on each of the 21 weather stations:

`python3 main.py --eval_wgn=True`

GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon on each of the 21 weather stations:

`python3 main.py --eval_gwn=True`
