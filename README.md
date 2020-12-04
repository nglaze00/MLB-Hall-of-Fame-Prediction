# MLB-Hall-of-Fame-Prediction
Predicting whether a player will make the Hall of Fame, based on their stats so far. Uses LSTMs and time series data to get 96% accuracy on retired players, and 89% on all players.

Read our paper [here](Predicting_MLB_Hall_of_Fame_Players_Paper.pdf).

## Repository outline:
* data/: raw tables
* data_normalized/: normalized tables
  * batting_norm_batters_only.csv: Normalized batting stats, pre-2015
  * batting_norm_2010_2020.csv: Normalized batting stats, 2010 - 2020
* data_ready/: matrices for training / testing models
* models/: all code for training / evaluating models
  * agg_nn/: trained neural network
  * good_lstm/: trained LSTM
  * model_outputs/: test set probabilities and predictions for all models
  * nn_code/: code for neural network
  * agg_models.ipynb: aggregate format models
  * evaluate_models.ipynb: code for evaluating models
  * predict_players.ipynb: code for making predictions on specific players
  * ts_models.ipynb: time series format models
 * agg_nn_data_setup.ipynb: code for setting up neural network datasets
 * data_normalization.ipynb: normalizing batting statistics
 * data_splits.ipynb: creating train/test splits 
