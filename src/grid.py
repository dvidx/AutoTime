import numpy
import os
import time

from math import sqrt
from numpy import mean, array, concatenate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



import sys
sys.path.insert(1, '../src')
import time_lag
import save_model

model_list = []

def drop_nonpred_cols(dataset, data, n_lag, pred_var_index):
    """[Drop columns that are not needed]

    Args:
        dataset ([type]): [description]
        data ([type]): [description]
        n_lag ([type]): [description]
        pred_var_index ([type]): [description]

    Returns:
        [type]: [description]
    """
    cols_drop = list()
    n_cols = dataset.shape[1]
    x = n_lag*n_cols + pred_var_index

    for i in range(n_lag*n_cols, len(data.columns)):
        if i != x:
            cols_drop.append(i)
        else:
            x = x + n_cols

    data.drop(data.columns[cols_drop], axis=1, inplace=True)

    return data


def train_test_split(data, n_test):
    """[Split data into train and test]

    Args:
        data ([type]): [description]
        n_test ([type]): [description]

    Returns:
        [type]: [description]
    """
    # split into train and test sets
    values = data.values
    train = values[:n_test, :]
    test = values[n_test:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train, test, train_X, train_y, test_X, test_y


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def difference(data, order):
    # Differencing # ! Not tested
    return [data[i] - data[i - order] for i in range(order, len(data))]


def model_fit(data, config, datapack):
    """[Model fitting]
    # Todo - Edit so multiple models can be tested and model is defined seperately
    Args:
        data ([type]): [description]
        config ([type]): [description]
        datapack ([type]): [description]

    Returns:
        [type]: [description]
    """
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff, n_dropout = config

    # unpack data
    train_X, train_y, test_X, test_y = datapack

    # prep data
    if n_diff > 0:
        data = difference(data, n_diff)

    # Build and compile model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(
        train_X.shape[1], train_X.shape[2]), dropout=n_dropout))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,
              verbose=0, shuffle=False, validation_data=(test_X, test_y))
    #validation_data=(test_X, test_y),

    return model

def model_predict(model, history, config):
    """[Compute predictions]

    Args:
        model ([type]): [description]
        history ([type]): [description]
        config ([type]): [description]
    """
    # unpack config
    n_input, _, _, _, _, n_dropout = config

    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    # shape input for model
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # make forecast
    yhat = model.predict(x_input, verbose=0)
    # correct forecast if it was differenced
    return correction + yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(dataset, data, cfg, n_test, pred_var_index):
    """[summary]

    Args:
        data ([type]): [description]
        cfg ([type]): [description]
        n_test ([type]): [description]
        pred_var_index ([type]): [description]

    Returns:
        [type]: [description]
    """    
    predictions = list()

    n_input, _, _, _, _, n_dropout = cfg

    data = time_lag.time_main(dataset, data, n_input, n_out=1)

    # drop columns that are not predicted
    data = drop_nonpred_cols(dataset, data, n_input, pred_var_index)

    # split dataset
    train, test, train_X, train_y, test_X, test_y = train_test_split(
        data, n_test)

    datapack = list()
    datapack.append(train_X)
    datapack.append(train_y)
    datapack.append(test_X)
    datapack.append(test_y)

    # fit model
    model = model_fit(train, cfg, datapack)
    # seed history with training dataset
    history = [x for x in train]

    # Get the scaler for the prediction variable
    values_pred = dataset.iloc[:, pred_var_index].values
    values_pred = values_pred.astype('float32')
    # Get normalized values for prediction variable
    scaler_pred = MinMaxScaler(feature_range=(0, 1))
    scaled_pred = scaler_pred.fit_transform(values_pred.reshape(-1, 1))

    # make a prediction
    yhat = model.predict(test_X)
    
    # invert scaling for forecast
    inv_yhat = scaler_pred.inverse_transform(yhat)

    # invert scaling for actual
    eval_test_y = test_y.reshape((len(test_y), 1))
    inv_y = scaler_pred.inverse_transform(eval_test_y)

    # estimate prediction error
    error = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(' > %.3f' % error)

    global model_list

    pred_df = dataset.copy()
    pred_df = pred_df.iloc[(n_test+n_input):, :]
    pred_df["Prediction"] = inv_yhat.reshape(inv_yhat.shape[0]).tolist()

    model_list.append((error, model, pred_df))
    # save_model(model, inv_yhat)
    # ToDo - Don't save directly, just append to stack (Keep top 3?)
    # ToDo - Then pass bakc the stack/list and further process (save etc.)

    return error


def repeat_evaluate(dataset, data, config, n_test, pred_var_index, n_repeats=5):  
     # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(
        dataset, data, config, n_test, pred_var_index) for _ in range(n_repeats)]
    # summarize score
    min_error = min(scores)
    result = mean(scores)
    print('> Model[%s] - Mean: %.3f - Min: %.3f ' % (key, result, min_error))
    return (key, result, min_error)

def grid_search(dataset, data, cfg_list, n_test, pred_var_index):  
    # evaluate configs
    scores = [repeat_evaluate(dataset, data, cfg, n_test, pred_var_index)
              for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def model_configs(config_dict):
    """[Create all possible combinations from given parameters]

    Returns:
        [type]: [List of config parameters to be tested]
    """    
    # Define scope of configs
    n_input = config_dict["n_input"]
    n_nodes = config_dict["n_nodes"]
    n_epochs = config_dict["n_epochs"]
    n_batch = config_dict["n_batch"]
    n_diff = config_dict["n_diff"]
    n_dropout = config_dict["n_dropout"]
    # create configs
    configs = list()
    for i in n_input:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    for m in n_diff:
                        for n in n_dropout:
                            cfg = [i, j, k, l, m, n]
                            configs.append(cfg)

    print('Total configs: %d' % len(configs))
    return configs


def grid_main(dataset, data, config_list, pred_var_index, n_test):
    """[summary]

    Args:
        dataset ([type]): [description]
        pred_var_index ([type]): [description]
        n_test ([type]): [description]
    """    

    # model configs
    cfg_list = model_configs(config_list)

    # grid search
    scores = grid_search(dataset, data, cfg_list, n_test, pred_var_index)
    
    # Save top 3 models 
    global model_list

    def sort_error(e):
        return e[0]
    model_list.sort(key=sort_error)

    for index, item in enumerate(model_list[:3]):
        save_model.save_model(index, item[1], item[2])


    for cfg, error, err_min in scores[:10]:
        print(cfg, error, err_min)

    return scores
