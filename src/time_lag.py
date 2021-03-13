# convert series to supervised learning

# Import
import time
import json
import os 
from pandas import DataFrame
from pandas import concat


# convert series to supervised learning
def series_to_supervised(dataset, data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{dataset.columns[j]}(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{dataset.columns[j]}(t)') for j in range(n_vars)]
        else:
            names += [(f'{dataset.columns[j]}(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def time_main(dataset, data, n_in, n_out):

    # * Frame as supervised learning
    reframed = series_to_supervised(dataset, data, n_in, n_out)
    
    # * Save step as csv
    # Read in Config
    with open("./cfg/config.json", "r") as f:
        config = json.load(f)
    
    out_path = os.path.join(config["output_path"], config["project_name"])

    pathname = "/files/"
    # ts = time.gmtime()
    # iteration = time.strftime("%Y_%m_%d_%H%M%S", ts)
    filename = "_supervised.csv"
    dirpath = "{}{}_{}".format(out_path, pathname, filename) # iteration,
    reframed.to_csv(dirpath, index=False, sep=",")

    return reframed