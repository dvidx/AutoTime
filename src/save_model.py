import numpy
import json
import os


def save_model(index, model, pred_df):
    """[summary]

    Args:
        model ([type]): [description]
        inv_yhat ([type]): [description]
    """
    with open("./cfg/config.json", "r") as f:
        config = json.load(f)

    out_path = os.path.join(config["output_path"], config["project_name"])
    iteration = index
    pathname = "/models/"


    # ts = time.gmtime()
    # iteration = time.strftime("%Y_%m_%d_%H%M%S", ts)

    # * Forecast values
    filename = "prediction.csv"
    dirpath = "{}{}{}_{}".format(out_path, "/", iteration, filename)
    if not os.path.exists(dirpath):
        pred_df.to_csv(dirpath, index=False, header=True)

    # * Serialize model to JSON
    modelname_json = "{}{}{}_{}".format(out_path, pathname, iteration, "lstm.json")
    model_json = model.to_json()
    with open(modelname_json, "w") as json_file:
        json_file.write(model_json)

    # * Serialize weights to HDF5
    modelname_weights = "{}{}{}_{}".format(out_path, pathname, iteration, "lstm_weights.h5")
    model.save_weights(modelname_weights)

    # * Serialize all to HDF5
    modelname = "{}{}{}_{}".format(out_path, pathname, iteration, "lstm.h5")
    model.save(modelname)
