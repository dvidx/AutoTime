#!/usr/bin/python

"""[summary]
"""

def main():


    # * Imports
    import os
    import sys
    import subprocess
    import json
    from pandas import read_csv
    from sklearn.preprocessing import MinMaxScaler, data

    sys.path.insert(1, './src')
    import stat_dist
    import grid


    print("...Imports finished...")
    print("")


    print("...Check Configuration...")
    # Read in Config
    with open("./cfg/config.json", "r") as f:
        config = json.load(f)

    # * Series of basic checks
    # Read in Data
    try:
        dataset = read_csv(config["input_file"])
    except:
        print("Could not find csv file or file damaged.")
        sys.exit()

    # Check for predictor variable
    try:
        pred_var_index = dataset.columns.get_loc(config["pred_var"])
    except:
        print("Error finding prediction variable. Check spelling and remove special characters.")
        sys.exit()

    # Check if project folder does not exist yet
    p_path = os.path.join(config["output_path"], config["project_name"])
    if os.path.isdir(p_path) == True:
        print("Your desired Output-Directory already exist! Change config or move/delete folder.")
        print(p_path)
        sys.exit()
    else:
        try:
            temp = os.path.join(p_path, "files")
            os.makedirs(temp)
            temp = os.path.join(p_path, "plots")
            os.mkdir(temp)
            temp = os.path.join(p_path, "models")
            os.mkdir(temp)
        except OSError:
            print("Creation of the directory %s failed" % p_path)
            sys.exit()

    print("...Check finished...")
    print("")


    # * Add statistical distributions
    # ToDO - Option to remove columns? Or better which wanna to not derive statistical distribution from
    print("...Add statistical distributions...")
    dataset = stat_dist.stat_main(dataset, config["stat_dist_steps"], p_path)
    print("...Finished...")
    print("")

    # * Normalize
    # Get values of dataset
    values = dataset.values
    values = values.astype('float32')  # ensure all data is float

    # Normalize all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Get the scaler for the prediction variable
    values_pred = dataset.iloc[:, pred_var_index].values
    values_pred = values_pred.astype('float32')
    # Get normalized values for prediction variable
    scaler_pred = MinMaxScaler(feature_range=(0, 1))
    scaled_pred = scaler_pred.fit_transform(values_pred.reshape(-1, 1))

    print("...Data normalized...")
    print("")

    # * Grid search
    print("...Starting Gridsearch...")
    grid_cfg = config["gridsearch"]
    n_test = config["train_split"]
    scores = grid.grid_main(dataset, scaled, grid_cfg, pred_var_index, n_test)

    # Display results
    print("")
    print("")
    print("Config with best result: ", scores[:1])

    # ! Already done in grid file
    # Create predictions on the best model


    # * Run R-File to produce visualization
    try:
        command = 'Rscript'
        path2script = './visualize.R'

        # Variable number of args in a list
        # wd, lag, forecast steps, results, Increment True/False, Time start, Time end, out path
        args = [
            p_path,
            config["n_input"],
            config["forecast_steps"],
            1,
            False,
            config["start_date"],
            config["end_date"],

        ]

        # Build subprocess command
        cmd = [command, path2script] + args
        subprocess.call(cmd)
        # # check_output will run the command and store to result
        # x = subprocess.check_output(cmd, universal_newlines=True)
        print('Your plots can be found under "', p_path, '"')

    except:
        print("R Executable could not be found. Please execute RScript manually. (File in Project Folder)")

if __name__ == "__main__":
    main()