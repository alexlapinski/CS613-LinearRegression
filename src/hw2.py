import os
import argparse
import plotting
import pandas as pd
import matplotlib.pyplot as plt

def standardize_dataframe(dataframe):
    return (dataframe - dataframe.mean()) / dataframe.std()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 613 - HW 2 Assignment")
    parser.add_argument("-r", "--plot-raw-data", action="store_true", dest="plot_raw_data",
                        help="Plot and save graphs of the raw data")
    parser.add_argument("-s", "--plot-standardized-data", action="store_true", dest="plot_standardized_data",
                        help="Plot and save graphs of the standardized data")


    parser.add_argument("--style", action="store", dest="style", default="ggplot",
                        help="Set the matplotlib render style (default: ggplot)")
    parser.add_argument("--data", action="store", dest="data_filepath", default="./x06Simple.csv",
                        help="Set the filepath of the data csv file. (default: ./x06Simple.csv)")
    parser.add_argument("--out", action="store", dest="output_folderpath", default="graphs",
                        help="Set the output path of the folder to save graphs (default: graphs)")

    args = parser.parse_args()

    if(not(args.plot_raw_data) and not(args.plot_standardized_data)):
        parser.print_help()

    plt.style.use(args.style)

    raw_data = pd.read_csv(args.data_filepath, index_col=0)

    if(args.plot_raw_data):
        raw_data.plot()

    if(args.plot_standardized_data):
        clean_df = standardize_dataframe(raw_data)
