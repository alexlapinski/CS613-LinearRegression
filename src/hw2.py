import argparse
import pandas as pd
import matplotlib.pyplot as plt
import closed_form_linear_regression as cflr
import s_fold_cross_validation as sfold
import locally_weighted_linear_regression as lwlr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 613 - HW 2 Assignment")
    parser.add_argument("-c", "--cflr", action="store_true", dest="do_cflr",
                        help="Execute the 'Closed Form Linear Regression' problem")
    parser.add_argument("-s", "--s-folds", action="store_true", dest="do_sfold",
                        help="Execute the 'S-Folds Cross Validation' problem")
    parser.add_argument("-l", "--lwlr", action="store_true", dest="do_lwlr",
                        help="Execute the 'Locally-Weighted Linear Regression' problem")


    parser.add_argument("--style", action="store", dest="style", default="ggplot",
                        help="Set the matplotlib render style (default: ggplot)")
    parser.add_argument("--data", action="store", dest="data_filepath", default="./x06Simple.csv",
                        help="Set the filepath of the data csv file. (default: ./x06Simple.csv)")
    parser.add_argument("--out", action="store", dest="output_folderpath", default="graphs",
                        help="Set the output path of the folder to save graphs (default: graphs)")

    args = parser.parse_args()

    if not args.do_cflr and not args.do_sfold and not args.do_lwlr:
        parser.print_help()

    plt.style.use(args.style)

    raw_data = pd.read_csv(args.data_filepath, index_col=0)

    if(args.do_cflr):
        weights, rmse = cflr.execute(raw_data)
        print "Weights: {0}".format(weights)
        print "RMSE (Root Mean Squared Error): {0}".format(rmse)


    if(args.do_sfold):
        sfold.execute(raw_data)

    if(args.do_lwlr):
        lwlr.execute(raw_data)