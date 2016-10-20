import argparse
import pandas as pd
import matplotlib.pyplot as plt
import closed_form_linear_regression as cflr
import s_fold_cross_validation as sfold
import locally_weighted_linear_regression as lwlr
import gradient_descent as gd

# TODO: Make sure entire solution can handle more than 2 features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 613 - HW 2 Assignment")
    parser.add_argument("-c", "--cflr", action="store_true", dest="do_cflr",
                        help="Execute the 'Closed Form Linear Regression' problem")

    parser.add_argument("-s", "--s-folds", action="store_true", dest="do_sfold",
                        help="Execute the 'S-Folds Cross Validation' problem")
    parser.add_argument("--num-folds", action="store", dest="num_folds", default=5, type=int,
                        help="Set the number of folds to use with s-folds (default: 5)")

    parser.add_argument("-l", "--lwlr", action="store_true", dest="do_lwlr",
                        help="Execute the 'Locally-Weighted Linear Regression' problem")
    parser.add_argument("--k-value", action="store", dest="k_value", default=1, type=int,
                        help="Set the K-Value for locally weighted linear regression (default: 1)")

    parser.add_argument("-g", "--gradient", action="store_true", dest="do_gradient",
                        help="Execute the 'Gradient Descent' problem")
    parser.add_argument("--learning-rate", action="store", dest="learning_rate", default=0.01, type=float,
                        help="Set the Learning Rate for Gradient Decent Linear Regression (default: 0.01)")

    parser.add_argument("--style", action="store", dest="style", default="ggplot",
                        help="Set the matplotlib render style (default: ggplot)")
    parser.add_argument("--data", action="store", dest="data_filepath", default="./x06Simple.csv",
                        help="Set the filepath of the data csv file. (default: ./x06Simple.csv)")

    args = parser.parse_args()

    if not args.do_cflr and not args.do_sfold and not args.do_lwlr and not args.do_gradient:
        parser.print_help()

    plt.style.use(args.style)

    raw_data = pd.read_csv(args.data_filepath, index_col=0)

    if args.do_cflr:
        print "Executing Closed Form Linear Regression"
        weights, rmse = cflr.execute(raw_data)
        print "Weights: {0}".format(weights)
        print "RMSE (Root Mean Squared Error): {0}".format(rmse)
        print ""

    if args.do_sfold:
        num_folds = args.num_folds
        print "Executing S-Folds Validation Closed Form Linear Regression"
        print "Using {0} folds".format(num_folds)
        rmse = sfold.execute(raw_data, num_folds)
        print "RMSE (Root Mean Squared Error): {0}".format(rmse)
        print ""

    if args.do_lwlr:
        k = args.k_value
        print "Executing Locally-Weighted Linear Regression"
        print "Using {0} for k in local weighting.".format(k)
        rmse = lwlr.execute(raw_data, k=k)
        print "RMSE (Root Mean Squared Error): {0}".format(rmse)
        print ""

    if args.do_gradient:
        learning_rate = args.learning_rate
        print "Executing Gradient Descent"
        print "Using {0} for the learning rate".format(learning_rate)
        weights, rmse = gd.execute(raw_data, learning_rate=learning_rate)
        print "Weights: {0}".format(weights)
        print "RMSE (Root Mean Squared Error): {0}".format(rmse)