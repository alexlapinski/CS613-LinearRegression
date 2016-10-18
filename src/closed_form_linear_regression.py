import pandas as pd
import numpy as np
import math
import util


def split_data(dataframe, training_data_ratio):
    """
    Split the given dataframe into 2 sets, one for training, and one for testing.
    :param dataframe: The source data to segment into training and test data
    :param training_data_ratio: The ration (e.g. 2/3) of the overall data to use for the training set, remaining is used for test set.
    :return: Tuple of training data and test data
    """
    if training_data_ratio > 1 or training_data_ratio < 0:
        raise ValueError("training_data_ratio must be between 0.0 and 1.0")

    max_training_index = int(math.floor(len(dataframe) * training_data_ratio))
    print "Using {0} percent of the input data for training.".format(training_data_ratio)

    training_data = dataframe.iloc[:max_training_index]
    test_data = dataframe.iloc[max_training_index:]

    print "Size of Training Data: {0}, Size of Test Data: {1}".format(len(training_data), len(test_data))

    return training_data, test_data


def find_weights(training_inputs, training_outputs):
    """
    Execute the closed form linear regression algorithm.
    Returns the predicted weights and the mean and standard deviation of the training set.
    :param training_inputs:
    :param training_outputs:
    :return:
    """

    # Compute Closed Form Linear Regression
    result = np.linalg.inv(np.dot(training_inputs.T, training_inputs))

    result = np.dot(result, training_inputs.T)
    result = np.dot(result, training_outputs)

    return result


def apply_solution(test_input, training_mean, training_std, weights):
    """
    Apply the closed form linear regression to the given dataframe.
    The input dataframe is expected to contain only the input columns, and not the output column
    :param test_input: Non-Standardized Dataframe, the expected output column is expected to be excluded
    :param weights: The weights produced by learning
    :param training_mean: The mean value used in standardizing the training set
    :param training_std: the standard deviation value using in standardizing the training set
    :return:
    """
    standardized_test_inputs, _, _ = util.standardize_data(test_input, training_mean, training_std)
    standardized_test_inputs.insert(0, "Bias", 1)

    results = np.dot(standardized_test_inputs, weights)

    return results


def execute(data):
    """

    :param data: Raw Data frame parsed from CSV
    :return: Nothing
    """

    # 2. Randomizes the data
    randomized_data = util.randomize_data(data)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
    training_data_size = 2.0 / 3.0
    training_data, test_data = split_data(randomized_data, training_data_size)

    # Capture the predicted outputs
    training_outputs = training_data[training_data.columns[-1]]

    # 4. Standardizes the data (except for the last column of course) using the training data
    (training_inputs, training_mean, training_std) = util.standardize_data(training_data[training_data.columns[0:2]])

    # Add offset column at the front
    training_inputs.insert(0, "Bias", 1)

    # 5. Computes the closed-form solution of linear regression
    weights = find_weights(training_inputs, training_outputs)

    # 6. Applies the solution to the testing samples
    test_input = test_data[test_data.columns[0:2]]
    expected = test_data[test_data.columns[-1]]
    actual = apply_solution(test_input, training_mean, training_std, weights)

    # 7. Computes the root mean squared error (RMSE)
    rmse = util.compute_rmse(expected, actual)

    return weights, rmse


