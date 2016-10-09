import pandas as pd
import numpy as np
import math


def standardize_data(dataframe, mean = None, std = None):
    """
    Standardize the given dataframe (subtract mean and divide by standard deviation).
    No columns or rows will be excluded except for the index and headers.
    The mean and standard deviation used in standardizing the data will be returned as well.
    :param dataframe: The dataframe containing data to standardize
    :param mean: A precaculauted mean to use
    :param std: A precalculated standard devation to use
    :return: Standardized dataframe, mean used and standard deviation
    """

    if mean is None:
        mean = dataframe.mean()

    if std is None:
        std = dataframe.std()

    standardized_dataframe = (dataframe - mean) / std

    return standardized_dataframe, mean, std


def randomize_data(dataframe):
    """
    Randomize the given dataframe.
    :param dataframe:
    :return: randomized dataframe
    """
    random_seed = 0
    return dataframe.sample(n=len(dataframe), random_state=random_seed)


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

    training_data = dataframe.iloc[:max_training_index]
    test_data = dataframe.iloc[max_training_index:]

    return training_data, test_data


def learn(dataframe):
    """
    Execute the linear regression algorithm and produce a set of weights and RMSE for evaluation
    :param dataframe: input dataframe containing all data
    :return:
    """
    training_data_size = 2.0 / 3.0

    # Randomize Data
    rand_df = randomize_data(dataframe)

    # Select Training / Test sets
    train_df, test_df = split_data(rand_df, training_data_size)

    # Capture the predicted outputs
    training_outputs = train_df[train_df.columns[-1]]

    # Standardize training (excluding last column)
    (training_inputs, training_mean, training_std) = standardize_data(train_df[train_df.columns[0:2]])

    # Add offset column at the front
    training_inputs.insert(0, "Bias", 1)

    weights = find_weights(training_inputs, training_outputs)

    test_inputs = test_df[test_df.columns[0:2]]
    test_inputs.insert(0, "Bias", 1)

    test_outputs = test_df[test_df.columns[-1]]
    test_inputs, _, _ = standardize_data(test_inputs, training_mean, training_std)

    print test_inputs.shape, weights.shape

    results = apply_solution(test_inputs, weights)

    rmse = compute_rmse(test_outputs, results)

    return weights, rmse


def find_weights(training_inputs, training_outputs):
    """
    Execute the closed form linear regression algorithm.
    Returns the predicted weights and the mean and standard deviation of the training set.
    :param training_inputs:
    :param training_outputs:
    :return:
    """

    # Compute Closed Form Linear Regression
    first_term = np.dot(training_inputs.T, training_inputs)
    first_term = np.linalg.inv(first_term)

    second_term = np.dot(training_inputs.T, training_outputs)

    return np.dot(first_term, second_term)


def apply_solution(dataframe, weights):
    """
    Apply the closed form linear regression to the given dataframe.
    The input dataframe is expected to contain only the input columns, and not the output column
    :param dataframe: Non-standardized inputs
    :param weights: The weights produced by learning
    :param training_mean: The mean value used in standardizing the training set
    :param training_std: the standard deviation value using in standardizing the training set
    :return:
    """

    return np.dot(dataframe, weights)


def compute_rmse(expected, actual):
    """
    Compute the root mean squared error. We are given 2 column vectors of expected values and actual values.
    We then compute the root means square error for this given dataset.
    :param expected: A Column vector containing the expected datapoints
    :param actual: A Column vector containing the actual datapoints (predicted using weights).
    :return: Float indicating RMSE
    """

    N = len(expected)

    difference = expected - actual
    sum = (difference**2).sum()

    return math.sqrt(sum/N)