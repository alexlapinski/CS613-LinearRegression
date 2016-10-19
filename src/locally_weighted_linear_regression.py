import util
import math
import pandas as pd
import numpy as np


def compute_local_weights(training_points, test_point, k):
    """
    Compute the local weight of the test_point to the given training points
    :param training_points: Source Point
    :param test_point: Target Point
    :param k: tuning parameter
    :return:
    """
    distance = (training_points - test_point).sum(axis=1)
    return math.e ** ((-1 * distance)/k**2)


def execute(data, training_data_ratio=2.0/3.0, k=1):
    """
    Execute the "Locally-Weighted" Linear Regression (using Closed-Form Linear Regression)
    :param data: Raw Data frame parsed from CSV
    :param training_data_ratio: The percent (0.0 to 1.0) of input data to use in training.
    :param k: Smoothing parameter for local weight computation
    :return: Nothing
    """
    # 2. Randomize the data
    randomized_data = util.randomize_data(data)

    # 3. Select the first 2 / 3(round up) of the data for training and the remaining for testing
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)
    training_outputs = training_data[training_data.columns[-1]]

    # 4. Standardize the data(except for the last column of course) using the training data
    standardized_training_data, mean, std = util.standardize_data(training_data[training_data.columns[0:2]])

    # Add offset column at the front
    standardized_training_data.insert(0, "Bias", 1)

    std_test_data, _, _ = util.standardize_data(test_data[test_data.columns[0:2]], mean, std)
    std_test_data.insert(0, "Bias", 1)

    squared_errors = []
    # 5. Then for each testing sample
    for i in xrange(0, len(std_test_data)):

        testing_sample = std_test_data.iloc[i]
        expected_output = test_data.loc[testing_sample.name][-1]

        theta_query = compute_theta_query(testing_sample, standardized_training_data, training_outputs, k)

        # (b) Evaluate the testing sample using the local model.
        actual_output = np.dot(testing_sample, theta_query)

        # (c) Compute the squared error of the testing sample.
        error = expected_output - actual_output
        squared_errors.append(error ** 2)

    # 6. Compute the root mean squared error (RMSE)
    sum_of_squared_errors = 0
    for error in squared_errors:
        sum_of_squared_errors += error

    mean_squared_error = sum_of_squared_errors / len(squared_errors)

    rmse = math.sqrt(mean_squared_error)

    return rmse


def compute_theta_query(query, standardized_training_data, training_outputs, k):
    """
    Compute the local theta for the given query using the standardized training data and training outputs
    :param query: The single row from the test data to compute the local theta for
    :param standardized_training_data: The standardized training data (inputs)
    :param training_outputs: The raw expected outputs for each training row
    :return: A theta weighted specifically for this query
    """

    # (a) Compute the necessary distance matrices relative to the training data in order to compute
    #     a local model.
    local_weights = compute_local_weights(standardized_training_data, query, k)
    wx = np.dot(np.diag(local_weights), standardized_training_data)
    wy = np.dot(np.diag(local_weights), training_outputs)

    first_term = np.linalg.inv(np.dot(wx.T, wx))
    theta_query = np.dot(np.dot(first_term, wx.T), wy)
    return theta_query
