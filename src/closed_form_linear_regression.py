import numpy as np
import util


def find_weights(inputs, output):
    """
    Execute the closed form linear regression algorithm.
    Returns the predicted weights and the mean and standard deviation of the training set.
    :param inputs: Standardized Input features (including bias)
    :param output: Output (dependant variable)
    :return:
    """

    # Compute Closed Form Linear Regression
    return np.linalg.inv(inputs.T.dot(inputs)).dot(inputs.T).dot(output)


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

    return standardized_test_inputs.dot(weights)


def execute(data):
    """

    :param data: Raw Data frame parsed from CSV
    :return: Nothing
    """

    # 2. Randomizes the data
    randomized_data = util.randomize_data(data)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
    training_data_size = 2.0 / 3.0
    training_data, test_data = util.split_data(randomized_data, training_data_size)

    # Capture the predicted outputs
    training_outputs = training_data[training_data.columns[-1]]

    # 4. Standardizes the data (except for the last column of course) using the training data
    training_inputs, training_mean, training_std = util.standardize_data(util.get_features(training_data))

    # Add offset column at the front
    training_inputs.insert(0, "Bias", 1)

    # 5. Computes the closed-form solution of linear regression
    weights = find_weights(training_inputs, training_outputs)

    # 6. Applies the solution to the testing samples
    test_input = util.get_features(test_data)
    expected = util.get_output(test_data)
    actual = apply_solution(test_input, training_mean, training_std, weights)

    # 7. Computes the root mean squared error (RMSE)
    rmse = util.compute_rmse(expected, actual)

    return weights, rmse