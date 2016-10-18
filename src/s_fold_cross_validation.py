import closed_form_linear_regression as cflr
import numpy as np
import util
import math
import sys


def divide_data(input, num_parts = 5):
    """
    Divide our data into 'num_parts' to be used for training / testing.
    If the data cannot be divided evenly, then the remaining data rows will be as evenly distributed
    to each of the groups as possible.
    :param input: dataframe to divide
    :param num_parts: The number of parts to split the data into
    :return: Array of dataframes. There will be num_parts, but all parts may not be the same size
    """

    assert num_parts > 1, "num_parts must be greater than one."
    folds = []

    fold_size = len(input) / num_parts
    remaining_items = len(input) - (fold_size * num_parts)

    # Distribute any remaining items as evenly as possible, rather than having one super-large last fold
    if remaining_items > 0:
        fold_size += 1

    for i in xrange(0, num_parts):
        from_index = i*fold_size
        to_index = from_index + fold_size

        if to_index >= len(input):
            to_index = len(input)

        folds.append(input.iloc[from_index:to_index])

    return folds


def select_training_data(folds, iteration):
    """
    Given the list of all folds, the current fold and the total number of folds.
    Select the training data for use in cross validation
    :param folds: A list of dataframes
    :param iteration: An integer between 0 and len(folds)-1 indicating the current iteration
    :return: A single dataframe of training data
    """
    num_folds = len(folds)
    assert iteration >= 0, "We expect iteration to be greater than or equal to zero, but it was {0}".format(iteration)
    assert iteration < num_folds, "We expect iteration to be less than num_folds, but it was {0}".format(iteration)

    # Find what indices we should be using (as we loop around)
    training_index = []
    for j in xrange(0, iteration):
        training_index.append(j)
    for j in xrange(iteration + 1, num_folds):
        training_index.append(j)

    # Build up the final dataframe of training data
    training_data = folds[training_index[0]]
    for index in xrange(1, len(training_index)):
        training_data = training_data.append(folds[training_index[index]])

    return training_data


def execute(data, num_folds = 5):
    """
    Compute the Root Mean Squared Error using num_folds for cross validation
    :param data: Raw Data frame parsed from CSV
    :param num_folds: The number of folds to use
    :return: Root Mean Squared Error
    """
    assert data is not None, "data must be a valid DataFrame"
    assert num_folds > 1, "num_folds must be greater than one."

    # 2. Randomizes the data
    randomized_data = util.randomize_data(data)

    # 3. Creates S folds (for our purposes S = 5, but make your code generalizable, that is it should
    #   work for any legal value of S)
    folds = divide_data(randomized_data, num_folds)

    squared_errors = []
    # 4. For i = 1 to S
    for i in xrange(0, num_folds):
        #   (a) Select fold i as your testing data and the remaining (S - 1) folds as your training data
        test_data = folds[i]
        training_data = select_training_data(folds, i)

        #   (b) Standardizes the data (except for the last column of course) based on the training data
        standardized_train_data, mean, std = util.standardize_data(training_data)

        #   (c) Train a closed-form linear regression model
        training_outputs = training_data[training_data.columns[-1]]
        weights = cflr.find_weights(standardized_train_data, training_outputs)

        #   (d) Compute the squared error for each sample in the current testing fold
        expected = test_data[test_data.columns[-1]]
        actual = apply_solution(mean, std, test_data, weights)

        squared_error = (expected - actual)**2
        squared_errors.append(squared_error)

    # 5. Compute the RMSE using all the errors.
    rmse = compute_rmse(len(data), squared_errors)

    return rmse


def compute_rmse(n, squared_errors):
    """
    Compute the root mean squared errors
    :param n: number of data points to use for sum
    :param squared_errors: list of squared errors (list of dataframes)
    :return: root mean squared error
    """
    sum_of_squared_errors = 0
    for squared_error in squared_errors:
        sum_of_squared_errors += squared_error.sum()
    rmse = math.sqrt(sum_of_squared_errors / n)
    return rmse


def apply_solution(mean, std, test_data, weights):
    """
    Compute the predicted values from the given test data and weights
    :param mean: training_data mean value
    :param std: training_data standard deviation value
    :param test_data: test dataframe to use for validation
    :param weights: weights produced from training
    :return: actual predicted values
    """
    test_input = test_data[test_data.columns[0:2]]
    standardized_test_inputs, _, _ = util.standardize_data(test_input, mean, std)
    standardized_test_inputs = standardized_test_inputs[test_input.columns]
    standardized_test_inputs.insert(0, "Bias", 1)
    return np.dot(standardized_test_inputs, weights)



