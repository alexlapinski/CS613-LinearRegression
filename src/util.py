import math


def get_features(dataframe):
    """
    Return all but the last column of the dataframe as features
    :param dataframe:
    :return:
    """
    return dataframe[dataframe.columns[0:-1]]


def get_output(dataframe):
    """
    Return the last column of the dataframe (this is our dependant variable, or the value we are trying to predict)
    :param dataframe:
    :return:
    """
    return dataframe[dataframe.columns[-1]]


def standardize_data(dataframe, mean = None, std = None):
    """
    Standardize the given dataframe (subtract mean and divide by standard deviation).
    No columns or rows will be excluded except for the index and headers.
    The mean and standard deviation used in standardizing the data will be returned as well.
    :param dataframe: The dataframe containing data to standardize
    :param mean: A pre-calculated mean to use
    :param std: A pre-calculated standard deviation to use
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
    print "Using {0} percent of the input data for training.".format(training_data_ratio)

    training_data = dataframe.iloc[:max_training_index]
    test_data = dataframe.iloc[max_training_index:]

    print "Size of Training Data: {0}, Size of Test Data: {1}".format(len(training_data), len(test_data))

    return training_data, test_data


def compute_se(expected, actual):
    """
    Compute the squared error. We are given 2 column vectors of expected values and actual values.
    We then compute the square error for this given dataset.
    :param expected: A Column vector containing the expected datapoints
    :param actual: A Column vector containing the actual datapoints (predicted using weights).
    :return: Float indicating SE
    """

    difference = expected - actual

    return difference**2


def compute_mse(expected, actual):
    """
    Compute the mean squared error. We are given 2 column vectors of expected values and actual values.
    We then compute the mean square error for this given dataset.
    :param expected: A Column vector containing the expected datapoints
    :param actual: A Column vector containing the actual datapoints (predicted using weights).
    :return: Float indicating MSE
    """

    n = len(expected)

    return (compute_se(expected, actual)).sum()/n


def compute_rmse(expected, actual):
    """
    Compute the root mean squared error. We are given 2 column vectors of expected values and actual values.
    We then compute the root means square error for this given dataset.
    :param expected: A Column vector containing the expected datapoints
    :param actual: A Column vector containing the actual datapoints (predicted using weights).
    :return: Float indicating RMSE
    """

    return math.sqrt(compute_mse(expected, actual))