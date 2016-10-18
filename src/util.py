import math


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