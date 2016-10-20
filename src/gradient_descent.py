import util
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def plot_rmse_values(test_values, training_values, learning_rate, output_dir="./graphs", filename="gradient_descent_errors.png"):
    """
    Plot the RMSE values of the test data set and training dataset for each iteration
    :param test_values: The RMSE value of the test data at each iteration
    :param training_values: The RMSE value of the training data at each iteration
    :return: Nothing
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fig = plt.figure()
    test_data, = plt.plot(test_values, label="Testing Error")
    train_data, = plt.plot(training_values, label="Training Error")
    plt.ylabel("RMSE Value")
    plt.xlabel("Iteration")
    plt.suptitle("learning_rate = {0}".format(learning_rate))
    plt.legend(handles=[test_data, train_data])

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path)
    return output_path



def execute(data, learning_rate=0.001, training_data_ratio=2.0/3, max_iterations=1000000):
    """
    Perform Batch Gradient Descent

    :param data: Raw Data frame parsed from CSV
    :param training_data_ratio: The percent of given data to use for training (remaining percent is used for testing)
    :param max_iterations: The maximum number of iterations to execute before exiting
    :return: Nothing
    """

    # 2. Randomizes the data
    print "Randomizing Data"
    randomized_data = util.randomize_data(data)

    # 3. Selects the first 2 / 3 (round up) of the data for training and the remaining for testing
    print "Selecting Training Data"
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)

    # 4. Standardizes the data(except for the last column of course) base on the training data
    print "Standardizing Data"
    std_training_data, mean, std = util.standardize_data(training_data[training_data.columns[0:2]])
    std_training_data.insert(0, "Bias", 1)

    std_test_data, _, _ = util.standardize_data(test_data[test_data.columns[0:2]], mean, std)
    std_test_data.insert(0, "Bias", 1)

    iteration = 0
    prior_rmse = 0
    current_rmse = 100 # Doesn't matter what this value is, so long as it doesn't equal prior rmse
    eps = np.spacing(1)
    N = len(std_training_data)

    # Start with randomized values for theta
    theta = np.array([random.uniform(-1, 1) for _ in xrange(0, 3)])

    # Capture our expected values for the training data
    expected = training_data[training_data.columns[-1]]
    test_data_expected = test_data[test_data.columns[-1]]

    # Capture the RMSE for test and training over all iterations
    test_rmse_values = []
    training_rmse_values = []

    print "Performing Gradient Descent Linear Regression"
    # 5. While the termination criteria (mentioned above in the implementation details) hasn't been met
    while iteration <= max_iterations and abs(current_rmse - prior_rmse) >= eps:
        prior_rmse = current_rmse

        #   (a) Compute the RMSE of the training data
        #       By applying the current theta values to the training set & comparing results
        actual = std_training_data.dot(theta)
        current_rmse = util.compute_rmse(expected, actual)

        #   (b) While we can't let the testing set affect our training process, also compute the RMSE of
        #       the testing error at each iteration of the algorithm (it'll be interesting to see).
        #       Same thing as (a), but use test inputs / outputs
        test_data_actual = std_test_data.dot(theta)
        test_data_rmse = util.compute_rmse(test_data_expected, test_data_actual)

        #   (c) Update each parameter using batch gradient descent
        #       By use of the learning rate
        for i in xrange(len(theta)):
            # We know the length of theta is the same as the num columns in std_training_data
            errors = (actual - expected) * std_training_data[std_training_data.columns[i]]
            cumulative_error = errors.sum()
            theta[i] -= learning_rate / N * cumulative_error

        iteration += 1
        test_rmse_values.append(test_data_rmse)
        training_rmse_values.append(current_rmse)

    print "Completed in {0} iterations".format(iteration)

    print "Plotting Errors"
    image_path = plot_rmse_values(test_rmse_values, training_rmse_values, learning_rate)
    print "Saved Image to '{0}'".format(image_path)


    # 6. Compute the RMSE of the testing data.
    print "Computing RMSE of Test Data"
    test_data_actual = std_test_data.dot(theta)
    test_data_rmse = util.compute_rmse(test_data_expected, test_data_actual)
    return theta, test_data_rmse