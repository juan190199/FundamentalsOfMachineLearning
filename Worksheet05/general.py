from sklearn.model_selection import train_test_split


def data_preparation(digits, test_percentage=0.33, random_seed=0):
    """

    :param digits:
    :param test_percentage:
    :param random_seed:
    :return:
    """
    data = digits['data']
    target = digits['target']

    # Random split
    X_train, X_test, Y_train, Y_test = train_test_split(
        data,
        target,
        test_size=test_percentage,
        random_state=random_seed
    )

    return X_train, X_test, Y_train, Y_test
