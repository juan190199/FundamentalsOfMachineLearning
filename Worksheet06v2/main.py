import Worksheet06v2.Data.DataPreparation as DPimport Worksheet06v2.Data.myDataset as mDimport Worksheet06v2.RegressionTree as RTfrom sklearn import linear_modelfrom sklearn import model_selectionimport numpy as npimport matplotlib.pyplot as pltdef evaluate(prediction, target):    assert prediction.shape == target.shape    N = prediction.shape[0]    return np.sum(np.square(prediction - target)) / Ndef k_fold_cv(mod, feats, targets, k):    assert targets.shape[0] == feats.shape[0]    kf = model_selection.KFold(n_splits=k)    err_train, err_test = [], []    for train, test in kf.split(feats):        mod.fit(feats[train], targets[train])        # Estimate for the training error, evaluated on the whole data        err_train.append(evaluate(mod.predict(feats), targets))        # Estimate for the test error, evaluated on the test fold        err_test.append(evaluate(mod.predict(feats[test]), targets[test]))    return (np.mean(err_test), np.std(err_test), np.mean(err_train), np.std(err_train))def test_data(linear, regtree, data, n_shufflesets, orig_err_linear, orig_err_regtree, pred_variable="rating"):    """    Create shuffled datasets    :param linear:    :param regtree:    :param data:    :param n_shufflesets:    :param orig_err_linear:    :param orig_err_regtree:    :param pred_variable:    :return:    """    shufflesets = []    # Set a random seed to make results reproducible    np.random.seed(1089045)    # Shuffling works inplace, therefore we retrieve a copy    rating = np.array(data[pred_variable]).copy()    for i in range(n_shufflesets):        np.random.shuffle(rating)        shuffled_data = data.copy()        shuffled_data[pred_variable] = rating        shufflesets.append(mD.myDataset(shuffled_data))    # Evaluate them:    shuffled_err_linear = []    shuffled_err_regtree = []    for i, dset in enumerate(shufflesets):        print('Run shuffleset no %i' % (i + 1))        feats = dset.get_X_oc_()        targets = dset.targets        assert feats.shape[0] == targets.shape[0]        k = 10        err_test, std_err_test, err_train, std_err_train = k_fold_cv(linear, feats, targets, k)        shuffled_err_linear.append(err_test)        err_test, std_err_test, err_train, std_err_train = k_fold_cv(regtree, feats, targets, k)        shuffled_err_regtree.append(err_test)    n_larger_linear = (shuffled_err_linear > orig_err_linear).sum()    n_larger_regtree = (shuffled_err_regtree > orig_err_regtree).sum()    return n_larger_linear, n_larger_regtreedef main():    # Data preparation    prediction_data, aggregated_df = DP.dataPreparation()    dset = mD.myDataset(prediction_data)    # Define the different models    linear = linear_model.LinearRegression()    regtree = RT.RegressionTree()    # Calculate mean squared error    # Evaluate all model / feature set combinations    k = 10    for mod in (linear, regtree):        print("Running:", mod)        for feats_str in ("X_oc", "X_on", "X_oc_"):            print("Dataset: %s" % feats_str)            feats = eval("dset.get_" + feats_str + "()")            targets = dset.targets            assert feats.shape[0] == targets.shape[0]            err_test, std_err_test, err_train, std_err_train = k_fold_cv(mod, feats, targets, k)            print('test_err: %.2E' % err_test)    print("Done.")    # Determine if there is skin color bias    orig_err_linear, _, _, _ = k_fold_cv(linear, dset.get_X_oc_(), dset.targets, k)    orig_err_regtree, _, _, _ = k_fold_cv(regtree, dset.get_X_oc_(), dset.targets, k)    orig_err_linear, orig_err_regtree    n_shufflesets = 19    n_larger_linear, n_larger_regtree = test_data(linear,                                                  regtree,                                                  prediction_data,                                                  n_shufflesets,                                                  orig_err_linear,                                                  orig_err_regtree)    print('Number of greater test errors with shuffled data and linear model: %i' % n_larger_linear)    print('Number of greater test errors with shuffled data and regression tree model: %i' % n_larger_regtree)    """    Here, the linear model would support the hypothesis of a racial bias, whereas the regression tree would reject it    Note that this result can also vary with your choice of random seeds for the shuffling operation.    """    # Tweak the data    """    This snip of code was inspired by {"Alexander Kunkel", "Daniel Galperin", "Jonas Hellgoth"}'s solution.    Two ways to provide the data are provided:    1. Cherry picking aka data selection: We select data that confirms our hypothesis beforehand.    We pick the data where the skin colour raters disagree. This choice has no other reason than that it works.    2. Feature selection: most of the features that are also relevant to the prediction are dropped,    so that only a few in addition to the skin colour remain.     Intuitively, with so few features, the skin colour must have an influence.    """    # Cherry picking    data_cherry_pick = aggregated_df.copy()    # Drop rows if raters disagree    data_cherry_pick = data_cherry_pick.loc[data_cherry_pick.rater1 == data_cherry_pick.rater2]    data_cherry_pick['rating'] = data_cherry_pick['rater1']    data_cherry_pick = data_cherry_pick.drop(['rater1', 'rater2'], axis=1)    new_count = data_cherry_pick.shape[0]    removed_count = aggregated_df.shape[0] - data_cherry_pick.shape[0]    print(f"{removed_count} entries removed because of inconsistent ratings, {new_count} remaining.")    n_larger_linear, n_larger_regtree = test_data(linear,                                                  regtree,                                                  data_cherry_pick,                                                  n_shufflesets,                                                  orig_err_linear,                                                  orig_err_regtree)    # Manual feature reduction    # drop many attributes    reduced_data = prediction_data.drop(['weight', 'height', 'victories', 'defeats', 'position', 'leagueCountry',                                         'birthday', 'club'], axis=1)    n_larger_linear, n_larger_regtree = test_data(linear,                                                  regtree,                                                  reduced_data,                                                  n_shufflesets,                                                  orig_err_linear,                                                  orig_err_regtree)    print('Number of greater test errors with shuffled data and linear model: %i' % n_larger_linear)    print('Number of greater test errors with shuffled data and regression tree model: %i' % n_larger_regtree)    # Alternative hypothesis    """    Since only correlation is showed, there are other causalities which could lead to the same manifestation of test     results besides an actual racial bias.     Possible hypothesis:     1. dark colored players play in positions prone to get more red cards such as defense.    2. dark colored players being more likely to be taller and heavier so the probability of winning a tackle increases and     referee could tend to give more cards to players winning more tackles        test_data is applied as before, but drop the skin colour rating and shuffle the variables "position" and "weight".     Additionally, the correlation between skin color and the new variables is checked    """    # Defense hypothesis    pos_data = prediction_data.copy()    pos_data["is_defense"] = pos_data["position"].isin(["Center Back", "Left Fullback", "Right Fullback"])    pos_data_pred = pos_data.drop(["rating"], axis=1)    n_larger_linear, n_larger_regtree = test_data(linear,                                                  regtree,                                                  pos_data_pred,                                                  19,                                                  orig_err_linear,                                                  orig_err_regtree,                                                  "is_defense")    print('Number of greater test errors with shuffled data and linear model: %i' % n_larger_linear)    print('Number of greater test errors with shuffled data and regression tree model: %i' % n_larger_regtree)    bins = np.arange(-0.125 / 2, 1 + .125, 0.125)    pos_data.boxplot("rating", "is_defense")    plt.title("")    plt.ylabel("Skin Colour Rating")    plt.tight_layout()    print(f"The correlation between rating and defense is: {pos_data.rating.corr(pos_data.is_defense):1.4f}")    # Weight hypothesis    weight_data = prediction_data.copy()    weight_data_pred = pos_data.drop(["rating"], axis=1)    n_larger_linear, n_larger_regtree = test_data(linear,                                                  regtree,                                                  prediction_data,                                                  19,                                                  orig_err_linear,                                                  orig_err_regtree,                                                  "weight")    print('Number of greater test errors with shuffled data and linear model: %i' % n_larger_linear)    print('Number of greater test errors with shuffled data and regression tree model: %i' % n_larger_regtree)    pos_data.boxplot("weight", "rating")    plt.title("")    plt.ylabel("weight")    plt.tight_layout()if __name__ == '__main__':    main()