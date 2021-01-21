import Worksheet06v2.Data.DataPreparation as DPimport Worksheet06v2.Data.myDataset as mDimport Worksheet06v2.RegressionTree as RTfrom sklearn import linear_modelfrom sklearn import model_selectionimport numpy as npdef evaluate(prediction, target):    assert prediction.shape == target.shape    N = prediction.shape[0]    return np.sum(np.square(prediction - target)) / Ndef k_fold_cv(mod, feats, targets, k):    assert targets.shape[0] == feats.shape[0]    kf = model_selection.KFold(n_splits=k)    err_train, err_test = [], []    for train, test in kf.split(feats):        mod.fit(feats[train], targets[train])        # Estimate for the training error, evaluated on the whole data        err_train.append(evaluate(mod.predict(feats), targets))        # Estimate for the test error, evaluated on the test fold        err_test.append(evaluate(mod.predict(feats[test]), targets[test]))    return (np.mean(err_test), np.std(err_test), np.mean(err_train), np.std(err_train))def test_data(linear, regtree, data, n_shufflesets, orig_err_linear, orig_err_regtree, pred_variable="rating"):    shufflesets = []    # Set a random seed to make results reproducible    np.random.seed(1089045)    # Shuffling works inplace, therefore we retrieve a copy    rating = np.array(data[pred_variable]).copy()    for i in range(n_shufflesets):        np.random.shuffle(rating)        shuffled_data = data.copy()        shuffled_data[pred_variable] = rating        shufflesets.append(mD.myDataset(shuffled_data))    # Evaluate them:    shuffled_err_linear = []    shuffled_err_regtree = []    for i, dset in enumerate(shufflesets):        print('Run shuffleset no %i' % (i+1))        feats = dset.get_X_oc_()        targets = dset.targets        assert feats.shape[0] == targets.shape[0]        k = 10        err_test, std_err_test, err_train, std_err_train = k_fold_cv(linear, feats, targets, k)        shuffled_err_linear.append(err_test)        err_test, std_err_test, err_train, std_err_train = k_fold_cv(regtree, feats, targets, k)        shuffled_err_regtree.append(err_test)    n_larger_linear = (shuffled_err_linear > orig_err_linear).sum()    n_larger_regtree = (shuffled_err_regtree > orig_err_regtree).sum()    return n_larger_linear, n_larger_regtreedef main():    # Data preparation    prediction_data = DP.dataPreparation()    dset = mD.myDataset(prediction_data)    # Define the different models    linear = linear_model.LinearRegression()    regtree = RT.RegressionTree()    # Calculate mean squared error    # Evaluate all model / feature set combinations    k = 10    for mod in (linear, regtree):        print("Running:", mod)        for feats_str in ("X_oc", "X_on", "X_oc_"):            print("Dataset: %s" % feats_str)            feats = eval("dset.get_" + feats_str + "()")            targets = dset.targets            assert feats.shape[0] == targets.shape[0]            err_test, std_err_test, err_train, std_err_train = k_fold_cv(mod, feats, targets, k)            print('test_err: %.2E' % err_test)    print("Done.")if __name__ == '__main__':    main()