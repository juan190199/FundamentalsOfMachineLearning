import Worksheet06.Data.DataPreparation as DP
import Worksheet06.LinearRegression as LR
import Worksheet06.RegressionForest as RF


def main():
    df, df_mean = DP.data_preparation()

    y = df["percentageReds"].to_numpy()
    x = df.drop(labels=["percentageReds"], axis=1).to_numpy()

    lr = LR.LinearRegression(df_mean)
    lr.train(x, y)

    print(lr.predict([180, 77, 1.4, 0.8, 1, 0.4, 0.35, 0.5, 1, 0.3, 0.15, 0.3, 0.25, 0.3, 0.21, 0.1, 0.35]))

    Y = df["percentageReds"].to_numpy()
    X = df.drop(labels=["percentageReds"], axis=1).to_numpy()

    forest = RF.RegressionForest(5, df_mean)
    forest.train(X, Y, n_min=500)


if __name__ == '__main__':
    main()
