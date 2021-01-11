import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr

import os


def data_preparation():
    # Load dataset
    filename = os.path.join('data', 'CrowdstormingDataJuly1st.csv')
    df = pd.read_csv(filename, sep=",", header=0)

    # print(df.columns)

    # Remove players without image
    df = df[pd.notnull(df['photoID'])]

    # Only players that have skin color rating
    df = df[pd.notnull(df['rater1'])]
    df = df[pd.notnull(df['rater2'])]

    # Remove rows where the "refCount" < 22
    # https://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb.
    df = df.loc[df["refCount"] > 21].reset_index()
    df = df.drop(["refNum", "refCount", "index"], axis=1)


    # Drop irrelevant features
    df = df.drop(
        labels=["playerShort", "player", "photoID", "refCountry", "nIAT", "seIAT", "nExp", "seExp", "yellowCards",
                "club", "birthday", "Alpha_3", "games"], axis=1)

    # print(df.columns)

    # Information of skin color for each player
    df["rating"] = (df["rater1"] + df["rater2"]) / 2
    df = df.drop(labels=["rater1", "rater2"], axis=1)

    df["percentageReds"] = (df["redCards"] + df["yellowReds"]) / (df["victories"] + df["ties"] + df["defeats"])
    df = df.drop(labels=["redCards", "yellowReds"], axis=1)

    # One-hot-encoding for league country
    onehot = pd.get_dummies(df.leagueCountry, prefix="Country")
    df = df.drop(labels=["leagueCountry"], axis=1)
    df = pd.concat([df, onehot], axis=1, sort=False)

    dic = {"Right Fullback": "Back",
           "Left Fullback": "Back",
           "Center Back": "Back",
           "Left Midfielder": "Midfielder",
           "Right Midfielder": "Midfielder",
           "Center Midfielder": "Midfielder",
           "Defensive Midfielder": "Midfielder",
           "Attacking Midfielder": "Midfielder",
           "Left Winger": "Forward",
           "Right Winger": "Forward",
           "Center Forward": "Forward"}

    df = df.replace({"position": dic})
    onehot = pd.get_dummies(df.position, prefix="Position")
    df = df.drop(labels=["position"], axis=1)
    df = pd.concat([df, onehot], axis=1, sort=False)

    # Appearances of referees and number of referee dyads pair (refNum)
    df['refCount'] = 0
    refs = pd.unique(df['refNum'].values.ravel())  # list all unique ref IDs
    # for each ref, count their dyads
    for r in refs:
        df.loc[df['refNum'] == r, "refCount"] = len(df[df['refNum'] == r])

    # Normalize data
    defeats = df["defeats"] / (df["defeats"] + df["ties"] + df["victories"])
    ties = df["ties"] / (df["defeats"] + df["ties"] + df["victories"])
    victories = df["victories"] / (df["defeats"] + df["ties"] + df["victories"])
    df["defeats"] = defeats
    df["ties"] = ties
    df["victories"] = victories

    # Centralize data
    df_mean = df.apply(np.mean, axis=0)
    df = df - df_mean

    return df






