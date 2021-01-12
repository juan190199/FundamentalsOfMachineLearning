import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr

import os


def data_preparation():
    """
    What do the feature names (e.g. column games) stands for?
    games: number of games in the player-referee dyad
    A referee-player dyad describes the interactions between a particular ref and one player.
    This means that each row in the dataset is of a unique player-ref combination,
    listing all of the games by a given player with a particular referee at any point in his career.

    How good are the skin color ratings? Do the raters agree?
    The skin ratings are subjective. This leads to disregarding ratings

    Should referees with very few appearances be excluded from the dataset?
    No. There is no index that implies a relation between the numbers of appearances of a referee and the number of red
    cards
    """
    # Load dataset
    filename = os.path.join('data', 'CrowdstormingDataJuly1st.csv')
    df = pd.read_csv(filename, sep=",", header=0)

    # print(df.columns)

    # Remove instances with NaN data
    df = df.dropna(axis=0)

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

    # Number of games where referee is involved
    df['refCount'] = 0
    refs = pd.unique(df['refNum'].values.ravel())  # list all unique ref IDs
    # for each ref, count their dyads
    for r in refs:
        df.loc[df['refNum'] == r, "refCount"] = len(df[df['refNum'] == r])

    # Remove rows where the "refNum" < 22
    # https://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb.
    df = df.loc[df["refCount"] > 21].reset_index()
    df = df.drop(["refNum", "refCount", "index"], axis=1)

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

    return df, df_mean






