import pandas as pd
import numpy as np

from IPython.display import display

import os


def data_preparation():
    filename = os.path.join('data', 'CrowdstormingDataJuly1st.csv')
    df = pd.read_csv(filename)

    # Only players that have an image
    df_has_image = df[pd.notnull(df['photoID'])]

    """
    What do the feature names (e.g. column games) stands for?
    games: number of games in the player-referee dyad
    A referee-player dyad describes the interactions between a particular ref and one player. 
    This means that each row in the dataset is of a unique player-ref combination, 
    listing all of the games by a given player with a particular referee at any point in his career. 
    
    Which irrelevant features might be dropped?
    player, club, leagueCountry, birthday, height, weight, position, games, victories, ties, defeats, goals, photoID, 
    refNum, refCountry, Alpha_3, meanIAT, nIAT, seIAT, meanExp, nExp, seExp
    
    What relevant features might be missing, but can be computed?
    skinColor: Average of rater1 and rater2
    redPerMatch: 
    """

    # Groupping data by soccer player and referee country
    # df_aggregated = df_has_image.drop(['refNum', 'refCountry'], 1)
    df_aggregated = df_has_image.groupby(['playerShort', 'position', 'refCountry'])[
        'games', 'yellowCards', 'yellowReds', 'redCards', 'meanIAT', 'meanExp'].sum()
    df_aggregated = df_aggregated.reset_index()

    # Information of skin color for each player
    df_skin_color = df_has_image
    df_skin_color.drop_duplicates('playerShort')
    df_skin_color['skinColor'] = (df_skin_color['rater1'] + df_skin_color['rater2']) / 2
    df_skin_color = pd.DataFrame(df_skin_color[['playerShort', 'skinColor']])

    df_aggregated = pd.merge(left=df_aggregated,
                             right=df_skin_color,
                             how='left',
                             left_on='playerShort',
                             right_on='playerShort')
    df_aggregated = df_aggregated.drop_duplicates(subset=['playerShort', 'refCountry'])
    df_aggregated = df_aggregated.reset_index(drop=True)

    return df_aggregated
