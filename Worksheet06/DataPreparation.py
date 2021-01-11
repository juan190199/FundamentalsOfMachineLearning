import pandas as pd
import numpy as np

from IPython.display import display

import os


def data_preparation():
    filename = os.path.join('data', 'CrowdstormingDataJuly1st.csv')
    df = pd.read_csv(filename)

    # Only players that have an image
    df_has_image = df[pd.notnull(df['photoID'])]

    # Groupping data by soccer player
    df_aggregated = df_has_image.drop(['refNum', 'refCountry'], 1)
    df_aggregated = df_aggregated.groupby(['playerShort', 'position'])[
        'games', 'yellowCards', 'yellowReds', 'redCards'].sum()
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
    df_aggregated = df_aggregated.drop_duplicates('playerShort')
    df_aggregated = df_aggregated.reset_index(drop=True)

    return df_aggregated
