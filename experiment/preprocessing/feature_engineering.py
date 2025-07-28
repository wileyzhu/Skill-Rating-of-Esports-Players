import numpy as np
import pandas as pd

def engineer_features(df):
    df['KDE'] = (df['Kills'] + df['Assists']) / (df['Deaths'] + 1)
    df['Multi-Kill'] = np.where(
        (df['Double kills'] > 0) | 
        (df['Triple kills'] > 0) | 
        (df['Quadra kills'] > 0) | 
        (df['Penta kills'] > 0), 1, 0
    )
    df['DTPD'] = df['Total damage taken'] / (df['Deaths'] + 1)
    df['DMG%'] = df['DMG%'].str.strip('%').astype(float) / 100
    df['KP%'] = df['KP%'].str.strip('%').astype(float) / 100
    return df