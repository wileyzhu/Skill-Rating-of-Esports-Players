import pandas as pd

def select_features(df):
    selected_features = [
        'KDE', 'DPM', 'Multi-Kill', 'GPM', 'VSPM', 'WCPM',
        'GD@15', 'XPD@15', 'CSD@15', 'LVLD@15', 'DTPD'
    ]
    df_clean = df[selected_features + ['Win']].copy()
    df_clean['Role'] = df['Role']
    df_clean['Game_ID'] = df['Game_ID']
    return df_clean