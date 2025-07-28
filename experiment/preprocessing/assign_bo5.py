import pandas as pd

def assign_bo5(df):
    game_df = df.drop_duplicates(subset='Game_ID').sort_values(by='Game_ID').reset_index(drop=True)
    bo5_flags = [False] * len(game_df)
    final3_game = [False] * len(game_df)
    series_start = 0

    for i in range(1, len(game_df)):
        teams_now = {game_df.loc[i, 'Blue_Team'], game_df.loc[i, 'Red_Team']}
        teams_prev = {game_df.loc[i - 1, 'Blue_Team'], game_df.loc[i - 1, 'Red_Team']}
        if teams_now != teams_prev:
            series_len = i - series_start
            if series_len == 3 and game_df.loc[series_start, 'Winning_Team'] == game_df.loc[series_start + 1, 'Winning_Team']:
                for j in range(series_start, i):
                    bo5_flags[j] = True
            elif series_len >= 4:
                for j in range(series_start, i):
                    bo5_flags[j] = True
                for j in range(i - 3, i):
                    final3_game[j] = True
            series_start = i

    series_len = len(game_df) - series_start
    if series_len == 2 and game_df.loc[series_start, 'Winning_Team'] == game_df.loc[series_start + 1, 'Winning_Team']:
        for j in range(series_start, len(game_df)):
            bo5_flags[j] = True
    elif series_len >= 3:
        for j in range(series_start, len(game_df)):
            bo5_flags[j] = True
        for j in range(len(game_df) - 3, len(game_df)):
            final3_game[j] = True

    game_df['bo5'] = bo5_flags
    game_df['final3_game'] = final3_game
    df = df.merge(game_df[['Game_ID', 'bo5', 'final3_game']], on='Game_ID', how='left')
    return df