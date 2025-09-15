"""
Data Preprocessing Module for League of Legends Match Analysis
Handles data loading, merging, feature engineering, and dataset preparation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class DataProcessor:
    """Handles all data preprocessing tasks for LoL match analysis"""
    
    def __init__(self):
        self.df = None
        self.processed_df = None
    
    def load_and_merge_batches(self, base_path: str = "/users/wiley/Documents/Downloads", 
                              num_batches: int = 11) -> pd.DataFrame:
        """Load and merge multiple batch files into single DataFrame"""
        print("Loading and merging batch files...")
        
        # Load main file
        df = pd.read_csv(f"{base_path}/all_matches.csv")
        print(f"Loaded main file: {len(df)} records")
        
        # Load and merge batch files
        for i in range(num_batches):
            try:
                batch_df = pd.read_csv(f"{base_path}/all_matches_batch_{i}.csv")
                df = pd.concat([df, batch_df], ignore_index=True)
                print(f"Merged batch {i}: {len(batch_df)} records")
            except FileNotFoundError:
                print(f"Batch file {i} not found, skipping...")
                continue
        
        # Save merged dataset
        output_path = f"{base_path}/all_matches_final.csv"
        df.to_csv(output_path, index=False)
        print(f"Final merged dataset: {len(df)} records saved to {output_path}")
        
        self.df = df
        return df
    
    def assign_bo5_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Identify Bo5 series and clutch games based on team matchups"""
        if df is None:
            df = self.df.copy()
        
        print("Identifying Bo5 series and clutch games...")
        
        # Get unique games sorted by ID
        game_df = df.drop_duplicates(subset='Game_ID').sort_values(by='Game_ID').reset_index(drop=True)
        
        bo5_flags = [False] * len(game_df)
        final3_game = [False] * len(game_df)
        series_start = 0
        
        # Iterate through games to identify series
        for i in range(1, len(game_df)):
            teams_now = {game_df.loc[i, 'Blue_Team'], game_df.loc[i, 'Red_Team']}
            teams_prev = {game_df.loc[i - 1, 'Blue_Team'], game_df.loc[i - 1, 'Red_Team']}
            
            # Detect new series (different team matchup)
            if teams_now != teams_prev:
                series_len = i - series_start
                
                # 3-game series where same team won first 2 games (likely Bo5)
                if (series_len == 3 and 
                    game_df.loc[series_start, 'Winning_Team'] == game_df.loc[series_start + 1, 'Winning_Team']):
                    for j in range(series_start, i):
                        bo5_flags[j] = True
                
                # Series with 4+ games (definitely Bo5)
                elif series_len >= 4:
                    for j in range(series_start, i):
                        bo5_flags[j] = True
                    # Mark final 3 games as clutch
                    for j in range(i - 3, i):
                        final3_game[j] = True
                
                series_start = i
        
        # Handle final series
        series_len = len(game_df) - series_start
        if (series_len == 3 and 
            game_df.loc[series_start, 'Winning_Team'] == game_df.loc[series_start + 1, 'Winning_Team']):
            for j in range(series_start, len(game_df)):
                bo5_flags[j] = True
        elif series_len >= 4:
            for j in range(series_start, len(game_df)):
                bo5_flags[j] = True
            for j in range(len(game_df) - 3, len(game_df)):
                final3_game[j] = True
        
        # Add flags to game dataframe
        game_df['bo5'] = bo5_flags
        game_df['final3_game'] = final3_game
        
        # Merge back with original dataframe
        df = df.merge(game_df[['Game_ID', 'bo5', 'final3_game']], on='Game_ID', how='left')
        
        print(f"Identified {sum(bo5_flags)} Bo5 games and {sum(final3_game)} clutch games")
        return df
    
    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create advanced features from raw statistics"""
        if df is None:
            df = self.df.copy()
        
        print("Engineering advanced features...")
        
        # Kill-Death-Assist efficiency
        df['KDE'] = (df['Kills'] + df['Assists']) / (df['Deaths'] + 1)
        
        # Multi-kill indicator
        df['Multi-Kill'] = np.where(
            (df['Double kills'] > 0) | 
            (df['Triple kills'] > 0) | 
            (df['Quadra kills'] > 0) | 
            (df['Penta kills'] > 0), 1, 0
        )
        
        # Damage taken per death
        df['DTPD'] = df['Total damage taken'] / (df['Deaths'] + 1)
        
        # Convert percentage strings to floats
        percentage_cols = ['DMG%', 'KP%']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip('%').astype(float) / 100
        
        print("Feature engineering completed")
        return df
    
    def prepare_role_datasets(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Prepare separate datasets for each role with selected features"""
        if df is None:
            df = self.df.copy()
        
        print("Preparing role-specific datasets...")
        
        # Define feature set for modeling
        selected_features = [
            'KDE', 'DPM', 'Multi-Kill', 'GPM', 'VSPM', 'WCPM',
            'GD@15', 'XPD@15', 'CSD@15', 'LVLD@15', 'DTPD'
        ]
        
        # Check which features are available
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        # Create clean dataset with available features
        feature_cols = available_features + ['Win', 'Role', 'Game_ID']
        df_clean = df[feature_cols].copy()
        
        # Split by role
        roles = ['Top', 'Jungle', 'Mid', 'ADC', 'Support']
        role_datasets = {}
        
        for role in roles:
            if 'Role' in df_clean.columns:
                role_data = df_clean[df_clean['Role'] == role].copy()
            else:
                # If no Role column, split by position (first 2 are Top/Jungle, etc.)
                # This is a fallback - ideally Role should be in the data
                print(f"Warning: No Role column found, using position-based assignment")
                break
            
            # Remove rows with missing values
            role_data = role_data.dropna(subset=available_features)
            role_datasets[role] = role_data
            
            print(f"{role}: {len(role_data)} samples")
        
        return role_datasets
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names used for modeling"""
        return [
            'KDE', 'DPM', 'Multi-Kill', 'GPM', 'VSPM', 'WCPM',
            'GD@15', 'XPD@15', 'CSD@15', 'LVLD@15', 'DTPD'
        ]
    
    def process_full_pipeline(self, base_path: str = "/users/wiley/Documents/Downloads") -> Tuple[pd.DataFrame, dict]:
        """Run complete preprocessing pipeline"""
        print("Starting full preprocessing pipeline...")
        
        # Step 1: Load and merge data
        df = self.load_and_merge_batches(base_path)
        
        # Step 2: Identify Bo5 series
        df = self.assign_bo5_series(df)
        
        # Step 3: Engineer features
        df = self.engineer_features(df)
        
        # Step 4: Prepare role datasets
        role_datasets = self.prepare_role_datasets(df)
        
        self.processed_df = df
        
        print("Preprocessing pipeline completed successfully!")
        return df, role_datasets


def main():
    """Example usage of DataProcessor"""
    processor = DataProcessor()
    
    # Run full pipeline
    df, role_datasets = processor.process_full_pipeline()
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Available roles: {list(role_datasets.keys())}")
    
    # Display sample statistics
    for role, data in role_datasets.items():
        win_rate = data['Win'].mean()
        print(f"{role} win rate: {win_rate:.3f}")


if __name__ == "__main__":
    main()