"""
Data Collection Module for League of Legends Match Scraping
Scrapes professional match data from Gol.gg using Selenium
"""

import numpy as np
import pandas as pd
import time
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class MatchScraper:
    """Scrapes League of Legends professional match data from Gol.gg"""
    
    def __init__(self, headless=True, max_games_per_session=300):
        self.headless = headless
        self.max_games_per_session = max_games_per_session
        self.driver = None
        self.wait = None
        warnings.filterwarnings("ignore", category=Warning)
    
    def _setup_driver(self):
        """Initialize Chrome WebDriver with options"""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.wait = WebDriverWait(self.driver, 5)
    
    def _extract_team_info(self, game_id):
        """Extract team names and winner from game page"""
        self.driver.get(f"https://gol.gg/game/stats/{game_id}/page-game/")
        
        # Extract blue team info
        blue_div = self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'blue-line-header')]"))
        )
        blue_text = blue_div.get_attribute("innerText").replace('\n', ' ').strip()
        blue_team_name = blue_text.split(' - ')[0].strip() if ' - ' in blue_text else blue_text.strip()
        blue_result = "WIN" if "WIN" in blue_text.upper() else "LOSS" if "LOSS" in blue_text.upper() else "Unknown"
        
        # Extract red team info
        red_div = self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'red-line-header')]"))
        )
        red_text = red_div.get_attribute("innerText").replace('\n', ' ').strip()
        red_team_name = red_text.split(' - ')[0].strip() if ' - ' in red_text else red_text.strip()
        red_result = "WIN" if "WIN" in red_text.upper() else "LOSS" if "LOSS" in red_text.upper() else "Unknown"
        
        # Determine winner
        if blue_result == "WIN":
            winning_team_name = blue_team_name
        elif red_result == "WIN":
            winning_team_name = red_team_name
        else:
            return None
        
        # Validate data quality
        if any(team == "Unknown" for team in [blue_team_name, red_team_name, winning_team_name]):
            return None
        
        return {
            'blue_team': blue_team_name,
            'red_team': red_team_name,
            'winning_team': winning_team_name
        }
    
    def _extract_match_stats(self, game_id, team_info):
        """Extract detailed player statistics from match"""
        self.driver.get(f"https://gol.gg/game/stats/{game_id}/page-fullstats/")
        
        # Parse match statistics table
        tables = pd.read_html(self.driver.page_source)
        match_df = tables[0].transpose()
        match_df.columns = match_df.iloc[0]
        match_df = match_df[1:]
        
        if len(match_df) != 10:  # Should have exactly 10 players
            return None
        
        # Assign win/loss outcomes (first 5 players are blue team)
        outcomes = ([1]*5 + [0]*5 if team_info['winning_team'] == team_info['blue_team'] 
                   else [0]*5 + [1]*5)
        
        # Add metadata
        match_df["Win"] = outcomes
        match_df["Game_ID"] = game_id
        match_df["Blue_Team"] = team_info['blue_team']
        match_df["Red_Team"] = team_info['red_team']
        match_df["Winning_Team"] = team_info['winning_team']
        
        return match_df
    
    def scrape_games(self, game_ids, output_path=None):
        """Scrape multiple games and return combined DataFrame"""
        self._setup_driver()
        all_matches = []
        games_processed = 0
        
        try:
            for game_id in game_ids:
                try:
                    # Extract team information
                    team_info = self._extract_team_info(game_id)
                    if not team_info:
                        continue
                    
                    # Extract match statistics
                    match_df = self._extract_match_stats(game_id, team_info)
                    if match_df is not None:
                        all_matches.append(match_df)
                    
                    games_processed += 1
                    
                    # Restart driver periodically to prevent memory issues
                    if games_processed % self.max_games_per_session == 0:
                        self.driver.quit()
                        self._setup_driver()
                        
                except Exception as e:
                    print(f"Error processing game {game_id}: {e}")
                    continue
                    
        finally:
            if self.driver:
                self.driver.quit()
        
        # Combine and save results
        if all_matches:
            result_df = pd.concat(all_matches, ignore_index=True)
            if output_path:
                result_df.to_csv(output_path, index=False)
                print(f"Saved {len(result_df)} records to {output_path}")
            return result_df
        else:
            print("No valid matches found")
            return pd.DataFrame()
    
    def scrape_batch(self, game_ids, batch_idx, output_dir="/users/wiley/Documents/Downloads"):
        """Scrape a batch of games and save to separate file"""
        output_path = f"{output_dir}/all_matches_batch_{batch_idx}.csv"
        return self.scrape_games(game_ids, output_path)


def main():
    """Example usage of the MatchScraper"""
    scraper = MatchScraper(headless=True)
    
    # Define game ID ranges to scrape
    game_ids = np.arange(53697, 54000)  # Example range
    
    # Scrape games
    df = scraper.scrape_games(game_ids, "/users/wiley/Documents/Downloads/example_matches.csv")
    print(f"Scraped {len(df)} match records")


if __name__ == "__main__":
    main()