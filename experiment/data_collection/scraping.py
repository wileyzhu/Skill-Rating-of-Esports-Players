# %%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# %%
import numpy as np
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import warnings

game_ids = np.arange(53697, 62820)
warnings.filterwarnings("ignore", category=Warning)

# Set Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

def get_driver():
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

driver = get_driver()
games_processed = 0
max_games_per_session = 300
all_matches = []

wait = WebDriverWait(driver, 5)

for game_id in game_ids:
    try:
        driver.get(f"https://gol.gg/game/stats/{game_id}/page-game/")
        blue_div = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'blue-line-header')]"))
        )
        blue_text = blue_div.get_attribute("innerText").replace('\n', ' ').strip()
        blue_team_name = blue_text.split(' - ')[0].strip() if ' - ' in blue_text else blue_text.strip()
        blue_result = "WIN" if "WIN" in blue_text.upper() else "LOSS" if "LOSS" in blue_text.upper() else "Unknown"

        red_div = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'red-line-header')]"))
        )
        red_text = red_div.get_attribute("innerText").replace('\n', ' ').strip()
        red_team_name = red_text.split(' - ')[0].strip() if ' - ' in red_text else red_text.strip()
        red_result = "WIN" if "WIN" in red_text.upper() else "LOSS" if "LOSS" in red_text.upper() else "Unknown"

        if blue_result == "WIN":
            winning_team_name = blue_team_name
        elif red_result == "WIN":
            winning_team_name = red_team_name
        else:
            continue

        if (
            blue_team_name == "Unknown" or
            red_team_name == "Unknown" or
            winning_team_name == "Unknown" or
            blue_result == "Unknown" or
            red_result == "Unknown"
        ):
            continue

        outcomes = [1]*5 + [0]*5 if winning_team_name == blue_team_name else [0]*5 + [1]*5

        driver.get(f"https://gol.gg/game/stats/{game_id}/page-fullstats/")
        tables = pd.read_html(driver.page_source)
        match_df = tables[0].transpose()
        match_df.columns = match_df.iloc[0]
        match_df = match_df[1:]

        if len(match_df) != 10:
            continue

        match_df["Win"] = outcomes
        match_df["Game_ID"] = game_id
        match_df["Blue_Team"] = blue_team_name
        match_df["Red_Team"] = red_team_name
        match_df["Winning_Team"] = winning_team_name

        all_matches.append(match_df)

        games_processed += 1
        if games_processed % max_games_per_session == 0:
            driver.quit()
            driver = get_driver()
            wait = WebDriverWait(driver, 5)
    except Exception:
        continue
driver.quit()


# Save final dataset
if all_matches:
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    all_matches_df.to_csv("/users/wiley/Documents/Downloads/all_matches.csv", index=False)
    print("[✔] Saved all data to all_matches.csv")
    print(all_matches_df.head())
else:
    print("[!] No valid matches found.")

# %%
len(all_matches)

# %%
import numpy as np
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import warnings

game_ids = np.arange(54721, 62821)
warnings.filterwarnings("ignore", category=Warning)

# Set Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

def get_driver():
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

wait = WebDriverWait(driver, 5)

def run_scraper_on_batch(batch, batch_idx):
    driver = get_driver()
    wait = WebDriverWait(driver, 5)
    games_processed = 0
    all_matches1 = []
    for game_id in batch:
        try:
            driver.get(f"https://gol.gg/game/stats/{game_id}/page-game/")
            blue_div = wait.until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'blue-line-header')]"))
            )
            blue_text = blue_div.get_attribute("innerText").replace('\n', ' ').strip()
            blue_team_name = blue_text.split(' - ')[0].strip() if ' - ' in blue_text else blue_text.strip()
            blue_result = "WIN" if "WIN" in blue_text.upper() else "LOSS" if "LOSS" in blue_text.upper() else "Unknown"

            red_div = wait.until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'red-line-header')]"))
            )
            red_text = red_div.get_attribute("innerText").replace('\n', ' ').strip()
            red_team_name = red_text.split(' - ')[0].strip() if ' - ' in red_text else red_text.strip()
            red_result = "WIN" if "WIN" in red_text.upper() else "LOSS" if "LOSS" in red_text.upper() else "Unknown"

            if blue_result == "WIN":
                winning_team_name = blue_team_name
            elif red_result == "WIN":
                winning_team_name = red_team_name
            else:
                continue

            if (
                blue_team_name == "Unknown" or
                red_team_name == "Unknown" or
                winning_team_name == "Unknown" or
                blue_result == "Unknown" or
                red_result == "Unknown"
            ):
                continue

            outcomes = [1]*5 + [0]*5 if winning_team_name == blue_team_name else [0]*5 + [1]*5

            driver.get(f"https://gol.gg/game/stats/{game_id}/page-fullstats/")
            tables = pd.read_html(driver.page_source)
            match_df = tables[0].transpose()
            match_df.columns = match_df.iloc[0]
            match_df = match_df[1:]

            if len(match_df) != 10:
                continue

            match_df["Win"] = outcomes
            match_df["Game_ID"] = game_id
            match_df["Blue_Team"] = blue_team_name
            match_df["Red_Team"] = red_team_name
            match_df["Winning_Team"] = winning_team_name

            all_matches1.append(match_df)
            games_processed += 1
        except Exception:
            continue
    driver.quit()
    if all_matches1:
        batch_df = pd.concat(all_matches1, ignore_index=True)
        batch_df.to_csv(f"/users/wiley/Documents/Downloads/all_matches_batch_{batch_idx}.csv", index=False)
        print(f"[✔] Saved batch {batch_idx} with {len(batch_df)} rows.")
    else:
        print(f"[!] No valid matches found in batch {batch_idx}.")
