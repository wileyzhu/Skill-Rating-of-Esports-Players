import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# -------------------
# Config
# -------------------
# Output folder: ./data (relative to script location)
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAME_ID_RANGE = np.arange(53697, 62821)   # adjust as needed
MAX_WAIT = 5
BATCH_SIZE = 500   # number of games per batch
warnings.filterwarnings("ignore", category=Warning)


# -------------------
# Chrome setup
# -------------------
def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )


# -------------------
# Scraping a single game
# -------------------
def scrape_game(driver, game_id):
    wait = WebDriverWait(driver, MAX_WAIT)

    try:
        # Game summary page
        driver.get(f"https://gol.gg/game/stats/{game_id}/page-game/")

        # Blue side
        blue_div = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'blue-line-header')]"))
        )
        blue_text = blue_div.get_attribute("innerText").replace("\n", " ").strip()
        blue_team = blue_text.split(" - ")[0].strip()
        blue_result = "WIN" if "WIN" in blue_text.upper() else "LOSS" if "LOSS" in blue_text.upper() else "Unknown"

        # Red side
        red_div = wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'red-line-header')]"))
        )
        red_text = red_div.get_attribute("innerText").replace("\n", " ").strip()
        red_team = red_text.split(" - ")[0].strip()
        red_result = "WIN" if "WIN" in red_text.upper() else "LOSS" if "LOSS" in red_text.upper() else "Unknown"

        # Winner
        if blue_result == "WIN":
            winner = blue_team
        elif red_result == "WIN":
            winner = red_team
        else:
            return None

        # Validation
        if "Unknown" in (blue_team, red_team, blue_result, red_result, winner):
            return None

        outcomes = [1] * 5 + [0] * 5 if winner == blue_team else [0] * 5 + [1] * 5

        # Player stats page
        driver.get(f"https://gol.gg/game/stats/{game_id}/page-fullstats/")
        tables = pd.read_html(driver.page_source)
        df = tables[0].transpose()
        df.columns = df.iloc[0]
        df = df[1:]

        if len(df) != 10:
            return None

        # Add metadata
        df["Win"] = outcomes
        df["Game_ID"] = game_id
        df["Blue_Team"] = blue_team
        df["Red_Team"] = red_team
        df["Winning_Team"] = winner

        return df

    except Exception:
        return None


# -------------------
# Batch scraper
# -------------------
def scrape_batch(game_ids, batch_idx):
    driver = get_driver()
    results = []

    for gid in game_ids:
        df = scrape_game(driver, gid)
        if df is not None:
            results.append(df)

    driver.quit()

    if results:
        batch_df = pd.concat(results, ignore_index=True)
        out_file = OUTPUT_DIR / f"all_matches_batch_{batch_idx}.csv"
        batch_df.to_csv(out_file, index=False)
        print(f"[✔] Saved batch {batch_idx}: {len(batch_df)} rows")
        return batch_df
    else:
        print(f"[!] No valid matches in batch {batch_idx}")
        return None


# -------------------
# Main
# -------------------
def main():
    batches = [
        GAME_ID_RANGE[i:i + BATCH_SIZE] for i in range(0, len(GAME_ID_RANGE), BATCH_SIZE)
    ]

    all_data = []
    for idx, batch in enumerate(batches, 1):
        df = scrape_batch(batch, idx)
        if df is not None:
            all_data.append(df)

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        out_file = OUTPUT_DIR / "all_matches.csv"
        full_df.to_csv(out_file, index=False)
        print(f"[✔] Saved combined dataset: {len(full_df)} rows at {out_file}")
    else:
        print("[!] No data scraped.")


if __name__ == "__main__":
    main()
