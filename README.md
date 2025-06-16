# League of Legends Tier 1 Match Scraper (Gol.gg)

This Python project extracts detailed match data from [Gol.gg](https://gol.gg) for professional League of Legends games (e.g., Worlds, LEC, LCK). It scrapes game-by-game statistics from full match pages and saves structured data into CSV format for use in analytics or machine learning projects.

## Features

- Collects **player-level stats** from the `page-fullstats` endpoint
- Retrieves **win/loss outcome** and **team names** from `page-game`
- Automatically handles errors (e.g., invalid pages, skipped games)
- Saves data to CSV for further use in model training or evaluation
- Designed to work in **headless mode** (does not open Chrome visibly)
- Optionally restart ChromeDriver to avoid long-running session issues

## Requirements

- Python 3.8+
- Google Chrome installed
- ChromeDriver (automatically managed)
- Recommended: Conda or venv environment

### Python Dependencies

```bash
pip install selenium pandas numpy webdriver-manager
Optional (for parsing HTML more robustly):
pip install html5lib
How It Works
	1.	You define a list or range of game_ids to scrape (e.g., [62733, 62734, ...])
	2.	The script visits each match page on Gol.gg and:
	•	Extracts team names and winner
	•	Collects detailed stats from the page-fullstats table
	•	Adds metadata (team, game ID, winner)
	3.	Scraped data is appended to a list and saved to all_matches.csv

Usage
	1.	Clone this repository or copy the script to your project.
	2.	Edit the list of game_ids inside the script.
	3.	Run the scraper:
python match_scraper.py
	4.	Output CSV will be saved to:
You can customize this output path.

Known Limitations
	•	Gol.gg game IDs are not sequential and include Tier 2/Tier 3 matches — consider filtering after scraping based on team names or tournaments.
	•	ChromeDriver can become unstable after ~1000 pages — recommended to restart the driver every 500–1000 matches.
	•	Some pages may lack full stats (e.g., forfeits) — they will be skipped automatically.

Future Improvements
	•	Match list auto-scraper (scrape from tournament matchlists directly)
	•	Add support for Bo3/Bo5 series parsing
	•	Filter only Tier 1 tournaments by team whitelist
	•	Export data in Parquet/Feather for faster analysis
