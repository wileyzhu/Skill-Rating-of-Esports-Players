import pandas as pd

def load_and_merge_data(base_path="/users/wiley/Documents/Downloads", output_file="all_matches_final.csv"):
    df = pd.read_csv(f"{base_path}/all_matches.csv")
    for i in range(0, 11):
        batch_df = pd.read_csv(f"{base_path}/all_matches_batch_{i}.csv")
        df = pd.concat([df, batch_df], ignore_index=True)
    output_path = f"{base_path}/{output_file}"
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    print(f"Merged data shape: {df.shape}")