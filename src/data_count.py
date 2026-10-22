# pip install datasets pandas
from datasets import load_dataset
import pandas as pd

# 1. Load the dataset in streaming mode (doesn't download the whole thing)
dataset = load_dataset("theelderemo/genius-lyrics-cleaned", split="train", streaming=True)

# 2. Filter for Pop songs from 2020-2025
filtered_stream = dataset.filter(lambda x: x["year"] >= 2020 and x["year"] <= 2025 and x["tag"] == "pop")

# 3. Grab exactly 3,000 songs for your control group and load them into a list
control_songs = []
for song in filtered_stream:
    control_songs.append(song)
    if len(control_songs) >= 3000:
        break

# 4. Convert to a Pandas dataframe so you can use Regex and spaCy on it
df = pd.DataFrame(control_songs)

print(f"Successfully loaded {len(df)} songs.")
print(df.head())