import pandas as pd
import numpy as np

# Load the dataset
file_path = "RT_IOT2022.csv"
data = pd.read_csv(file_path)

# Shuffle the rows multiple times for more randomness
for _ in range(5):
    data = data.sample(frac=1, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

# Save the shuffled dataset back to a file
shuffled_file_path = "RT_IOT2022_shuffled.csv"
try:
    data.to_csv(shuffled_file_path, index=False)
    print(f"Shuffled dataset saved to {shuffled_file_path}")
except PermissionError:
    temp_file_path = "RT_IOT2022_shuffled_temp.csv"
    print(f"Permission denied for {shuffled_file_path}. Saving to a temporary file instead.")
    data.to_csv(temp_file_path, index=False)
    print(f"Shuffled dataset saved to {temp_file_path}")