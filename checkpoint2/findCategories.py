import pandas as pd
import os

# Load the CSV
csv_path = "ReCANVo/dataset_file_directory.csv"
df = pd.read_csv(csv_path, header=None, names=["filename", "PXX", "category"])

# Extract original filenames (without .wav extensions)
df["original_filename"] = df["filename"].apply(lambda x: os.path.splitext(x)[0])

# Get renamed files
audio_dir = "ReCANVo/"
renamed_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

# Create a dictionary to store the final mappings
file_category_mapping = {}

# Loop through renamed files and find their correct categories
for file in renamed_files:
    parts = file.split("_")  # Assuming format "PXX_category_N.wav"
    if len(parts) > 1:
        category = parts[1]  # Extract category from filename
        
        # Try to match this renamed file back to an entry in the CSV
        matching_row = df[df["category"].str.contains(category, case=False, na=False)]
        
        if not matching_row.empty:
            file_category_mapping[file] = matching_row.iloc[0]["category"]
        else:
            file_category_mapping[file] = "UNKNOWN"  # Fallback if no match found

# Print the first few mappings
for i, (file, category) in enumerate(file_category_mapping.items()):
    print(f"{file}: {category}")
    if i > 10:  # Print only a few for readability
        break
