import polars as pl
import os

csv_path = "ReCANVo/dataset_file_directory.csv"
audio_path = "ReCANVo/"
output_path = "RenamedFiles"
df = pl.read_csv(csv_path)

renamed_files = []
file_counts = {}
for file in df.iter_rows(named=True):
    org_name = file['Filename']
    id = file['Participant']
    label = file['Label']

    key = (id, label)
    file_counts[key] = file_counts.get(key, 0) + 1
    index = file_counts[key]

    new_name = f"{id}_{label}_{index}.wav"
    old_path = os.path.join(audio_path, org_name)
    new_path = os.path.join(audio_path, new_name)
    
    os.rename(old_path, new_path)
    renamed_files.append((new_name, id, label, index))

renamed_df = pl.DataFrame(renamed_files, schema=["Filename", "ID", "Label", "Index"])

renamed_df.write_csv(os.path.join(audio_path, "renamed_metadata.csv"))

renamed_df = pl.read_csv(os.path.join(audio_path, "renamed_metadata.csv"))
print(renamed_df)