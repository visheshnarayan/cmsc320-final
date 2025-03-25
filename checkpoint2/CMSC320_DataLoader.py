import os
import librosa
import polars as pl

csv_path = "ReCANVo/renamed_metadata.csv"
audio_path = "ReCANVo/"

df = pl.read_csv(csv_path)

audio_data = []
for row in df.head().iter_rows(named=True):
    file_name = row['Filename']
    file_path = os.path.join(audio_path, file_name)

    y, sr = librosa.load(file_path, sr=None) 
    duration = len(y) / sr
    audio_data.append((file_name, y.tolist(), row['ID'], row['Label'], duration,row['Index']))

audio_df = pl.DataFrame(audio_data, schema=["Filename", "Audio", "ID", "Label", "Duration","Index"], orient='row')

print(audio_df)