import zipfile

# Define the source zip file and the destination directory
zip_path = "kanjivg-all.zip"
extract_path = "kanji/"

# Open the zip file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all contents to the specified directory
    zip_ref.extractall(extract_path)

print(f"Files extracted successfully to {extract_path}")
