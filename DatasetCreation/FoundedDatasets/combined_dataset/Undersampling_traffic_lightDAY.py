import os
import re

# Percorso alla cartella contenente le immagini
folder_path = "combined_dataset/images/train"  # <-- Sostituiscilo con il tuo path reale

# Regex per trovare file nel formato "dayClip6--000XX" con estensione
pattern = re.compile(r"dayClip6--(\d+)\.(jpg|png|jpeg|bmp|tif)", re.IGNORECASE)

# Lista dei file validi ordinati numericamente
matching_files = []

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        matching_files.append((number, filename))

# Ordina i file per numero
matching_files.sort()

# Elimina ogni secondo file
for index, (number, filename) in enumerate(matching_files):
    if index % 2 == 1:  # cancella ogni secondo
        try:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Cancellato: {filename}")
        except Exception as e:
            print(f"Errore nel cancellare {filename}: {e}")
