import os
import re

# Percorso alla cartella contenente i file txt
folder_path = "combined_dataset/labels/train"  # <-- Sostituisci con il tuo percorso se diverso

# Regex per trovare i file con formato "dayClip6--000XX.txt"
pattern = re.compile(r"dayClip6--(\d{5})\.txt", re.IGNORECASE)

# Lista dei file validi con il numero estratto
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
    if index % 2 == 1:  # Elimina ogni secondo file
        try:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Cancellato: {filename}")
        except Exception as e:
            print(f"Errore nel cancellare {filename}: {e}")
