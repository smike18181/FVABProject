import os

# Percorso alla cartella contenente i file .txt
folder_path = "combined_dataset/labels/train"  # <-- Modifica se il percorso è diverso

# Itera su tutti i file nella cartella
for filename in os.listdir(folder_path):
    if not filename.endswith(".txt"):
        continue  # Salta i non-txt

    try:
        parts = filename.split("_")
        if len(parts) < 2:
            continue  # Formato non valido

        middle = parts[1][:5]  # Prende i primi 5 caratteri dopo il primo underscore

        if middle > "00006":
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Cancellato: {filename}")
    except Exception as e:
        print(f"Errore con il file {filename}: {e}")
