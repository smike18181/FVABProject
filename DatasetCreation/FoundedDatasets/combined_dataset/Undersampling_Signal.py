import os

# Imposta il percorso alla cartella contenente le immagini
folder_path = "combined_dataset/images/train"  # <-- Sostituisci con il tuo percorso

# Itera su tutti i file nella cartella
for filename in os.listdir(folder_path):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif")):
        continue  # Salta i file non immagine

    try:
        parts = filename.split("_")
        if len(parts) < 2:
            continue  # formato non valido, salta

        middle = parts[1][:5]  # Prende i primi 5 caratteri della parte centrale

        if middle > "00006":
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Cancellato: {filename}")
    except Exception as e:
        print(f"Errore con il file {filename}: {e}")
