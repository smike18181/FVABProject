import csv
import os
from PIL import Image
from PathDatasets import PATHS

def convert_gtsrb_to_yolo(annotations_file, images_base_path, output_images_path, output_labels_path, class_names):
    """
    Converte le annotazioni GTSRB nel formato YOLO e copia le immagini.

    Args:
        annotations_file (str): Percorso del file CSV delle annotazioni (Train.csv).
        images_base_path (str): Percorso base dove si trovano le cartelle delle immagini GTSRB (es. la cartella 'Train').
        output_images_path (str): Percorso dove salvare le immagini per YOLO.
        output_labels_path (str): Percorso dove salvare i file di testo delle etichette YOLO.
        class_names (dict): Dizionario che mappa ClassId numerico ai nomi delle classi YOLO.
    """
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    with open(annotations_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            width = int(row['Width'])
            height = int(row['Height'])
            x_min = int(row['Roi.X1'])
            y_min = int(row['Roi.Y1'])
            x_max = int(row['Roi.X2'])
            y_max = int(row['Roi.Y2'])
            class_id = int(row['ClassId'])
            image_path_relative = row['Path']
            image_filename = os.path.basename(image_path_relative)

            # Costruzione del percorso completo dell'immagine originale
            input_image_path = os.path.join(images_base_path, image_path_relative)

            # Costruzione del percorso di output per l'immagine
            output_image_filepath = os.path.join(output_images_path, image_filename)

            # Costruisci il percorso di output per il file di testo delle etichette
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            output_label_filepath = os.path.join(output_labels_path, label_filename)

            # Calcola le coordinate YOLO normalizzate
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # Ottieni l'ID della classe YOLO dal dizionario class_names
            if class_id in class_names:
                yolo_class_id = class_names[class_id]

                # Crea il contenuto del file di testo delle etichette YOLO
                label_content = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

                # Salva il file di testo delle etichette
                with open(output_label_filepath, 'w') as label_file:
                    label_file.write(label_content)

                # Copia l'immagine nel percorso di output
                try:
                    Image.open(input_image_path).save(output_image_filepath)
                except FileNotFoundError:
                    print(f"Errore: Immagine non trovata al percorso: {input_image_path}")
                except Exception as e:
                    print(f"Errore durante la copia dell'immagine: {e}")
            else:
                print(f"Avviso: ClassId {class_id} non trovato nel dizionario class_names. Immagine: {image_filename} non processata per le etichette.")

if __name__ == '__main__':

    # --- CONFIGURAZIONE ---
    annotations_file = PATHS["GTSRB_PATH"] + r"\Test.csv"    #oppure \
    images_base_path = PATHS["GTSRB_PATH"]
    miopercorso_immagini = PATHS["DATASET_PATH"] + r"\images\val"
    miopercorso_labels = PATHS["DATASET_PATH"] + r"\labels\val"

    # Definisci la mappatura tra ClassId di GTSRB e gli ID delle tue classi YOLO
    # Assicurati che gli ID YOLO partano da 0 e siano consecutivi per tutte le tue classi.
    # Esempio (adatta in base alle TUE classi YOLO):
    class_names_mapping = {
        stop: 0,   # Esempio: ClassId 0 di GTSRB -> ID 7 in YOLO_CLASSES
        1: 8,   # Esempio: ClassId 1 di GTSRB -> ID 8 in YOLO_CLASSES
        2: 9,
        3: 10,
        4: 11,
        5: 12,
        6: 13,
        7: 14,
        8: 15,
        9: 16,
        10: 17,
        11: 18,
        12: 19,
        13: 20,
        14: 21,
        15: 22,
        16: 23,
        17: 24,
        18: 25,
        19: 26,
        20: 27,
        21: 28,
        22: 29,
        23: 30,
        24: 31,
        25: 32,
        26: 33,
        27: 34,
        28: 35,
        29: 36,
        30: 37,
        31: 38,
        32: 39,
        33: 40,
        34: 41,
        35: 42,
        36: 43,
        37: 44,
        38: 45,
        39: 46,
        40: 47,
        41: 48,
        42: 49,
    }

    convert_gtsrb_to_yolo(annotations_file, images_base_path, miopercorso_immagini, miopercorso_labels, class_names_mapping)

    print(f"\nConversione completata.")
    print(f"Immagini salvate in: {miopercorso_immagini}")
    print(f"Etichette YOLO salvate in: {miopercorso_labels}")