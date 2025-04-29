import csv
import os
from PIL import Image
from PathDatasets import PATHS

def convert_lisa_to_yolo(annotations_file, images_base_path, output_images_path, output_labels_path, class_mapping):
    """
    Converte le annotazioni nel formato YOLO per la struttura specifica del file CSV LISA.

    Args:
        annotations_file (str): Percorso del file CSV delle annotazioni LISA.
        images_base_path (str): Percorso base dove si trovano le immagini LISA (es. la cartella principale con le sottocartelle 'dayTraining', etc.).
        output_images_path (str): Percorso dove salvare le immagini per YOLO.
        output_labels_path (str): Percorso dove salvare i file di testo delle etichette YOLO.
        class_mapping (dict): Dizionario che mappa i tag LISA ai nomi delle classi YOLO.
                              Es: {'stop': 0, 'speedLimitUrdbl': 1, ...}.
    """
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    processed_images = set()  # Tiene traccia delle immagini già processate

    with open(annotations_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:

            filename_with_path = row['Filename']
            filename_only = filename_with_path.split('/')[-1]
            filename = filename_only.replace('/', '\\')

            annotation_tag = row['Annotation tag']
            x_min = int(row['Upper left corner X'])
            y_min = int(row['Upper left corner Y'])
            x_max = int(row['Lower right corner X'])
            y_max = int(row['Lower right corner Y'])

            # Costruzione del percorso completo dell'immagine originale
            input_image_path = os.path.join(images_base_path, filename)
            image_filename = os.path.basename(filename)
            output_image_filepath = os.path.join(output_images_path, image_filename)

            # Costruisci il percorso di output per il file di testo delle etichette
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            output_label_filepath = os.path.join(output_labels_path, label_filename)

            try:
                img = Image.open(input_image_path)
                width = img.width
                height = img.height

                # Calcola le coordinate YOLO normalizzate
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

                # Ottieni l'ID della classe YOLO dal dizionario class_mapping
                if annotation_tag in class_mapping:
                    yolo_class_id = class_mapping[annotation_tag]

                    # Crea il contenuto del file di testo delle etichette YOLO
                    label_content = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

                    # Se l'immagine non è ancora stata processata, crea il file di etichette
                    if image_filename not in processed_images:
                        with open(output_label_filepath, 'w') as label_file:
                            label_file.write(label_content)
                        processed_images.add(image_filename) # Segna l'immagine come processata
                    else:
                        # Se l'immagine è già stata processata, aggiungi l'etichetta al file esistente
                        with open(output_label_filepath, 'a') as label_file:
                            label_file.write(label_content)

                    # Copia l'immagine nel percorso di output solo se non è già stata copiata
                    if not os.path.exists(output_image_filepath):
                        img.save(output_image_filepath)

                else:
                    print(f"Avviso: Annotation tag '{annotation_tag}' non trovato nel dizionario class_mapping. Oggetto in '{filename}' non etichettato.")

            except FileNotFoundError:
                print(f"Errore: Immagine non trovata al percorso: {input_image_path}")
            except Exception as e:
                print(f"Errore durante la lettura o copia dell'immagine: {e}")

if __name__ == '__main__':

    # --- CONFIGURAZIONE ---
    annotations_file = PATHS["LISA_PATH"] + r"\sample-nightClip1\sample-nightClip1\frameAnnotationsBOX.csv"
    images_base_path = PATHS["LISA_PATH"] + r"\sample-nightClip1\sample-nightClip1\frames"
    miopercorso_immagini = PATHS["DATASET_PATH"] + r"\images\val"
    miopercorso_labels = PATHS["DATASET_PATH"] + r"\labels\val"

    class_names_mapping = {
        "stop": 50,   # Esempio: AnnotazionTag: stop di LISA -> ID 50 in YOLO_CLASSES
        "stopLeft": 51,
        "go": 52,
        "goLeft": 53,
        "warning": 54,
        "warningLeft": 55
    }

    convert_lisa_to_yolo(annotations_file, images_base_path, miopercorso_immagini, miopercorso_labels, class_names_mapping)

    print(f"\nConversione completata.")
    print(f"Immagini salvate in: {miopercorso_immagini}")
    print(f"Etichette YOLO salvate in: {miopercorso_labels}")