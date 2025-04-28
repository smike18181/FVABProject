import xml.etree.ElementTree as ET
import os
from PathDatasets import PATHS

def convert_xml_to_yolo(xml_path, txt_path, class_names):
    """
    Converte un file di annotazione XML (PASCAL VOC) nel formato YOLO .txt.

    Args:
        xml_path (str): Il percorso del file XML di input.
        txt_path (str): Il percorso del file .txt di output da creare.
        class_names (list): Una lista di stringhe contenente i nomi delle classi.
                             L'ordine della lista determina l'indice della classe (a partire da 0).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in class_names:
                    print(f"Warning: Classe '{name}' non trovata in class_names. Oggetto ignorato in '{xml_path}'.")
                    continue
                class_id = class_names.index(name)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        print(f"File XML '{xml_path}' convertito con successo in '{txt_path}'.")

    except FileNotFoundError:
        print(f"Errore: Il file XML '{xml_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore durante la conversione di '{xml_path}': {e}")

def convert_folder_xml_to_yolo(xml_folder, output_folder, class_names):
    """
    Itera su tutti i file XML in una cartella e li converte nel formato YOLO .txt.

    Args:
        xml_folder (str): Il percorso della cartella contenente i file XML.
        output_folder (str): Il percorso della cartella dove salvare i file .txt convertiti.
        class_names (list): Una lista di stringhe contenente i nomi delle classi.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(xml_folder):
        if filename.endswith(".xml"):
            xml_filepath = os.path.join(xml_folder, filename)
            txt_filename = filename.replace(".xml", ".txt")
            txt_filepath = os.path.join(output_folder, txt_filename)
            convert_xml_to_yolo(xml_filepath, txt_filepath, class_names)

if __name__ == '__main__':
    # Esempio di utilizzo per una cartella:
    xml_annotations_folder = PATHS['PEDESTRIAN_PATH'] + r"\Test\Test\Annotations"
    yolo_annotations_folder = PATHS['DATASET_PATH'] + r"\labels\val"
    class_list = ['person']

    convert_folder_xml_to_yolo(xml_annotations_folder, yolo_annotations_folder, class_list)

    print("Conversione di tutti i file XML nella cartella completata.")