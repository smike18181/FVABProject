# feature_extraction.py
from pathlib import Path
import cv2
import logging # Aggiunto per loggare errori potenziali
from PathDatasets import PATHS # Importa il dizionario PATHS dal file

# Configura un logger base (opzionale ma utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetFeatures:
    def __init__(self):
        self.datasets = {
            "GTSRB": PATHS.get("GTSRB_PATH"),
            "LISA": PATHS.get("LISA_PATH"),
            "VeRi": PATHS.get("VeRi-776_PATH") # Assicurati che la chiave corrisponda
        }
        # Verifica che i percorsi siano stati trovati
        for name, path in self.datasets.items():
            if not path:
                logging.warning(f"Percorso per {name} non trovato nel dizionario PATHS.")
            elif not Path(path).exists():
                 logging.warning(f"Percorso per {name} non trovato: {path}")


    def get_basic_stats(self, dataset_name):
        """Estrae statistiche base e struttura del dataset"""
        path = self.datasets.get(dataset_name)
        if not path:
            return {"error": f"Percorso per {dataset_name} non configurato nel dizionario PATHS."}
        if not Path(path).exists():
            return {"error": f"Dataset {dataset_name} non trovato al percorso: {path}"}

        stats = {
            "dataset": dataset_name,
            "path": path,
            "total_size": self._get_folder_size(path),
            "structure": self._scan_structure(path, max_depth=2)
        }

        # Aggiungi statistiche specifiche per tipo di dataset
        try:
            if dataset_name == "GTSRB":
                stats.update(self._analyze_gtsrb(path))
            elif dataset_name == "LISA":
                stats.update(self._analyze_lisa(path))
            elif dataset_name == "VeRi":
                stats.update(self._analyze_veri(path))
        except Exception as e:
            logging.error(f"Errore durante l'analisi specifica di {dataset_name}: {e}")
            stats["specific_analysis_error"] = str(e)


        return stats

    def _analyze_gtsrb(self, path):
        """Analisi specifica per GTSRB (segnali stradali)"""
        features = {"type": "traffic_signs"}
        path_obj = Path(path)
        train_path = path_obj / "Train"
        test_path = path_obj / "Test"
        num_classes = 0
        train_samples = 0
        test_samples = 0
        avg_resolution = "N/A"
        image_format = "N/A"

        if train_path.exists() and train_path.is_dir():
            # Conta le sottocartelle in Train come classi
            num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
            try:
                ppm_files_train = list(train_path.rglob("*.ppm"))
                train_samples = len(ppm_files_train)
                if ppm_files_train:
                    image_format = "ppm"
                    avg_resolution = self._get_avg_resolution(ppm_files_train[:10]) # Usa il campione da train
            except Exception as e:
                 logging.error(f"Errore durante scansione training GTSRB: {e}")
        else:
            logging.warning(f"Cartella Training non trovata o non è una directory: {train_path}")


        if test_path.exists() and test_path.is_dir():
             try:
                 ppm_files_test = list(test_path.rglob("*.ppm"))
                 test_samples = len(ppm_files_test)
                 # Se non trovato in train, controlla formato e risoluzione qui
                 if image_format == "N/A" and ppm_files_test:
                     image_format = "ppm"
                 if avg_resolution == "N/A" and ppm_files_test:
                      avg_resolution = self._get_avg_resolution(ppm_files_test[:10])
             except Exception as e:
                 logging.error(f"Errore durante scansione test GTSRB: {e}")
        else:
            logging.warning(f"Cartella Test non trovata o non è una directory: {test_path}")


        features.update({
            "classes": num_classes,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "image_format": image_format,
            "avg_resolution": avg_resolution
        })
        return features

    def _analyze_lisa(self, path):
        """Analisi specifica per LISA (semafori)"""
        features = {"type": "traffic_lights"}
        path_obj = Path(path)
        annotation_files = 0
        video_sequences = 0
        avg_resolution = "N/A"
        image_format = "N/A"

        try:
            annotations = list(path_obj.rglob("*.csv"))
            annotation_files = len(annotations)

            # Conta le sottocartelle dirette come sequenze video
            video_sequences = len([d for d in path_obj.iterdir() if d.is_dir()])

            jpg_files = list(path_obj.rglob("*.jpg"))
            if jpg_files:
                image_format = "jpg"
                avg_resolution = self._get_avg_resolution(jpg_files[:10])

            features.update({
                "annotation_files": annotation_files,
                "video_sequences": video_sequences,
                "image_format": image_format, # Aggiunto formato immagine
                "avg_resolution": avg_resolution
            })
        except Exception as e:
            logging.error(f"Errore durante analisi LISA: {e}")
            features["error"] = str(e) # Mantiene la segnalazione di errore
        return features

    def _analyze_veri(self, path):
        """Analisi specifica per VeRi (veicoli)"""
        features = {"type": "vehicle_reid"}
        path_obj = Path(path)
        train_image_path = path_obj / "image_train"
        num_vehicles = 0 # Questo sarà 0 se non possiamo calcolarlo facilmente
        num_cameras = 0
        avg_resolution = "N/A"
        image_format = "N/A"
        train_samples = 0 # Aggiunto conteggio immagini training

        if train_image_path.exists() and train_image_path.is_dir():
            try:
                train_files = list(train_image_path.rglob("*.jpg"))
                train_samples = len(train_files)

                if train_files:
                    image_format = "jpg"
                    # Estrae ID telecamera dai nomi file
                    camera_ids = set()
                    vehicle_ids = set() # Prova a estrarre anche ID veicolo
                    for f in train_files:
                        parts = f.name.split('_')
                        if len(parts) >= 3:
                            vehicle_ids.add(parts[0]) # ID veicolo è la prima parte
                            camera_ids.add(parts[1]) # ID camera è la seconda parte (es. c001)
                        else:
                             logging.warning(f"Nome file VeRi non nel formato atteso: {f.name}")

                    num_cameras = len(camera_ids)
                    num_vehicles = len(vehicle_ids) # Numero veicoli unici basato sul nome file
                    avg_resolution = self._get_avg_resolution(train_files[:10])
                else:
                    logging.warning(f"Nessun file .jpg trovato in {train_image_path}")

            except Exception as e:
                logging.error(f"Errore durante analisi VeRi: {e}")
                features["error"] = str(e) # Mantiene la segnalazione di errore
        else:
             logging.warning(f"Cartella image_train non trovata o non è una directory: {train_image_path}")

        features.update({
            "train_samples": train_samples, # Numero immagini training
            "vehicles": num_vehicles, # Numero veicoli unici (stimato da nome file)
            "cameras": num_cameras, # Numero telecamere uniche
            "image_format": image_format, # Aggiunto formato immagine
            "avg_resolution": avg_resolution
        })
        return features


    def _get_avg_resolution(self, image_paths):
        """Calcola la risoluzione della prima immagine valida in un campione"""
        resolutions = []
        for img_path in image_paths: # Campiona fino a 10 immagini valide
            if len(resolutions) >= 1: # Basta una risoluzione valida per questo metodo
                break
            try:
                # Usa str() per sicurezza con cv2.imread
                img = cv2.imread(str(img_path))
                if img is not None:
                    # shape è (altezza, larghezza, canali)
                    resolutions.append(f"{img.shape[1]}x{img.shape[0]}")
            except Exception as e: # Cattura eccezioni specifiche
                logging.warning(f"Impossibile leggere l'immagine {img_path}: {e}")
                continue # Continua con la prossima immagine
        # Restituisce la prima risoluzione trovata o N/A
        return resolutions[0] if resolutions else "N/A"

    def _get_folder_size(self, path):
        """Calcola la dimensione totale della cartella in MB"""
        total_size_bytes = 0
        try:
            for f in Path(path).rglob('*'):
                if f.is_file(): # Assicurati che sia un file prima di accedere a stat()
                    try:
                        total_size_bytes += f.stat().st_size
                    except FileNotFoundError:
                        logging.warning(f"File sparito durante la scansione: {f}")
                        continue # Salta questo file
                    except PermissionError:
                        logging.warning(f"Permesso negato per leggere la dimensione di: {f}")
                        continue # Salta questo file
            return f"{total_size_bytes / (1024 * 1024):.2f} MB"
        except PermissionError:
             logging.error(f"Permesso negato per accedere a parti di: {path}")
             return "N/A (Permesso negato)"
        except Exception as e: # Cattura altre eccezioni generiche
            logging.error(f"Errore nel calcolare la dimensione della cartella {path}: {e}")
            return "N/A (Errore)"

    def _scan_structure(self, path, max_depth=2):
        """Scansione ricorsiva della struttura delle cartelle"""
        structure = []
        path_obj = Path(path)

        def _scan(current_path, depth):
            if depth > max_depth:
                structure.append({ # Indica che la scansione è stata troncata
                    "type": "truncated",
                    "name": "...",
                    "depth": depth
                })
                return
            try:
                # Usa try-except qui dentro per gestire errori per singolo item
                items = []
                try:
                    items = sorted(current_path.iterdir())
                except PermissionError:
                     structure.append({
                        "type": "error",
                        "message": f"Accesso negato a {current_path}",
                        "depth": depth
                     })
                     return # Non si può continuare in questa cartella

                for item in items:
                    item_info = {
                        "name": item.name,
                        "depth": depth
                    }
                    try:
                        if item.is_dir():
                            item_info["type"] = "directory"
                            structure.append(item_info)
                            _scan(item, depth + 1) # Scansiona sottocartella
                        elif item.is_file():
                            item_info["type"] = "file"
                            try:
                                item_info["size"] = f"{item.stat().st_size / 1024:.2f} KB"
                            except FileNotFoundError:
                                item_info["size"] = "N/A (Sparito)"
                            except PermissionError:
                                item_info["size"] = "N/A (Permesso negato)"
                            structure.append(item_info)
                        # else: ignora altri tipi come link simbolici, ecc.

                    except (FileNotFoundError, PermissionError) as item_e:
                         logging.warning(f"Errore durante l'analisi dell'elemento {item}: {item_e}")
                         structure.append({
                             "type": "error",
                             "name": item.name,
                             "message": str(item_e),
                             "depth": depth
                         })

            except Exception as e: # Cattura altri errori imprevisti durante la scansione
                logging.error(f"Errore imprevisto durante la scansione di {current_path}: {e}")
                structure.append({
                    "type": "error",
                    "message": f"Errore imprevisto scansionando {current_path}: {e}",
                    "depth": depth
                })


        _scan(path_obj, 0)
        return structure


if __name__ == "__main__":
    extractor = DatasetFeatures()

    # Analisi per tutti i dataset configurati
    # Usiamo i nomi definiti nell'init per robustezza
    for dataset_name in extractor.datasets.keys():
        print(f"\n=== Analisi {dataset_name} ===")
        stats = extractor.get_basic_stats(dataset_name)

        if "error" in stats:
            print(f"❌ Errore: {stats['error']}")
            continue # Passa al prossimo dataset

        # Stampa i risultati principali
        print(f"\n📊 Statistiche principali:")
        print(f"- Tipo dataset: {stats.get('type', 'N/A')}")
        print(f"- Percorso: {stats.get('path', 'N/A')}")
        print(f"- Dimensione totale: {stats.get('total_size', 'N/A')}")

        # Stampa features specifiche in base al tipo o nome
        data_type = stats.get('type')
        if data_type == "traffic_signs" or dataset_name == "GTSRB":
            print(f"- Classi: {stats.get('classes', 'N/A')}")
            print(f"- Immagini training: {stats.get('train_samples', 'N/A')}")
            print(f"- Immagini test: {stats.get('test_samples', 'N/A')}")
            print(f"- Formato immagine: {stats.get('image_format', 'N/A')}")
            print(f"- Risoluzione esempio: {stats.get('avg_resolution', 'N/A')}") # Rinominato per chiarezza

        elif data_type == "traffic_lights" or dataset_name == "LISA":
            print(f"- File annotazioni: {stats.get('annotation_files', 'N/A')}")
            print(f"- Sequenze video (cartelle): {stats.get('video_sequences', 'N/A')}")
            print(f"- Formato immagine: {stats.get('image_format', 'N/A')}")
            print(f"- Risoluzione esempio: {stats.get('avg_resolution', 'N/A')}")

        elif data_type == "vehicle_reid" or dataset_name == "VeRi":
            print(f"- Immagini training: {stats.get('train_samples', 'N/A')}") # Aggiunto
            print(f"- Veicoli unici (stimato): {stats.get('vehicles', 'N/A')}")
            print(f"- Telecamere uniche: {stats.get('cameras', 'N/A')}")
            print(f"- Formato immagine: {stats.get('image_format', 'N/A')}")
            print(f"- Risoluzione esempio: {stats.get('avg_resolution', 'N/A')}")

        if "specific_analysis_error" in stats:
             print(f"⚠️ Errore durante analisi specifica: {stats['specific_analysis_error']}")
