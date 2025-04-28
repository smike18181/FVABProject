import os
import shutil
import json
import xmltodict
from pathlib import Path
from PathDatasets import PATHS  # Importa i path definiti

# === CONFIG ===
OUTPUT_DIR = "combined_dataset"
YOLO_CLASSES = [
    "person",               # 0
    "car_front",            # 1
    "car_rear",             # 2
    "car_side",             # 3
    "traffic_light_red",    # 4 (AnnotationTag stop or stopLeft in LISA)
    "traffic_light_yellow", # 5 (AnnotationTag warning in LISA)
    "traffic_light_green",   # 6 (AnnotationTag go or goLeft in LISA)
    "speed_limit_20",          # 7 (ClassId 0 in GTSRB)
    "speed_limit_30",          # 8 (ClassId 1 in GTSRB)
    "speed_limit_50",          # 9 (ClassId 2 in GTSRB)
    "speed_limit_60",          # 10 (ClassId 3 in GTSRB)
    "speed_limit_70",          # 11 (ClassId 4 in GTSRB)
    "speed_limit_80",          # 12 (ClassId 5 in GTSRB)
    "end_speed_limit_80",      # 13 (ClassId 6 in GTSRB)
    "speed_limit_100",         # 14 (ClassId 7 in GTSRB)
    "speed_limit_120",         # 15 (ClassId 8 in GTSRB)
    "no_overtaking_general",   # 16 (ClassId 9 in GTSRB)
    "no_overtaking_trucks",    # 17 (ClassId 10 in GTSRB)
    "priority_road",         # 18 (ClassId 11 in GTSRB)
    "yield",                   # 19 (ClassId 12 in GTSRB)
    "stop",                    # 20 (ClassId 13 in GTSRB)
    "no_entry",                # 21 (ClassId 14 in GTSRB)
    "no_entry_trucks",         # 22 (ClassId 15 in GTSRB)
    "one_way_right",           # 23 (ClassId 16 in GTSRB)
    "one_way_left",            # 24 (ClassId 17 in GTSRB)
    "danger",                  # 25 (ClassId 18 in GTSRB)
    "curve_left",              # 26 (ClassId 19 in GTSRB)
    "curve_right",             # 27 (ClassId 20 in GTSRB)
    "double_curve",            # 28 (ClassId 21 in GTSRB)
    "bumpy_road",              # 29 (ClassId 22 in GTSRB)
    "slippery_road",           # 30 (ClassId 23 in GTSRB)
    "road_narrows_right",      # 31 (ClassId 24 in GTSRB)
    "road_works",              # 32 (ClassId 25 in GTSRB)
    "traffic_lights",          # 33 (ClassId 26 in GTSRB)
    "pedestrians",             # 34 (ClassId 27 in GTSRB)
    "children_crossing",       # 35 (ClassId 28 in GTSRB)
    "bicycles_crossing",       # 36 (ClassId 29 in GTSRB)
    "beware_ice_snow",         # 37 (ClassId 30 in GTSRB)
    "wild_animals_crossing",   # 38 (ClassId 31 in GTSRB)
    "end_of_all_restrictions", # 39 (ClassId 32 in GTSRB)
    "turn_right_mandatory",    # 40 (ClassId 33 in GTSRB)
    "turn_left_mandatory",     # 41 (ClassId 34 in GTSRB)
    "straight_ahead_mandatory",# 42 (ClassId 35 in GTSRB)
    "straight_or_right",       # 43 (ClassId 36 in GTSRB)
    "straight_or_left",        # 44 (ClassId 37 in GTSRB)
    "keep_right",              # 45 (ClassId 38 in GTSRB)
    "keep_left",               # 46 (ClassId 39 in GTSRB)
    "roundabout_mandatory",    # 47 (ClassId 40 in GTSRB)
    "end_of_no_overtaking_general", # 48 (ClassId 41 in GTSRB)
    "end_of_no_overtaking_trucks",  # 49 (ClassId 42 in GTSRB)
]

CLASS_MAP = {name: idx for idx, name in enumerate(YOLO_CLASSES)}

# === UTILS ===
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_and_rename(src, dst, prefix):
    filename = prefix + "_" + os.path.basename(src)
    shutil.copy(src, os.path.join(dst, filename))
    return filename

def convert_coco(coco_json, image_dir, prefix, img_out, lbl_out, include_person=False):
    with open(coco_json, "r") as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    labels = {}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = categories[ann["category_id"]]
        yolo_label = None
        if cat_name in CLASS_MAP:
            yolo_label = CLASS_MAP[cat_name]
        elif include_person and cat_name == "person":
            yolo_label = CLASS_MAP["person"]

        if yolo_label is not None:
            img_file = images[img_id]["file_name"]
            if img_id not in labels:
                labels[img_id] = []
            bbox = ann["bbox"]  # [x, y, width, height]
            w, h = images[img_id]["width"], images[img_id]["height"]
            x_center = (bbox[0] + bbox[2]/2) / w
            y_center = (bbox[1] + bbox[3]/2) / h
            labels[img_id].append(f"{yolo_label} {x_center:.6f} {y_center:.6f} {bbox[2]/w:.6f} {bbox[3]/h:.6f}")

    for img_id, anns in labels.items():
        img_info = images[img_id]
        img_file = os.path.join(image_dir, img_info["file_name"])
        new_name = copy_and_rename(img_file, img_out, prefix)
        label_file = os.path.splitext(new_name)[0] + ".txt"
        with open(os.path.join(lbl_out, label_file), "w") as f:
            f.write("\n".join(anns))

def convert_voc(voc_dir, prefix, img_out, lbl_out):
    for xml_file in Path(voc_dir).rglob("*.xml"):
        with open(xml_file, "r") as f:
            ann = xmltodict.parse(f.read())["annotation"]
        img_file = os.path.join(voc_dir, ann["filename"])
        if not os.path.exists(img_file): continue
        w, h = int(ann["size"]["width"]), int(ann["size"]["height"])
        objects = ann["object"] if isinstance(ann["object"], list) else [ann["object"]]
        yolo_lines = []
        for obj in objects:
            label = obj["name"]
            if label in CLASS_MAP:
                box = obj["bndbox"]
                x_center = (int(box["xmin"]) + int(box["xmax"])) / 2 / w
                y_center = (int(box["ymin"]) + int(box["ymax"])) / 2 / h
                bw = (int(box["xmax"]) - int(box["xmin"])) / w
                bh = (int(box["ymax"]) - int(box["ymin"])) / h
                yolo_lines.append(f"{CLASS_MAP[label]} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
        if yolo_lines:
            new_name = copy_and_rename(img_file, img_out, prefix)
            label_file = os.path.splitext(new_name)[0] + ".txt"
            with open(os.path.join(lbl_out, label_file), "w") as f:
                f.write("\n".join(yolo_lines))

def merge_yolo(src_img_dir, src_lbl_dir, prefix, img_out, lbl_out):
    for img_file in Path(src_img_dir).rglob("*.[jp][pn]g"):
        base = os.path.splitext(os.path.basename(img_file))[0]
        lbl_file = os.path.join(src_lbl_dir, base + ".txt")
        if not os.path.exists(lbl_file): continue
        new_name = copy_and_rename(img_file, img_out, prefix)
        shutil.copy(lbl_file, os.path.join(lbl_out, os.path.splitext(new_name)[0] + ".txt"))

# === MAIN ===
def main():
    # Percorsi dei tuoi dataset definiti in Pathdatasets.py
    gtsrb_path = PATHS.get("GTSRB_PATH")
    lisa_path = PATHS.get("LISA_PATH")
    veri_path = PATHS.get("VeRi-776_PATH")

    # Verifica se i path sono definiti
    if not all([gtsrb_path, lisa_path, veri_path]):
        print("Errore: Alcuni path dei dataset non sono definiti correttamente nel file Pathdatasets.py")
        return

    # Definisci i percorsi specifici per i dati che ti interessano
    gtsrb_train_coco_json = Path(gtsrb_path) / "Meta"  # Assumendo che il COCO sia in Meta
    gtsrb_train_images_dir = Path(gtsrb_path) / "Train"
    gtsrb_test_coco_json = Path(gtsrb_path) / "Meta"  # Assumendo che il COCO sia in Meta
    gtsrb_test_images_dir = Path(gtsrb_path) / "Test"

    lisa_voc_dir = Path(lisa_path) / "Annotations"  # Assumi che le annotazioni VOC siano qui
    lisa_images_dir = Path(lisa_path) / "JPEGImages"  # Assumi che le immagini siano qui

    veri_yolo_images_dir = Path(veri_path) / "image_train"  # Assumi che le immagini YOLO siano qui
    veri_yolo_labels_dir = Path(veri_path) / "label_train"  # Assumi che le label YOLO siano qui

    # Configura il dataset YOLO delle persone (ipotetico, adatta il percorso se necessario)
    person_yolo = {
        "images": "yolo_person/images",  # <---------------------- ADATTA QUESTO PERCORSO SE NECESSARIO
        "labels": "yolo_person/labels"   # <---------------------- ADATTA QUESTO PERCORSO SE NECESSARIO
    }

    img_out = os.path.join(OUTPUT_DIR, "images")
    lbl_out = os.path.join(OUTPUT_DIR, "labels")
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    print("▶ Convert GTSRB (Train) COCO...")
    # Dovrai identificare il vero file .json COCO all'interno della cartella "Meta"
    gtsrb_train_coco_file = next(Path(gtsrb_train_coco_json).glob("*.json"), None)
    if gtsrb_train_coco_file and gtsrb_train_images_dir.exists():
        convert_coco(str(gtsrb_train_coco_file), str(gtsrb_train_images_dir), "gtsrb_train", img_out, lbl_out, include_person=True)
    else:
        print(f"   ⚠️ File COCO non trovato in Meta o directory immagini non trovata per GTSRB Train: {gtsrb_train_coco_json}, {gtsrb_train_images_dir}")

    print("▶ Convert GTSRB (Test) COCO...")
    # Dovrai identificare il vero file .json COCO all'interno della cartella "Meta"
    gtsrb_test_coco_file = next(Path(gtsrb_test_coco_json).glob("*.json"), None)
    if gtsrb_test_coco_file and gtsrb_test_images_dir.exists():
        convert_coco(str(gtsrb_test_coco_file), str(gtsrb_test_images_dir), "gtsrb_test", img_out, lbl_out, include_person=True)
    else:
        print(f"   ⚠️ File COCO non trovato in Meta o directory immagini non trovata per GTSRB Test: {gtsrb_test_coco_json}, {gtsrb_test_images_dir}")

    print("▶ Convert LISA (VOC)...")
    if lisa_voc_dir.exists() and lisa_images_dir.exists():
        convert_voc(str(lisa_voc_dir), "lisa", img_out, lbl_out)
    else:
        print(f"   ⚠️ Directory VOC o immagini non trovate per LISA: {lisa_voc_dir}, {lisa_images_dir}")

    print("▶ Merge VeRi-776 (YOLO)...")
    if veri_yolo_images_dir.exists() and veri_yolo_labels_dir.exists():
        merge_yolo(str(veri_yolo_images_dir), str(veri_yolo_labels_dir), "veri", img_out, lbl_out)
    else:
        print(f"   ⚠️ Directory immagini o label YOLO non trovate per VeRi-776: {veri_yolo_images_dir}, {veri_yolo_labels_dir}")

    print("▶ Add YOLO person...")
    if Path(person_yolo["images"]).exists() and Path(person_yolo["labels"]).exists():
        merge_yolo(person_yolo["images"], person_yolo["labels"], "person", img_out, lbl_out)
    else:
        print(f"   ⚠️ Directory immagini o label YOLO non trovate per 'person': {person_yolo['images']}, {person_yolo['labels']}")

    print("✅ Dataset combinato in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()