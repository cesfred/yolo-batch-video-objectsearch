# yolo-batch-video-objectsearch

[Deutsch](#deutsch) | [English](#english)

Rekursive KI-Objektsuche in Videodateien nach COCO-Klassen. Gefundene Objekte werden
mit Bounding-Boxes markiert und als Video-Schnipsel exportiert und optional am Ende
der Prozessierung gemerged. Zusätzlich kann eine Log-Datei mit Datei-Pfad und 
Zeit-Indexen der gematchten Klassen exportiert werden.

Recursive AI object search in video files according to COCO classes. Found objects are
marked with bounding boxes and exported as video snippets and optionally merged at the end
of processing. In addition, a log file with file paths and 
time indexes of the matched classes can be exported.

- See: [Ultralytics YOLO Dokumentatio](https://docs.ultralytics.com/de/datasets/detect/coco/)
---

## Deutsch

### 🛠️ Installation

1. Repository klonen oder Dateien herunterladen  
   ```bash
   git clone https://github.com/cesfred/yolo-batch-video-objectsearch.git
   cd yolo-batch-video-objectsearch
   ```

2. Virtuelle Umgebung erstellen und aktivieren  
   ```bash
   python3 -m venv ./venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. Abhängigkeiten installieren  
   ```bash
   pip install --upgrade pip
   pip install ultralytics opencv-python tqdm
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   ```

   > Für Video-Export wird **ffmpeg** benötigt:  
   > - Linux: `sudo apt install ffmpeg`  
   > - macOS: `brew install ffmpeg`  
   > - Windows: [ffmpeg.org](https://ffmpeg.org/download.html)

---

### 🚀 Nutzung

Grundbefehl:
```bash
python3 object_search.py <video-ordner> --objects <klassen-ids> [optionen]
```

Beispiel (Personen + Fahrräder erkennen, Szenen exportieren, Overlay unten rechts):
```bash
python3 object_search.py ./videos --objects 0,1 --export --overlay --overlay-pos br
```

---

### ⚙️ Funktionen

- **Mehrere Videos rekursiv durchsuchen** (`.mp4`, `.avi`, `.mkv`, `.vs`, …)  
- **Objekterkennung mit YOLOv8** (Standardmodell: `yolov8x.pt`)  
- **Logdatei** mit Zeitstempeln aller erkannten Szenen  
- **Export**: erkannte Szenen als Clips speichern (mit 3s Vor- und Nachlauf)  
- **Overlay**: Dateiname + erkannte Objekte ins Video einblenden  
- **Merge**: alle Clips automatisch zu `highlights.mp4` zusammenfügen  
- **Pause**: mit Taste `p` pausieren/fortsetzen  
- **Progressbars**: Fortschritt pro Video und Gesamtbatch  
- **Quiet Mode**: mit `--quiet` werden alle Konsolenmeldungen außer Progressbars unterdrückt  

---

### 📋 Optionen

```bash
--model yolov8x.pt   # Modell (Standard: yolov8x.pt)
--objects 0,1        # COCO-IDs (z.B. 0=Person, 1=Fahrrad)
--log objekt_log.txt # Logdatei
--export             # Szenen exportieren
--overlay            # Overlay aktivieren
--overlay-pos tl/tr/bl/br  # Overlay-Position (Standard: tl)
--overlay-size 0.5   # Schriftgröße (Standard: 0.5)
--overlay-color 255,255,255  # Schriftfarbe (B,G,R)
--merge              # Clips zu highlights.mp4 zusammenfügen
--quiet              # Unterdrückt Konsolenausgaben außer Progressbars
```

---

### 📑 COCO-Klassen (IDs)

| ID | Klasse        | ID | Klasse       | ID | Klasse       | ID | Klasse         |
|----|---------------|----|--------------|----|--------------|----|----------------|
| 0  | person        | 20 | elephant     | 40 | bottle       | 60 | microwave      |
| 1  | bicycle       | 21 | bear         | 41 | wine glass   | 61 | oven           |
| 2  | car           | 22 | zebra        | 42 | cup          | 62 | toaster        |
| 3  | motorcycle    | 23 | giraffe      | 43 | fork         | 63 | sink           |
| 4  | airplane      | 24 | backpack     | 44 | knife        | 64 | refrigerator   |
| 5  | bus           | 25 | umbrella     | 45 | spoon        | 65 | book           |
| 6  | train         | 26 | handbag      | 46 | bowl         | 66 | clock          |
| 7  | truck         | 27 | tie          | 47 | banana       | 67 | vase           |
| 8  | boat          | 28 | suitcase     | 48 | apple        | 68 | scissors       |
| 9  | traffic light | 29 | frisbee      | 49 | sandwich     | 69 | teddy bear     |
| 10 | fire hydrant  | 30 | skis         | 50 | orange       | 70 | hair drier     |
| 11 | stop sign     | 31 | snowboard    | 51 | broccoli     | 71 | toothbrush     |
| 12 | parking meter | 32 | sports ball  | 52 | carrot       |    |                |
| 13 | bench         | 33 | kite         | 53 | hot dog      |    |                |
| 14 | bird          | 34 | baseball bat | 54 | pizza        |    |                |
| 15 | cat           | 35 | baseball glove| 55 | donut       |    |                |
| 16 | dog           | 36 | skateboard   | 56 | cake         |    |                |
| 17 | horse         | 37 | surfboard    | 57 | chair        |    |                |
| 18 | sheep         | 38 | tennis racket| 58 | couch        |    |                |
| 19 | cow           | 39 | bottle       | 59 | potted plant |    |                |

---

## English

### 🛠️ Installation

1. Clone repository or download files  
   ```bash
   git clone https://github.com/cesfred/yolo-batch-video-objectsearch.git
   cd yolo-batch-video-objectsearch
   ```

2. Create and activate virtual environment  
   ```bash
   python3 -m venv ./venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies  
   ```bash
   pip install --upgrade pip
   pip install ultralytics opencv-python tqdm
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   ```

   > For video export, **ffmpeg** is required:  
   > - Linux: `sudo apt install ffmpeg`  
   > - macOS: `brew install ffmpeg`  
   > - Windows: [ffmpeg.org](https://ffmpeg.org/download.html)

---

### 🚀 Usage

Basic command:
```bash
python object_search.py <video-folder> --objects <class-ids> [options]
```

Example (detect persons + bicycles, export scenes with overlay bottom-right):
```bash
python object_search.py ./videos --objects 0,1 --export --overlay --overlay-pos br
```

---

### ⚙️ Features

- **Recursive video scanning** (`.mp4`, `.avi`, `.mkv`, `.vs`, …)  
- **Object detection with YOLOv8** (default: `yolov8x.pt`)  
- **Log file** with timestamps of detected scenes  
- **Export**: save detected scenes as clips (+3s before and after)  
- **Overlay**: add filename + detected objects to video  
- **Merge**: combine all clips into `highlights.mp4`  
- **Pause**: press `p` to pause/resume  
- **Progressbars**: per video and overall batch progress  
- **Quiet Mode**: with `--quiet`, suppress all console output except progressbars  

---

### 📋 Options

```bash
--model yolov8x.pt   # Model (default: yolov8x.pt)
--objects 0,1        # COCO IDs (e.g., 0=Person, 1=Bicycle)
--log objekt_log.txt # Log file
--export             # Export detected scenes
--overlay            # Enable overlay
--overlay-pos tl/tr/bl/br  # Overlay position (default: tl)
--overlay-size 0.5   # Font size (default: 0.5)
--overlay-color 255,255,255  # Font color (B,G,R)
--merge              # Merge clips into highlights.mp4
--quiet              # Suppress console output except progressbars
```

---

### 📑 COCO Classes (IDs)

| ID | Class         | ID | Class        | ID | Class        | ID | Class         |
|----|---------------|----|--------------|----|--------------|----|---------------|
| 0  | person        | 20 | elephant     | 40 | bottle       | 60 | microwave     |
| 1  | bicycle       | 21 | bear         | 41 | wine glass   | 61 | oven          |
| 2  | car           | 22 | zebra        | 42 | cup          | 62 | toaster       |
| 3  | motorcycle    | 23 | giraffe      | 43 | fork         | 63 | sink          |
| 4  | airplane      | 24 | backpack     | 44 | knife        | 64 | refrigerator  |
| 5  | bus           | 25 | umbrella     | 45 | spoon        | 65 | book          |
| 6  | train         | 26 | handbag      | 46 | bowl         | 66 | clock         |
| 7  | truck         | 27 | tie          | 47 | banana       | 67 | vase          |
| 8  | boat          | 28 | suitcase     | 48 | apple        | 68 | scissors      |
| 9  | traffic light | 29 | frisbee      | 49 | sandwich     | 69 | teddy bear    |
| 10 | fire hydrant  | 30 | skis         | 50 | orange       | 70 | hair drier    |
| 11 | stop sign     | 31 | snowboard    | 51 | broccoli     | 71 | toothbrush    |
| 12 | parking meter | 32 | sports ball  | 52 | carrot       |    |               |
| 13 | bench         | 33 | kite         | 53 | hot dog      |    |               |
| 14 | bird          | 34 | baseball bat | 54 | pizza        |    |               |
| 15 | cat           | 35 | baseball glove| 55 | donut       |    |               |
| 16 | dog           | 36 | skateboard   | 56 | cake         |    |               |
| 17 | horse         | 37 | surfboard    | 57 | chair        |    |               |
| 18 | sheep         | 38 | tennis racket| 58 | couch        |    |               |
| 19 | cow           | 39 | bottle       | 59 | potted plant |    |               |

