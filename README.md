# YOLOv8 Batch Video Object Detection

Dieses kleine Projekt ist für die **rekursive Analyse von Videodateien** mit [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) gedacht.
Es exportiert die Szenen mit gematchten Objekten entsprechend den [YOLO8 COCO Klassen](https://docs.ultralytics.com/de/datasets/detect/coco8/), normalisiert
sie im Anschluss und erzeugt ein Merge/Highlight Video aus den Einzel-CLips. Per Default wird eine Logdatei mit allen prozessierten Videos und gematchted 
Timecodes erzeugt und beim Abbruch automtaisch geprüft ob das Log bereits existiert um eine Resume zu ermöglichen. Mit "p" kann der Prozess pausiert werden.

---

## 📦 Installation

### Mit `venv` (empfohlen)
```bash
git clone https://github.com/cesfred/yolo-batch-video-objectsearch.git
cd yolo-batch-video-objectsearch
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
pip install --upgrade pip
pip install ultralytics opencv-python tqdm
```

---

## ▶️ Nutzung

### Aufruf
```bash
python object_search.py <verzeichnis> --objects 0,1
```

### Argumente

- `root` – Wurzelverzeichnis, in dem rekursiv nach Videos gesucht wird  
- `--objects` – Liste der zu suchenden Objekt-IDs (z. B. `0,1` für Person und Fahrrad)  
- `--model` – YOLOv8 Modell, Standard: `yolov8x.pt`  
- `--log` – Logdatei (Standard: `objekt_log.txt`)  
- `--export` – Exportiert erkannte Sequenzen als Clips
- `--merge` – Fasst alle exportierten Clips in einem Video zusammen  
- `--merge-ratio` – Erzwingt eine feste Zielgröße (z. B. `1920x1080`) beim Merge  
- `--merge-file` – Name der Highlight-Datei (Default `highlights.mp4`)  
- `--export-dir` – Ausgabeverzeichnis für Clips (Default `./export`)  
- `--overlay` – Blendet Dateinamen im Export ein
- `--overlay-pos` – Position der Overlay-Beschriftung (`tl`, `tr`, `bl`, `br`)  
- `--overlay-size` – Schriftgröße des Overlays (Default: `0.5`)  
- `--overlay-color` – Overlay-Farbe als `R,G,B` (Default: `255,255,255`)  
- `--pre` – Vorlaufzeit in Sekunden pro Clip (Default: `0.0`)  
- `--post` – Nachlaufzeit in Sekunden pro Clip (Default: `2.0`)  
- `--confidence` – Confidence Threshold zwischen `0.0` und `1.0` (Default: `0.8`)  
- `--cluster-gap` – Maximale Lücke (Sekunden), um Erkennungen zu einer Szene zu clustern (Default: `5.0`)  
- `--silence-decoder-warnings` – Unterdrückt FFmpeg-Decoder-Warnungen (`Missing reference picture`, …)  
- `--skip-bad-clips` – Überspringt fehlerhafte Clips beim merge
- `--no-boxes` – keine Bounding Boxes um die erkannten Objekte zeichnen
- `--quiet` – Unterdrückt alle Konsolenausgaben

### Beispiel
```bash
python object_search.py ./videos --objects 0,1 --export --merge --overlay \
  --pre 1 --post 3 --confidence 0.85 --cluster-gap 4.0
```

---

## 📝 Log-Datei

- Für **jedes Video** wird eine Zeile ins Log geschrieben:  
  - Mit Treffern:  
    ```
    /pfad/zum/video.mp4: 00:12, 01:23
    ```  
  - Ohne Treffer:  
    ```
    /pfad/zum/video.mp4: -
    ```

- Beim Resume werden **alle Videos im Log übersprungen**.  

---

## ⌨️ Steuerung

- **`p`** → Pause / Fortsetzen während der Verarbeitung  
- **`STRG+C`** → Abbruch, Logdatei bleibt erhalten  

---

## 🌍 English

This small project is intended for the **recursive analysis of video files** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
It exports scenes with matched objects according to the [YOLO8 COCO classes](https://docs.ultralytics.com/de/datasets/detect/coco8/), normalizes
them, and generates a merge/highlight video from the individual clips. By default, a log file with all processed videos and matched 
timecodes is generated, and if the process is interrupted, it automatically checks whether the log already exists to enable resumption. The process can be paused with “p”.

### Installation
```bash
git clone https://github.com/cesfred/yolo-batch-video-objectsearch.git
cd yolo-batch-video-objectsearch
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
pip install --upgrade pip
pip install ultralytics opencv-python tqdm
```

### Usage
```bash
python object_search.py <directory> --objects 0,1
```

### Arguments
- `root` – root directory to search recursively for videos  
- `--objects` – list of object IDs (e.g. `0,1` = person + bicycle)  
- `--model` – YOLOv8 model, default: `yolov8x.pt`  
- `--log` – logfile, default: `objekt_log.txt`  
- `--export` – export detected sequences as clips  
- `--merge` – merge exported clips into one highlight video  
- `--merge-ratio` – force a fixed output size (e.g. `1920x1080`) when merging  
- `--merge-file` – output filename for merged highlights (default `highlights.mp4`)  
- `--export-dir` – output directory for clips (default `./export`)  
- `--overlay` – overlay filename in export clips
- `--overlay-pos` – overlay position (`tl`, `tr`, `bl`, `br`)  
- `--overlay-size` – overlay font scale (default `0.5`)  
- `--overlay-color` – overlay color as `R,G,B` (default `255,255,255`)  
- `--pre` – seconds before detection to include (default `0.0`)  
- `--post` – seconds after detection to include (default `2.0`)  
- `--confidence` – confidence threshold between `0.0` and `1.0` (default `0.8`)  
- `--cluster-gap` – maximum gap (seconds) to group detections into one scene (default `5.0`)  
- `--silence-decoder-warnings` – suppress FFmpeg decoder warnings (`Missing reference picture`, …)  
- `--skip-bad-clips` – skip buggy clips for merge to keep the merged video intact
- `--no-boxes` – suppress drawing of bounding boxes
- `--quiet` – suppress all console output 

### Example
```bash
python object_search.py ./videos --objects 0,1 --export --merge --overlay \
  --pre 1 --post 3 --confidence 0.85 --cluster-gap 4.0
```

---

## 📜 License
Released under the **AGPL-3.0** license. See [LICENSE](LICENSE) for details.
