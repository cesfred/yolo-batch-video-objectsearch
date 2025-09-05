# YOLOv8 Batch Video Object Detection

Dieses Projekt ermöglicht die **rekursive Analyse von Videodateien** mit [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
Es unterstützt **Resume**, **Pause**, **Export von Clips mit Bounding Boxes**, **Highlight-Zusammenfassung** und flexible Parameter.

---

## 📦 Installation

### Mit `venv` (empfohlen)
```bash
# Neues virtuelles Environment erstellen
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell

# Abhängigkeiten installieren
pip install --upgrade pip
pip install ultralytics opencv-python tqdm
```

---

## ▶️ Nutzung

### Basis-Aufruf
```bash
python batch_multi_objects.py <verzeichnis> --objects 0,1
```

### Wichtige Argumente

- `root` – Wurzelverzeichnis, in dem rekursiv nach Videos gesucht wird  
- `--objects` – Liste der Objekt-IDs (z. B. `0,1` für Person und Fahrrad)  
- `--model` – YOLOv8 Modell, Standard: `yolov8l.pt`  
- `--log` – Logdatei (Standard: `objekt_log.txt`)  
- `--export` – Exportiert erkannte Sequenzen als Clips mit Bounding Boxes  
- `--merge` – Fasst alle exportierten Clips in einem Highlight-Video zusammen  
- `--overlay` – Blendet Dateinamen und erkannte Objekte im Export ein  
- `--overlay-pos` – Position der Overlay-Beschriftung (`tl`, `tr`, `bl`, `br`)  
- `--overlay-size` – Schriftgröße des Overlays (Standard: `0.5`)  
- `--overlay-color` – Overlay-Farbe als `R,G,B` (Standard: `255,255,255`)  
- `--pre` – Vorlaufzeit in Sekunden pro Clip (Default: `0.0`)  
- `--post` – Nachlaufzeit in Sekunden pro Clip (Default: `2.0`)  
- `--confidence` – Confidence Threshold zwischen `0.0` und `1.0` (Default: `0.8`)  
- `--quiet` – Unterdrückt Konsolenausgaben außer Progressbars  

### Beispiel
```bash
python batch_multi_objects.py ./videos --objects 0,1 --export --merge --overlay --pre 1 --post 3 --confidence 0.85
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
- Die Logdatei wird nur bei **Neu-Start** überschrieben, niemals beim Resume.

---

## ⌨️ Steuerung

- **`p`** → Pause / Fortsetzen während der Verarbeitung  
- **`STRG+C`** → Abbruch, Logdatei bleibt erhalten  

---

## 🌍 English

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell

pip install --upgrade pip
pip install ultralytics opencv-python tqdm
```

### Usage
```bash
python batch_multi_objects.py <directory> --objects 0,1
```

### Arguments
- `root` – root directory to search recursively for videos  
- `--objects` – list of object IDs (e.g. `0,1` = person + bicycle)  
- `--model` – YOLOv8 model, default: `yolov8l.pt`  
- `--log` – logfile, default: `objekt_log.txt`  
- `--export` – export detected sequences as clips with bounding boxes  
- `--merge` – merge exported clips into one highlight video  
- `--overlay` – overlay filename + detected objects in export  
- `--overlay-pos` – overlay position (`tl`, `tr`, `bl`, `br`)  
- `--overlay-size` – overlay font scale (default `0.5`)  
- `--overlay-color` – overlay color as `R,G,B` (default `255,255,255`)  
- `--pre` – seconds before detection to include (default `0.0`)  
- `--post` – seconds after detection to include (default `2.0`)  
- `--confidence` – confidence threshold between `0.0` and `1.0` (default `0.8`)  
- `--quiet` – suppress console output except progress bars  

### Example
```bash
python batch_multi_objects.py ./videos --objects 0,1 --export --merge --overlay --pre 1 --post 3 --confidence 0.85
```

---

## 📜 License
Released under the **AGPL-3.0** license. See [LICENSE](LICENSE) for details.
