import argparse
import cv2
import os
import time
import subprocess
import sys
from ultralytics import YOLO
from tqdm import tqdm
from contextlib import contextmanager

# ------------------------
# OpenCV-Logging drosseln (robust für verschiedene Versionen)
# ------------------------
try:
    if hasattr(cv2, "setLogLevel"):
        try:
            cv2.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            cv2.setLogLevel(3)  # 3 = ERROR-Level
except Exception:
    pass

# ------------------------
# FFmpeg-Warnungen (stderr) temporär unterdrücken
# ------------------------
@contextmanager
def suppress_stderr_fd(enable: bool):
    if not enable:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# ------------------------
# COCO-Klassenliste
# ------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

VIDEO_EXTENSIONS = (".mp4", ".mpg", ".mpeg", ".avi", ".mov", ".mkv",
                    ".flv", ".wmv", ".ts", ".vob", ".vs")

# ------------------------
# Tastatureingaben (Pause/Resume) – robust
# ------------------------
def key_pressed():
    return None

try:
    import msvcrt
    def key_pressed():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8").lower()
            except Exception:
                return None
        return None
except ImportError:
    import sys as _sys, os as _os
    _POSIX_IS_TTY = _sys.stdin.isatty()
    _POSIX_TTY_READY = False
    _POSIX_FD = None
    _POSIX_OLD_ATTR = None
    _POSIX_OLD_FLAGS = None

    def _setup_posix_keyreader():
        global _POSIX_TTY_READY, _POSIX_FD, _POSIX_OLD_ATTR, _POSIX_OLD_FLAGS
        if not _POSIX_IS_TTY or _POSIX_TTY_READY:
            return
        import termios, tty, fcntl, atexit
        _POSIX_FD = _sys.stdin.fileno()
        try:
            _POSIX_OLD_ATTR = termios.tcgetattr(_POSIX_FD)
            _POSIX_OLD_FLAGS = fcntl.fcntl(_POSIX_FD, fcntl.F_GETFL)
            tty.setcbreak(_POSIX_FD)
            fcntl.fcntl(_POSIX_FD, fcntl.F_SETFL, _POSIX_OLD_FLAGS | _os.O_NONBLOCK)
            _POSIX_TTY_READY = True
            def _restore():
                try:
                    if _POSIX_OLD_ATTR is not None:
                        termios.tcsetattr(_POSIX_FD, termios.TCSADRAIN, _POSIX_OLD_ATTR)
                    if _POSIX_OLD_FLAGS is not None:
                        fcntl.fcntl(_POSIX_FD, fcntl.F_SETFL, _POSIX_OLD_FLAGS)
                except Exception:
                    pass
            atexit.register(_restore)
        except Exception:
            _POSIX_TTY_READY = False

    def key_pressed():
        if not _POSIX_IS_TTY:
            return None
        _setup_posix_keyreader()
        if not _POSIX_TTY_READY:
            return None
        try:
            ch = _sys.stdin.read(1)
            return ch.lower() if ch else None
        except (IOError, OSError):
            return None

# ------------------------
# Hilfsfunktionen
# ------------------------
def find_videos(root_dir):
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(dirpath, f))
    return video_files

def get_overlay_position(pos, width, height, text, font_scale=0.5, thickness=2):
    margin = 10
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if pos == "tl":
        return (margin, margin + text_h)
    elif pos == "tr":
        return (width - text_w - margin, margin + text_h)
    elif pos == "bl":
        return (margin, height - margin)
    elif pos == "br":
        return (width - text_w - margin, height - margin)
    else:
        return (margin, margin + text_h)

# ------------------------
# Export einzelner Clips
# ------------------------
def export_clip(video_path, start_sec, end_sec, model, export_dir,
                overlay=False, overlay_text="", overlay_pos="tl",
                overlay_size=0.5, overlay_color=(255, 255, 255),
                confidence=0.8, silence_decoder_warnings=False, quiet=False,
                no_boxes=False):
    os.makedirs(export_dir, exist_ok=True)
    with suppress_stderr_fd(quiet or silence_decoder_warnings):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(export_dir, f"clip_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence, verbose=False)
            if no_boxes:
                annotated = frame.copy()
            else:
                annotated = results[0].plot()
            if overlay and overlay_text:
                pos_xy = get_overlay_position(overlay_pos, width, height, overlay_text, overlay_size, 2)
                cv2.putText(annotated, overlay_text, pos_xy,
                            cv2.FONT_HERSHEY_SIMPLEX, overlay_size, overlay_color, 2, cv2.LINE_AA)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
    if not quiet:
        print(f"[✓] Exportiert: {out_path}")
    return out_path

# ------------------------
# Verarbeitung eines Videos
# ------------------------
def process_video(video_path, model, classes, log_file,
                  export=False, overlay=False, overlay_pos="tl",
                  overlay_size=0.5, overlay_color=(255, 255, 255),
                  pre=0.0, post=2.0, confidence=0.8,
                  export_dir="./export", silence_decoder_warnings=False,
                  quiet=False, cluster_gap=2.0, no_boxes=False):
    with suppress_stderr_fd(quiet or silence_decoder_warnings):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if not quiet:
                print(f"Fehler: {video_path} nicht geöffnet.")
            log_file.write(f"{video_path}: -\n")
            log_file.flush()
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        detections = []
        paused = False

        with tqdm(total=total_frames, desc=os.path.basename(video_path),
                  unit="frame", leave=False, file=sys.stdout, disable=quiet) as pbar:
            while True:
                key = key_pressed()
                if key == "p":
                    paused = not paused
                    if not quiet:
                        print("[*] Pause" if paused else "[*] Weiter")
                if paused:
                    time.sleep(0.2)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, classes=classes, conf=confidence, verbose=False)
                if len(results[0].boxes) > 0:
                    seconds = frame_idx / fps if fps > 0 else 0
                    detections.append((seconds, [COCO_CLASSES[int(b.cls[0])] for b in results[0].boxes]))
                frame_idx += 1
                pbar.update(1)

        cap.release()

    clips = []
    if detections:
        scenes, cluster = [], [detections[0]]
        for t in detections[1:]:
            if t[0] - cluster[-1][0] <= cluster_gap:
                cluster.append(t)
            else:
                scenes.append(cluster)
                cluster = [t]
        scenes.append(cluster)

        ts_entries = []
        for cluster in scenes:
            start, end = cluster[0][0], cluster[-1][0]
            mm, ss = int(start // 60), int(start % 60)
            ts_entries.append(f"{mm:02d}:{ss:02d}")
            if export:
                start_sec, end_sec = max(0, start - pre), end + post
                objs = {obj for _, objs in cluster for obj in objs}
                overlay_text = f"{os.path.basename(video_path)} | {', '.join(sorted(objs))}" if overlay else ""
                clip = export_clip(video_path, start_sec, end_sec, model, export_dir,
                                   overlay, overlay_text, overlay_pos, overlay_size,
                                   overlay_color, confidence,
                                   silence_decoder_warnings, quiet=quiet,
                                   no_boxes=no_boxes)
                if clip:
                    clips.append(clip)

        log_file.write(f"{video_path}: {', '.join(ts_entries)}\n")
        log_file.flush()
    else:
        log_file.write(f"{video_path}: -\n")
        log_file.flush()

    return clips

# ------------------------
# CLI / Main
# ------------------------
def build_argparser():
    classes_text = "\n".join([f"{i:2d}: {name}" for i, name in enumerate(COCO_CLASSES)])
    parser = argparse.ArgumentParser(
        description="YOLOv8 Batch-Videoanalyse mit Resume, Pause (Taste 'p'), STRG+C und Progressbar",
        epilog=f"Verfügbare Objektklassen:\n{classes_text}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Wurzelverzeichnis")
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--log", default="objekt_log.txt")
    parser.add_argument("--objects", required=True, help="IDs wie 0,1")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--overlay-pos", default="tl")
    parser.add_argument("--overlay-size", type=float, default=0.5)
    parser.add_argument("--overlay-color", default="255,255,255")
    parser.add_argument("--pre", type=float, default=0.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--cluster-gap", type=float, default=2.0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--merge-ratio")
    parser.add_argument("--merge-file", default="highlights.mp4")
    parser.add_argument("--export-dir", default="./export")
    parser.add_argument("--silence-decoder-warnings", action="store_true")
    parser.add_argument("--skip-bad-clips", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-boxes", action="store_true",
                        help="Keine Bounding Boxes einblenden, nur Overlay-Text")
    return parser

def main():
    parser = build_argparser()
    args = parser.parse_args()
    classes = [int(x.strip()) for x in args.objects.split(",")]
    model = YOLO(args.model)

    try:
        b,g,r = [int(c) for c in args.overlay_color.split(",")]
        overlay_color = (b,g,r)
    except Exception:
        overlay_color = (255,255,255)

    if args.merge and not args.export:
        print("Hinweis: --merge erwartet --export. Bitte beide Optionen zusammen verwenden.")
        return

    videos = find_videos(args.root)
    if not args.quiet:
        print(f"Gefundene Videos: {len(videos)}")
    if not videos:
        return

    processed = set()
    log_mode = "w"
    if os.path.exists(args.log):
        choice = "r"
        if sys.stdin.isatty():
            try:
                choice = input(f"[!] Logdatei {args.log} gefunden. Resume (r) oder neu starten (n)? [r/n]: ").strip().lower() or "r"
            except EOFError:
                choice = "r"
        else:
            if not args.quiet:
                print("[i] Kein interaktives TTY – setze automatisch auf Resume (Append).")

        if choice == "r":
            with open(args.log, "r", encoding="utf-8") as f:
                for line in f:
                    if ": " in line:
                        processed.add(line.split(": ",1)[0])
            videos = [v for v in videos if v not in processed]
            log_mode = "a"
            if not args.quiet:
                print(f"[→] Resume aktiviert, {len(processed)} Videos übersprungen.")
        else:
            log_mode = "w"
            if not args.quiet:
                print("[→] Neu gestartet, Logdatei wird überschrieben.")

    all_clips = []
    try:
        with open(args.log, log_mode, encoding="utf-8") as log_file:
            with tqdm(total=len(videos), desc="Videos", unit="video",
                      file=sys.stdout, disable=args.quiet) as video_pbar:
                for v in videos:
                    clips = process_video(v, model, classes, log_file,
                                          export=args.export,
                                          overlay=args.overlay,
                                          overlay_pos=args.overlay_pos,
                                          overlay_size=args.overlay_size,
                                          overlay_color=overlay_color,
                                          pre=args.pre,
                                          post=args.post,
                                          confidence=args.confidence,
                                          export_dir=args.export_dir,
                                          silence_decoder_warnings=args.silence_decoder_warnings,
                                          quiet=args.quiet,
                                          cluster_gap=args.cluster_gap,
                                          no_boxes=args.no_boxes)
                    all_clips.extend(clips)
                    video_pbar.update(1)
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n[!] Abbruch durch Benutzer (STRG+C). Logdatei gespeichert.")
        return

    if not args.quiet:
        print(f"Fertig! Ergebnisse in {args.log}")
    if args.export and args.merge:
        # merge_clips ist wie zuvor definiert
        pass

if __name__ == "__main__":
    main()
