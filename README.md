# Drone Detector Fire

Pipeline a bassa latenza per:
- acquisire stream Android senza duplicare la finestra desktop
- applicare inferenza YOLO Ultralytics su ogni frame
- mostrare overlay con bounding box e metriche live
- cambiare modello in tempo reale con hotkey o mini GUI

## Architettura scelta
- Core: Python puro
- Capture: ADB `screenrecord --output-format=h264` + FFmpeg decode rawvideo
- Overlay: finestra OpenCV dedicata
- Switch modello: hotkey `1..9` e pannello Tkinter
- Config: YAML
- Registry modelli: cartella locale `models/`
- Hardware: auto (`CUDA` se disponibile, altrimenti `CPU`)

## Struttura

- `config/settings.yaml`: configurazione runtime e modelli
- `src/drone_detector_fire/capture.py`: stream low-latency adb+ffmpeg
- `src/drone_detector_fire/detector.py`: wrapper Ultralytics con model switch
- `src/drone_detector_fire/pipeline.py`: loop capture -> detect -> overlay
- `src/drone_detector_fire/gui.py`: mini pannello Tkinter
- `src/drone_detector_fire/main.py`: entrypoint CLI
- `run.py`: launcher progetto

## Prerequisiti

1. Android con debug USB (o adb tcpip) attivo
2. `adb.exe` e `ffmpeg.exe` disponibili nel path configurato
3. Modelli `.pt` in `models/`

## Avvio rapido (Windows)

```bat
scripts\run.bat
```

Oppure manuale:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py --config config/settings.yaml --gui
```

## Utilizzo runtime

- Tasto `q`: arresta pipeline
- Tasti `1..9`: switch rapido modello secondo ordine registry YAML
- GUI: seleziona modello e premi `Switch Modello`

## Note latenza

- Non viene catturata la finestra scrcpy: il flusso entra da adb H264 e viene decodificato direttamente
- Per ridurre latenza:
  - aumenta `capture.bitrate_mbps`
  - riduci risoluzione (`use_original_resolution: false` + `resize_width/height`)
  - aumenta `runtime.yolo_interval_sec` per ridurre il carico inferenza (es. `0.5` = YOLO ogni 500 ms)
  - usa GPU CUDA quando disponibile

## Autore

Mirto Musci
