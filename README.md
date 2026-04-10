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
- `src/capture.py`: stream low-latency adb+ffmpeg
- `src/detector.py`: wrapper Ultralytics con model switch
- `src/pipeline.py`: loop capture -> detect -> overlay
- `src/gui.py`: mini pannello Tkinter
- `src/main.py`: entrypoint CLI
- `run.py`: launcher progetto

## Prerequisiti

1. Android con debug USB (o adb tcpip) attivo
2. Tool locali in `tools/`:
  - `tools/platform-tools/adb.exe`
  - `tools/ffmpeg/bin/ffmpeg.exe`
  - `tools/scrcpy/scrcpy-win64-v3.3.1/scrcpy.exe`
3. Modelli `.pt` in `models/`

### Setup tool esterni (ADB/FFmpeg/Scrcpy)

Per scaricare automaticamente i binari dentro il progetto:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_tools.ps1
```

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

Fallback testing con webcam (senza Android/ADB):

```bat
python run.py --config config/settings.yaml --source webcam --webcam-index 0 --gui
```

## Profili VS Code Debug/Launch

Sono disponibili profili multipli in `.vscode/launch.json`:

- `Debug Webcam (index 0)`
- `Debug Webcam (custom index)`
- `Debug ADB USB (wait + run)`
- `Debug ADB TCP (connect + run all-in-one)`

Task ADB collegati in `.vscode/tasks.json`:

- `adb: wait usb`
- `adb: connect tcp all-in-one`
- `adb: disconnect tcp`

Flusso all-in-one per ADB TCP:

1. Avvia il profilo `Debug ADB TCP (connect + run all-in-one)`.
2. Inserisci host e porta quando richiesto.
3. VS Code esegue prima `adb connect` + `wait-for-device`, poi lancia `run.py` in debug.

## Utilizzo runtime

- Tasto `q`: arresta pipeline
- Tasti `1..9`: switch rapido modello secondo ordine registry YAML
- GUI: selezione modello con applicazione automatica (senza pulsante)
- Source webcam: usa `--source webcam` per test locali senza telefono
- GUI: puoi cambiare in tempo reale la frequenza YOLO (Hz). `0` = inferenza a ogni frame. Le modifiche vengono applicate automaticamente
- GUI: puoi cambiare in tempo reale la soglia di riconoscimento `conf` (0..1). Le modifiche vengono applicate automaticamente
- GUI: puoi cambiare in tempo reale la soglia NMS `iou` (0..1). Valori piu bassi riducono box sovrapposte
- GUI: puoi cambiare in tempo reale `max_det` (numero massimo di box per frame)
- GUI: puoi cambiare in tempo reale `imgsz` (risoluzione di inferenza YOLO, consigliato 640 se il training era a 640)

## Note su overlay e persistenza BB

- Le bounding box ora restano visibili tra due inferenze consecutive (niente lampeggio tra un detect e il successivo)
- Le bounding box persistenti sono renderizzate con il plot nativo di Ultralytics, quindi mantengono stile e coordinate identiche all'output YOLO
- La frequenza modello si regola live dalla GUI, senza riavviare la pipeline

Default consigliati (profilo preciso e box pulite):

- `runtime.detection_conf_threshold: 0.6`
- `runtime.detection_iou_threshold: 0.30`
- `runtime.detection_max_det: 12`
- `runtime.yolo_interval_sec: 0.0` (inferenza a ogni frame)
- `runtime.detection_imgsz: 640`

Parametri che ha senso tenere configurabili live in GUI:

- `model`: switch del checkpoint
- `detection_conf_threshold`: precisione vs recall
- `detection_iou_threshold`: aggressivita NMS per ridurre overlap
- `detection_max_det`: limite box/frame
- `detection_imgsz`: risoluzione inferenza (impatto su precisione/latency)
- `yolo_interval_sec` (tramite Hz): frequenza inferenza

## Note latenza

- Non viene catturata la finestra scrcpy: il flusso entra da adb H264 e viene decodificato direttamente
- Per ridurre latenza:
  - aumenta `capture.bitrate_mbps`
  - riduci risoluzione (`use_original_resolution: false` + `resize_width/height`)
  - aumenta `runtime.yolo_interval_sec` per ridurre il carico inferenza (es. `0.5` = YOLO ogni 500 ms)
  - usa GPU CUDA quando disponibile

## Autore

Mirto Musci, Copyright 2026
