# airbud

A simple webcam-based reaction game built with pygame and OpenCV.  The
repository contains the following submodules:

* `engine/` – core game logic (timer, player, target, collision detection)
* `vision/` – helper for running a YOLOv8 person detector (used optionally
  by the game)

## Setup

Create a Python virtual environment and install the packages in
`requirements.txt` (the one bundled with the project already includes
`ultralytics`).

```bash
python -m pip install -r requirements.txt
```

## Running

The entry point is `main.py` and it supports three modes:

* `game` – run the reaction game only (default)
* `detector` – run the YOLO person detector as a standalone window
* `combo` – run the game with live people count shown on-screen

The game window now displays the live webcam feed by default.  All human
(YOLO) detection functionality has been removed from this version, so no
annotations or yellow squares are drawn.

The `--mode` argument formerly allowed `detector`/`combo` modes but is now a
placeholder; only the game is available.

```bash
python main.py
```

Use `Ctrl+C` or close the window to quit.