# CamWatch
Face and Car detector from video streams using Computer Vision 
with Face Recognition and Car Number Plate detection for intgration
with MQTT and Home Assistant

![GitHub Logo](docs/camwatch-target_arch.png)

> Still under-development and some features (e.g. Number Plate 
> detection) are not available.
> 
> Further updates and documentation improvement are coming in the next few days.
 
## Quick Start
- `git clone https://github.com/joelee/camwatch.git`
- `cd camwatch && cp config/camwatch-quick_start.yaml config/camwatch.yaml`
- Edit and customise `config/camwatch.yaml` (see `camwatch-defaults.yaml` for more settings)
- `python -m pip inatall -m requirements.txt`
- `python src/capture.py {channel_name}`


