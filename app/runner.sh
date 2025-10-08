#!/bin/bash
export JETSON_MODEL_NAME=JETSON_ORIN_NANO
sudo busybox devmem 0x2448030 w 0xA
exec sudo python3 app.py
