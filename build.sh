#!/usr/bin/env bash
set -o errexit

# Ensure clean OpenCV installation on Render
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
pip install -r requirements.txt
