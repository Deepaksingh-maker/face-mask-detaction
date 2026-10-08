#!/usr/bin/env bash
set -o errexit

# Clean out any conflicting or 5.0.0 opencv packages
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
pip install -r requirements.txt
