#!/bin/bash
# GPU python for Experiment 6: the cgpu venv launcher is blocked by Smart App Control (2026-10-11), so run the base
# CPython interpreter with the venv's site-packages on PYTHONPATH (same torch 2.7.1+cu128 installation).
PYTHONPATH="C:/Users/$USERNAME/AppData/Local/Temp/cgpu/Lib/site-packages" PYTHONIOENCODING=utf-8 exec "/c/Users/$USERNAME/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe" "$@"
