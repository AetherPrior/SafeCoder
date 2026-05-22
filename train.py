#!/usr/bin/env python3
"""Entry point for SafeCoder training (delegates to scripts/train.py)."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    train_script = Path(__file__).resolve().parent / 'scripts' / 'train.py'
    runpy.run_path(str(train_script), run_name='__main__')
