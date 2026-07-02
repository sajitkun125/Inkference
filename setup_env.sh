#!/bin/bash

pyenv local 3.11.3

python -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Setup abgeschlossen und Requirements installiert."
else
    echo "⚠️ Keine requirements.txt gefunden."
fi
