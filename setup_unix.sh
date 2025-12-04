#!/bin/bash

echo '[1/3] Creating virtual environment (.venv)...'
python3 -m venv .venv

if [ ! -d ".venv" ]; then
    echo 'Error: The virtual environment could not be created'
    exit 1
fi

echo '[2/3] Activating environment...'
source .venv/bin/activate

echo '[3/3] Installing requirements...'
pip install --upgrade pip
pip install -r requirements.txt

echo 'Done! Environment is active.'


# Antes de executar, rode no o comando chmod +x setup_unix.sh