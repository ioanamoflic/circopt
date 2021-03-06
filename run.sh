#!/bin/bash
echo "Create venv ..."
python3 -m venv .venv

echo "Activate venv ..."
source .venv/bin/activate

echo "Install requirements in venv ..."
python3 -m pip install -r requirements.txt

python3 -m identity_QL