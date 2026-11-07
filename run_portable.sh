#!/bin/bash
# Move to the script's directory
cd "$(dirname "$0")" || exit

# Tell Python to use the dependencies stored locally on the pendrive
export PYTHONPATH="$(pwd)/pendrive_libs"

# Execute using the host computer's Python
python3 main_workflow.py
