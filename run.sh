#!/bin/bash
# Move to the script's directory
cd "$(dirname "$0")" || exit

# Activate the virtual environment
source venv/bin/activate

# Execute the workflow script
python main_workflow.py
