#!/bin/bash

echo "Terminating processes on port 6009..."
lsof -ti:6009 | xargs -r kill -9

cd ~/Documents/splatviz || { echo "Directory not found"; exit 1; }

echo "Activating Conda environment 'gs-view'..."
eval "$(conda shell.bash hook)"
conda activate gs-view || { echo "Failed to activate conda environment"; exit 1; }

echo "Starting splatviz..."
python run_main.py --mode=attach --port=6009
