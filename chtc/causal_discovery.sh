#!/bin/bash
echo "Starting run $1 ($2)..."
python3 causal_discovery.py --run_index $1 --run_name $2 --data_path $3 --alpha $4 --sample_size $5 --depth $6
echo "Done."