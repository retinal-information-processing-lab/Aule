#!/bin/bash
# Launch the Aule GUI in its conda environment (creates the env on first use).
parent_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd "$parent_path"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if ! conda env list | grep -qE "^aule[[:space:]]"; then
    echo "Environment 'aule' not found: creating it from environment.yml (a few minutes)..."
    conda env create -f environment.yml -y || { echo "Environment creation failed."; read -p "Press Enter to close."; exit 1; }
fi

conda activate aule
python -m aule gui "$@"
