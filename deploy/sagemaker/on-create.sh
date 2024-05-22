#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF'
unset SUDO_UID

# Install a separate conda installation via Miniconda
WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
mkdir -p "$WORKING_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda" 
rm -rf "$WORKING_DIR/miniconda.sh"

# Create a custom conda environment
source "$WORKING_DIR/miniconda/bin/activate"
KERNEL_NAME="mtpy"
PYTHON="3.9"

conda create --yes --name "$KERNEL_NAME" python="$PYTHON"
conda activate "$KERNEL_NAME"

pip install --quiet ipykernel

# Install the required packages
wget https://wuolah-redshift.s3.eu-west-1.amazonaws.com/requirements.txt
pip install -r $PWD/requirements.txt

EOF
