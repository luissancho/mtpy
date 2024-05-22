#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF'
unset SUDO_UID

WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda/
source "$WORKING_DIR/miniconda/bin/activate"

for env in $WORKING_DIR/miniconda/envs/*; do
    BASENAME=$(basename "$env")
    source activate "$BASENAME"
    
    python -m ipykernel install --user --name "$BASENAME" --display-name "$BASENAME"
    
    wget https://mtpy.s3.eu-west-1.amazonaws.com/requirements.txt
    pip install -r $PWD/requirements.txt
done

IDLE_TIME=3600

wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py
(crontab -l 2>/dev/null; echo "5 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME") | crontab -

EOF

echo "Restarting the Jupyter server.."
systemctl restart jupyter-server
