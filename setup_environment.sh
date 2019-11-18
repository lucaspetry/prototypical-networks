conda create -n snell python=3.6 -y
conda activate snell
conda install unzip -y
pip install -r requirements.txt
pip install -e ../.
pip install -e .
