pip install --upgrade pip
conda create -n AggNet python=3.10
source /home/hew/anaconda3/bin/activate AggNet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install biotite -c conda-forge
conda install pandas matplotlib seaborn scikit-learn biopython openpyxl jupyter ipywidgets
pip install -r dependencies.txt
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric
pip install torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
