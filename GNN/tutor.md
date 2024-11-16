### Install conda  
- [Github Page](https://github.com/conda/conda)  
- Select your version from the Distribution [page](https://repo.anaconda.com/archive/).  

### Basics of conda  
```shell
conda create --name env python=3.8.20
conda active env
```

### Prerequisite  
```shell
pip install numpy pandas matplotlib tqdm networkx -i https://pypi.tuna.tsinghua.edu.cn/simple

# Download Chinese Font for matplotlib 

loc=`pip show matplotlib | grep Location | awk '{print $2}'`

wget https://github.com/StellarCN/scp_zh/raw/refs/heads/master/fonts/SimHei.ttf -O ${loc}/matplotlib/mpl-data/fonts/ttf/SimHei.ttf

rm -rf ~/.cache/matplotlib
```

### Install cuda  
- [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit)  
- [pytorch](https://pytorch.org/)  
- [pyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
```python
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```




