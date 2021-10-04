# Differentiable DAG Sampling

## Requirements

To install requirements:

```
conda create -n differentiable-dag-sampling python=3.7
conda activate differentiable-dag-sampling
conda install --force-reinstall -y -q --name differentiable-dag-sampling -c conda-forge --file requirements-conda.txt
pip install -r requirements-pip.txt
```

To run notebooks:
```
conda install -c conda-forge jupyterlab
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=differentiable-dag-sampling
python setup.py develop
```

To install R requirements (useful to load Sachs dataset):
```
cd /cppext
python setup.py install
R < install_requirements.R --no-save
```

### Data

You can find the datasets at the following anonymous [link](https://ln5.sync.com/dl/cbfe8c1e0#ygpj5jgf-kf5tqmnb-hzheftz7-dny2kxp8).

## Experiments

You can find a notebook to run DP-DAG and VI-DP-DAG in the folder `src/notebooks`.