# Differentiable DAG Sampling


This repository presents the models of the paper:

[Differentiable DAG Sampling](https://openreview.net/pdf?id=9wOQOgNe-w)<br>
Bertrand Charpentier, Simon Kibler, Stephan Günnemann<br>
International Conference on Learning Representations (ICLR), 2022.

[[Paper](https://openreview.net/pdf?id=9wOQOgNe-w)|[Video](https://youtu.be/JiS7wJle2Ao)]

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

You can find the datasets at the following anonymous [link](https://ln5.sync.com/dl/b442986b0#5xpiy2n2-q9j87qze-kydrb7wn-xgqjiw2c).

## Experiments

You can find a notebook to run DP-DAG and VI-DP-DAG in the folder `src/notebooks`.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@incollection{dpdag,
title = {Differentiable DAG Sampling},
author = {Charpentier, Bertrand, Kibler, Simon and G\"{u}nnemann, Stephan},
booktitle = {International Conference on Learning Representations 10},
year = {2022},
}
