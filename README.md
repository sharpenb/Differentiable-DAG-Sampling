# Probabilistic DAG Learning

## Repository Structure



## Setup

After cloning the repository, there are three subdirectories: `gran-dag`, `probabilistic_dag_learning` and `DAG-NF`.

### Python/R environment

```
conda create -n probabilistic-dag-learning python=3.7
conda activate probabilistic-dag-learning
conda install --force-reinstall -y -q --name probabilistic-dag-learning -c conda-forge --file requirements-conda.txt
pip install -r requirements-pip.txt
```

```
cd z_gran-dag/notears/notears/cppext
python setup.py install
R < install_requirements.R --no-save
```

Old instructions: Create a new conda environment from the requirements file

```
conda create --name <env_name> --file requirements-conda.txt
```

Activate the environment and install the pip requirements

```
pip install -r requirements-pip.txt
```

Next, go to `gran-dag/notears/notears/cppext` and run:

```
python setup.py install
```

Next, install the needed R-packages:

```
R < install_requirements.R --no-save
```

### Data
Data is stored under `gran-dag/data` in zip files. They need to be extracted to be used for experiments.

If you want to run the likelihood experiment, you will have to generate the data for this first. Go to `probabilistic_dag_learning` and run

```
python generate_perturbed_dags.py --noise 0.15
```

for a noise level of 0.15. For the experiments both noise levels of 0.05 and 0.15 were used. You might wanna change the output directory and number of DAGs in the script to your liking.


## Experiments

### Sanity Checks - Probabilistic DAG Generator
For sanity checks of the Probabilistic DAG Generators, see `probabilistic_dag_learning/experiments`. There are exemplary scripts for both direct DAG Learning and Likelihood learning. Open them to adjust parameters to your liking, run them and find results in `probabilistic_dag_learning/experiments/results`. E.g. for direct dag learning, run:

```
python direct_dag_learning.py
```

### Sanity Checks -  DAG-Masked Autoencoder
For the sanity checks of the masked autoencoder see the other two scripts in the same folder `gt_vs_random.py` and `gt_vs_reverse.py`. The same as above applies.

### Structure Learning in GraN-DAG environment
For structure learning in the GraN-DAG environment, there are multiple options. To run on just one DAG, you can use `gran-dag/pdl/main.py`. Parameters are explained in the file. Run from within the `gran-dag` folder e.g.:

```
python pdl/main.py --data-path data/data_p10_e10_n1000_GP --i-dataset 1 --pdg-name topk --dagma-name strict --hidden-list 50 50
```

To run the baseline methods, run the equivalent main methods in `gran-dag/notears` and `gran-dag/`.

More recommended, especially when running on multiple DAGS is to use one of the experiment scripts in `gran-dag/experiments`. There is one script for each method. Fill in the datasets you want to run on and choose which DAGs to run on. Then run the script.

### DAG-NF
Follow the readme in `DAG_NF` for instructions on how to download data.

Some experiments on DAG Normalizing Flows can be run using the jupyter notebook in `DAG-NF/Experiments.ipynb`. The key here is the use of the conditioner `DAG-NF/models/Conditionners/DAGConditionerPDL.py`. This uses the probabilistic dag learning approach.