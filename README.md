KAPpaxsim
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
==============================

Simulation tools based on Simpy library applied to KAP terminal Pax flows

Project Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── interim             <- Intermediate data that has been transformed. (not uploaded to github)
    │   ├── processed           <- The final, canonical data sets for modeling. (not uploaded to github)
    │   ├── raw                 <- The original, immutable data dump. (not uploaded to github)
    │   └── secret              <- Secret data like Sharepoint login (not uploaded to github)
    │       └── .env            <- File with local environment parameters
    │                               and passwords. Please ask separately (not uploaded to github)
    │
    ├── notebooks               <- Jupyter notebooks for analysis and report creation
    │   ├── full                <- Including outputs (not uploaded to github)
    │   └── stripped            <- Manually copied and stripped of outputs (uploaded to github)
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── .pre-commit-config.yaml <- Pre-commit hooks (https://github.com/pre-commit/pre-commit)
    │                                 │
    ├── environement.yml        <- conda env file, use with `conda env create -f environment.yml`
    │                             generated with `conda env export > environment.yml`
    │
    ├── setup.py                <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py
    │   │
    │   ├── utils
    │   │   ├── __init__.py
    │   │   ├── simulation.py   <- Simulation running class
    │   │
    │   ├── simfunc
    │   │   ├── __init__.py
    │   │   ├── simparam.py     <- Simulation parameters class

Prerequisite
------------

- [conda ](https://docs.conda.io/en/latest/)

Quick start guide
------------

```bash
# 1. clone the repo
git clone https://github.com/AntoineGlacet/KAPpaxsim.git
# 2. create a new conda environment from environment.yml
cd KAPpaxsim
conda env create --file environment.yml
# 3. install source code
pip install -e .
```

Contribute
------------
- Fork
- Install pre-commit hooks
`pre-commit install`
- Install nb_stripout
`nbstripout --install`
- Open pull request

Remarks
------------

Project under progress and regularly updated
