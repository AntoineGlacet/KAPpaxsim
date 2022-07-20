KAPpaxsim
==============================

Simulation tools based on Simpy library applied to KAP terminal Pax flows

Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data
    │   ├── interim            <- Intermediate data that has been transformed. (not uploaded to github) 
    │   ├── processed          <- The final, canonical data sets for modeling. (not uploaded to github) 
    │   ├── raw                <- The original, immutable data dump. (not uploaded to github) 
    │   └── secret             <- Secret data like Sharepoint login (not uploaded to github) 
    │       └── .env           <- File with local environment parameters
    │                               and passwords. Please ask separately (not uploaded to github) 
    │
    ├── notebooks              <- Jupyter notebooks for analysis and report creation
    │   ├── full               <- Including outputs (not uploaded to github) 
    │   └── stripped           <- Manually copied and stripped of outputs (uploaded to github)
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── environement.yml       <- The requirements file for reproducing the analysis environment, e.g.
    │                             generated with `conda export > environment.yml`
    │
    ├── setup.py               <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes src a Python module
    │   │
    │   ├── utils              <- regroup utilities in a module  
    │   │   ├── __init__.py    <- Makes utils python module
    │   │   ├── graph.py       <- For uniform graphs
    │   │   ├── helpers.py     <- Calculate stuff (eg.LBS)
    │   │   ├── optimizers.py  <- Optimizers & callbacks
    │   │   ├── sharepoint.py  <- download/upload data with sharepoint
    │   │
    │   ├── simfunc            <- Scripts to simulate a busy day
    │   │   ├── __init__.py    <- Makes simfunc a python module
    │   │   ├── KIX_T1a.py
    │   │   ├── KIX_T1a_covid.py
    │   │   ├── KIX_T1d.py
    │   │   ├── KIX_T1d_CUSBD.py
    │   │   ├── KIX_T2a.py
    │   │   └── KIX_T2d.py

Quick start guide
------------

1. clone the repo
2. create a new conda environment from environment.yml
3. install src (pip install -e . in conda terminal from KAPpaxsim)
4. run the tutorial notebook

Detailed start guide
------------

##### OUTDATED TO BE REWRITTEN #######

1. Install & start [Anaconda](https://www.anaconda.com/products/individual "Anaconda download") <-
to install and manage python environments
2. Download & unzip [source code](https://github.com/AntoineGlacet/KAPpaxsim/archive/refs/heads/main.zip "download code as a zip") in your project directory
3. Install the conda env from template
    1. locate environment.yml in root of downloaded code
    2. open a conda command prompt and go to that directory (root)
    3. execute `conda env create --name myenv --file environment.yml
4. install src (pip install -e . in conda terminal from root)
5. run the tutorial notebook /notebook/tutorial.ipynb

Remarks
------------


Project under progress and regularly updated
