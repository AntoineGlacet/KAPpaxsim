KAPpaxsim
==============================

Simulation tools based on Simpy library applied to KAP terminal Pax flows

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── secret         <- Secret data like Sharepoint login
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── profiles.py    <- generates Pax & counters from schedule   
    │   │
    │   ├── utils.py       <- utilities (import schedules from Sharepoint, optimizers, callbacks
    │   │                     an various support function)
    │   │
    │   ├── simfunc         <- Scripts to simulate a busy day
    │   │   ├── __init__.py <- Makes simfunc a python module
    │   │   ├── KIX_T1a.py
    │   │   ├── KIX_T1a_covid.py
    │   │   ├── KIX_T1d.py
    │   │   ├── KIX_T1d_CUSBD.py
    │   │   ├── KIX_T2a.py
    │   │   └── KIX_T2d.py

Remarks
------------
Project under progress and regularly updated