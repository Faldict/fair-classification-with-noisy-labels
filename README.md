# Fair ERM

This code is the official implementation of *Fair Classification with Group-Dependent Label Noise*.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Running

To run the code

```
python3 run.py
```

Specifically, the surrogate loss function and the group peer loss function is implemented in `PeerLoss.py`.
The proxy fairness constraints are implemented in `ProxyConstraint.py`. The code for data preprocessing is 
in `datasets.py`. `utils.py` defines some utility functions.

