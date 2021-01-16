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

## Reference

If you found this code useful for your research, please cite the following paper:

```
Jialu Wang, Yang Liu, and Caleb Levy. 2021. Fair Classification with Group-Dependent Label Noise. In ACM Conference on Fairness, Accountability, and Transparency (FAccT ’21), March 1–10, 2021, Virtual Event, Canada. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3442188.3445915
```
