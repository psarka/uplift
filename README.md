# uplift

Work in progress.

## Installation

For now run this to install:

```
sudo apt-get install build-essential python3-dev
```

```
git clone https://github.com/psarka/uplift
cd uplift
python3.6 -m 'venv' venv
source venv/bin/activate
pip install -e .
```

## Example

```python
from uplift.ensemble import RandomForestClassifier
from uplift.datasets import make_radcliffe_surry
from uplift.metrics import  qini_q

X_train, y_train, group_train = make_radcliffe_surry()
X_test, y_test, group_test, uplift_test = make_radcliffe_surry(return_uplift=True)

rfc = RandomForestClassifier(n_estimators=50, min_samples_leaf=200, criterion='uplift_gini')

rfc.fit(X_train, y_train, group_train)
uplift_pred = rfc.predict_uplift(X_test)

print(qini_q(y_test, uplift_pred, group_test))
```

