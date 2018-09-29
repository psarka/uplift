# uplift

[![Build Status](https://travis-ci.org/psarka/uplift.svg?branch=master)](https://travis-ci.org/psarka/uplift) 

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

## Resources

1999, N.Radcliffe, P.Surry, Differential Response Analysis: Modeling True Responses by Isolating the Effect of a Single Action  
2002, B.Hansotia, B.Rukstales, Incremental Value Modeling  
2007, N.Radcliffe, Using Control Groups to Target on Predicted Lift: Building and Assessing Uplift Models  
2010, P.Rzepakowski, S.Jaroszewicz, Decision trees for uplift modeling  
2011, N.Radcliffe, P.Surry, Real-World  Uplift Modelling with Significance-Based Uplift Trees  
2012, P.Rzepakowski, S.Jaroszewicz, Decision trees for uplift modeling with single and multiple treatments  
2015, L.Guelman, M.Guillen, M.Perez-Marin, Uplift Random  Forests  
2015, M.Soltys, S.Jaroszewicz, P.Rzepakowski, Ensemble methods for uplift modeling  
2017, W.Verbeke, C.Bravo, B.Baesens, Profit drive business analytics: A practitioner's guide to transforming big data into added value. (Chapter  4)

- Wiki page: https://en.wikipedia.org/wiki/Uplift_modelling
- R package: http://cran.r-project.org/web/packages/uplift/index.html
