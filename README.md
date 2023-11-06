# KS_EVALUATION

KS_EVALUATION is a Python library that evaluates the Kolmogorov–Smirnov (KS) metric, which compares the two cumulative distributions (binary target) and returns the maximum difference between them.

## Installation
```bash
pip install git+https://github.com/luisguicomp/ks_evaluation
```

## Use
Functions:
* functions.ks_calculate  --> returns the KS value and the KS table
* functions.plot_ks_chart --> Plot the KS graph based on the binary target

#### Example:
```python
# KS calculus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#Importing the library
from ks_evaluation import ks_calculate, plot_ks_chart

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Input parameters: predictive model, predictor variables (X), array with binary target (y)
ks_score, ks_table = ks_calculate(rfc, X_test, y_test, target='y', verbose=True)
plot_ks_chart(ks_table)
```
## References

https://www.listendata.com/2019/07/KS-Statistics-Python.html

https://medium.com/@xiaowei_6531/using-ks-stat-as-a-model-evaluation-metric-in-scikit-learns-gridsearchcv-33135101601c

https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

## License
LGR - Data Science
