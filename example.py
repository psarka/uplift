from uplift.ensemble import RandomForestClassifier
from uplift.datasets import make_radcliffe_surry
from uplift.metrics import qini_q

X_train, y_train, group_train = make_radcliffe_surry()
X_test, y_test, group_test = make_radcliffe_surry()

rfc = RandomForestClassifier(n_estimators=50, min_samples_leaf=200, criterion='uplift_gini')

rfc.fit(X_train, y_train, group_train)
uplift_pred = rfc.predict_uplift(X_test)

print(qini_q(y_test, uplift_pred, group_test))
