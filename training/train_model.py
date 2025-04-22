# 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import f1_score,roc_auc_score
# from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pickle



dfx= pd.read_csv('/home/veerendra/SPE/MlOps_chatbot/training/patients_data.csv')

dfx[dfx.columns[1:]].sum(axis=0).sort_values()


data = dfx.iloc[:,1:].values
labels = dfx['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7,random_state=42)
x_train, x_val, y_train,y_val=train_test_split(data,labels,test_size=0.3,random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_val.shape,y_val.shape)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val=le.transform(y_val)

y=le.classes_
# y


# Assuming you've already loaded your dataset into DataFrames x_train, x_test, x_val, y_train, y_test, y_val

classifiers = {
    'ExtraTrees': ExtraTreesClassifier()
}

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

# K-fold Cross Validation model evaluation
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='f1_weighted')
    print(f'{name} cross-validation mean F1 score: %.3f' % cv_scores.mean())

    # Train and test each classifier
    clf.fit(x_train, y_train)

    test_predictions = clf.predict(x_test)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_roc = roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovr')
    print(f'{name} test F1 Score: {test_f1:.4f}, AUC-ROC Score: {test_roc:.4f}')

    val_predictions = clf.predict(x_val)
    val_f1 = f1_score(y_val, val_predictions, average='weighted')
    val_roc = roc_auc_score(y_val, clf.predict_proba(x_val), multi_class='ovr')
    print(f'{name} validation F1 Score: {val_f1:.4f}, AUC-ROC Score: {val_roc:.4f}')
    pickle.dump(clf, open(f"{name}", "wb"))



