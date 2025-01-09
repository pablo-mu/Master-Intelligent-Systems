import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone
import shap
import seaborn as sns
from joblib import dump, load
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

'''
In this programme we train an XGboost model for binary classification of chess-games
with its embeddings. We filter the chess games that have huge difference of ELO.

'''

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reading data from a parquet file
df = pd.read_parquet('no_diff_elo')

#Filter for binary classification
df = df[df['Result']!='1/2-1/2']
df.drop('ID', inplace = True, axis= 1)

# Extracting labels data and mapping them to numerical values
labels = df[['Result']]
df.drop('Result', inplace = True, axis = 1)
class_mapping = {'0-1': 0, '1-0': 1}
labels['Result'] = labels['Result'].map(class_mapping)

# Extracting categorical variables, excluding 'Moves' feature
df_cat = df.select_dtypes(exclude = ['int64', 'float64','bool']).copy()
#Exclude the moves features because we already have it as embedding
df_cat.drop(['Moves'], axis=1,inplace = True)
#Also from the original data
df.drop(['Moves'], axis=1,inplace = True)
# One-hot-encoding the categorical variables
df_cat_dm = pd.get_dummies(df_cat)
#Numerical data
df_num = df.select_dtypes(include=['int64', 'float64', 'bool']).copy()

#merging numerical data and one-hot-encodings
df_final = pd.concat([df_num,df_cat_dm],axis = 1)

# Deleting unnecessary dataframes to save memory
del df_num
del df_cat_dm
del df
del df_cat

#df.drop(['Moves'], axis = 1, inplace= True)

# Splitting the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(df_final, labels, test_size=0.2, random_state=42)

# Split the remaining 20% into 50% testing and 50% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

'''
X_train, X_temp, y_train, y_temp = train_test_split(df, labels, test_size=0.2, random_state=42)

# Split the remaining 20% into 50% testing and 50% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
'''
#to save memory
del df_final
del X_temp
del y_temp

columns_to_normalize = ['WhiteElo', 'BlackElo'] + [f'C{i}' for i in range(384)]
scaler = StandardScaler().fit(X_train.loc[:, columns_to_normalize].astype('float32'))
X_train_scale = X_train.copy()
X_train_scale.loc[:, columns_to_normalize] = scaler.transform(X_train.loc[:, columns_to_normalize].astype('float32'))

# Transform the selected columns in the test set
X_test_scale = X_test.copy()
X_test_scale.loc[:, columns_to_normalize] = scaler.transform(X_test.loc[:, columns_to_normalize].astype('float32'))

# Transform the selected columns in the validation set
X_val_scale = X_val.copy()
X_val_scale.loc[:, columns_to_normalize] = scaler.transform(X_val.loc[:, columns_to_normalize].astype('float32'))

#Converting the dataframe into 'float32' to save memory. Also, this is to convert
#the bool data into 1s and 0s due to make simplier the SHAP analysis.
X_val_scale = X_val_scale.astype('float32')
X_test_scale = X_test_scale.astype('float32')
X_train_scale = X_train_scale.astype('float32')


# Deleting unnecessary dataframes to save memory
del X_train
del X_test
del X_val

del labels
del scaler

#dataframe to series
y_test = y_test.squeeze()
y_train = y_train.squeeze()
y_val = y_val.squeeze()

'''
cat_columns =X_val_scale.select_dtypes(exclude = ['int64', 'float64','bool']).columns
X_train_scale[cat_columns] = X_train_scale[cat_columns].astype("category")
X_test_scale[cat_columns] = X_test_scale[cat_columns].astype("category")
X_val_scale[cat_columns] = X_val_scale[cat_columns].astype("category")
'''

# Initializing an XGBoost model for multiclasification problem
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=5, njobs = 8)

# Fitting the model on the training set and evaluating on the validation set
xgb_model.fit(X_train_scale, y_train, eval_set=[(X_val_scale, y_val)])

# Making predictions on the training set and evaluating performance
preds_train = xgb_model.predict(X_train_scale)
# preds = cross_val_predict(xgb_model, X_train_scale, y_train, cv=3, n_jobs=-1)
labels_cm = ['0-1', '1-0']
print('accuracy:', accuracy_score(y_train, preds_train))
print('precision:', precision_score(y_train, preds_train, average='weighted'))
print('recall:', recall_score(y_train, preds_train, average='weighted'))

'''
accuracy: 0.5347417941161137
precision: 0.5509641830712416
recall: 0.5347417941161137

'''
# Plotting the confusion matrix for the training set
cm = confusion_matrix(y_train, preds_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()

del preds_train

#Making predictions from the test set.
preds_test = xgb_model.predict(X_test_scale)
print('accuracy:', accuracy_score(y_test, preds_test))
print('precision:', precision_score(y_test, preds_test, average='weighted'))
print('recall:', recall_score(y_test, preds_test, average='weighted'))
'''
accuracy: 0.5176153091265947
precision: 0.5117834940131792
recall: 0.5176153091265947
'''
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()

explainer = shap.TreeExplainer(xgb_model)
explanation = explainer(X_test_scale)
shap_values = explanation.values
#summary plot
shap.summary_plot(shap_values, X_test_scale, class_names = ['0-1', '1-0'])
shap.decision_plot(explainer.expected_value, shap_values, X_test_scale, ignore_warnings = True)
#shap.force_plot(explainer.expected_value, shap_values, X_test_scale)

explainer = shap.TreeExplainer(xgb_model, X_train_scale, feature_names=X_test_scale.columns)
shap_values = explainer(X_test_scale)
shap.plots.scatter(shap_values[:,"WhiteElo"],color=shap_values)

dump(xgb_model, 'trained_model_binary.joblib')

#This parameter is to try to de-bias the model due to possible imbalance in the data.
class_counts = y_train.value_counts()
proportion_instances = class_counts[0] / class_counts[1]  # negative instances
del class_counts

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 0,9),
        'eta': hp.uniform('eta',0,1),
        'reg_lambda': hp.uniform('reg_lambda', 0,5),
        'min_child_weight' : hp.quniform('min_child_weight',0,10,1),
        'scale_pos_weight' : hp.choice('scale_pos_weight',[0, proportion_instances])
    }

'''
We could define a K-cross-validation with the next function. However,
we should do a different splitting of the data, like 85 for train and 
15 % test because the cross_val_score function performs the split between train and 
validation.

from sklearn.model_selection import cross_val_score
def objective(space):
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                            max_depth=int(space['max_depth']), gamma=space['gamma'],
                            eta=space['eta'], reg_lambda=space['reg_lambda'],
                            min_child_weight=space['min_child_weight'],
                            scale_pos_weight=space['scale_pos_weight'])
    accuracy = cross_val_score(clf, X_train_scale, y_train, cv=5).mean()
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}
'''

# Holdout
def objective(space):
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=3,
        max_depth=int(space['max_depth']), gamma=space['gamma'],
        eta=space['eta'], reg_lambda = space['reg_lambda'], min_child_weight = space['min_child_weight'],
                            scale_pos_weight = proportion_instances)

    evaluation = [(X_train_scale, y_train), (X_val_scale, y_val)]

    clf.fit(X_train_scale, y_train,
            eval_set=evaluation, verbose=False)

    pred = clf.predict(X_val_scale)
    accuracy = accuracy_score(y_val, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 6,
                        trials = trials)

clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=5,
         eta = best_hyperparams['eta'], max_depth=int(best_hyperparams['max_depth']), gamma=best_hyperparams['gamma'], reg_lambda=best_hyperparams['reg_lambda'],
                        min_child_weight = best_hyperparams['min_child_weight'],
                        scale_pos_weight=proportion_instances)


clf.fit(X_train_scale,y_train,eval_set=[(X_val_scale, y_val)])

preds_train = clf.predict(X_train_scale)
    # preds = cross_val_predict(xgb_model, X_train_scale, y_train, cv=3, n_jobs=-1)
print('accuracy:', accuracy_score(y_train, preds_train))
print('precision:', precision_score(y_train, preds_train, average='weighted'))
print('recall:', recall_score(y_train, preds_train, average='weighted'))
labels_cm = ['0-1', '1-0']
cm = confusion_matrix(y_train, preds_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()
'''
accuracy: 0.5405502381339377
precision: 0.5518081758426097
recall: 0.5405502381339377
'''

preds_test = clf.predict(X_test_scale)
print('accuracy:', accuracy_score(y_test, preds_test))
print('precision:', precision_score(y_test, preds_test, average='weighted'))
print('recall:', recall_score(y_test, preds_test, average='weighted'))
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()

'''
accuracy: 0.5135181550539745
precision: 0.5041360789815973
recall: 0.5135181550539745
'''

explainer = shap.TreeExplainer(clf)
explanation = explainer(X_test_scale)
shap_values = explanation.values
shap.summary_plot(shap_values, X_test_scale, class_names = ['0-1', '1-0'])
shap.decision_plot(explainer.expected_value, shap_values, X_test_scale, ignore_warnings = True)
#shap.force_plot(explainer.expected_value, shap_values, X_test_scale)
explainer = shap.TreeExplainer(clf, X_train_scale, feature_names=X_test_scale.columns)
shap_values = explainer(X_test_scale)
shap.plots.scatter(shap_values[:,"BlackElo"],color=shap_values)
shap.plots.scatter(shap_values[:,"WhiteElo"],color=shap_values)
