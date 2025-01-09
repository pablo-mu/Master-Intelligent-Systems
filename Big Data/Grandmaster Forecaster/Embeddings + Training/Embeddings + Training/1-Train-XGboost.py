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
Esto fue una prueba
df = pd.read_parquet('merged')
df = df[df['WhiteElo']>=2500]
df = df[df['BlackElo']>=2500]
df_data = df[['Event', 'WhiteElo', 'BlackElo','Opening','Moves']]
labels = df[['Result']]
del df
'''
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reading data from a parquet file
df = pd.read_parquet('data_concat_embed_R')
# Dropping the 'ID' column from the dataframe
df.drop('ID', inplace = True, axis= 1)

# Extracting labels data and mapping them to numerical values
labels = df[['Result']]
df.drop('Result', inplace = True, axis = 1)
class_mapping = {'0-1': 0, '1-0': 1, '1/2-1/2': 2}
labels['Result'] = labels['Result'].map(class_mapping)


#embeddings = pd.read_csv('embeddings.csv')
#df = pd.concat([df_data.reset_index(), embeddings], axis = 1)
#df.set_index('index',inplace = True)

# Extracting categorical variables, excluding 'Moves' feature
df_cat = df.select_dtypes(exclude = ['int64', 'float64','bool']).copy()
df_cat.drop(['Moves'], axis=1,inplace = True)
# Excluding 'Moves' feature from the original dataframe
df.drop(['Moves'], axis=1,inplace = True)
# One-hot-encoding the categorical variables
df_cat_dm = pd.get_dummies(df_cat)
# Extracting numerical data
df_num = df.select_dtypes(include=['int64', 'float64', 'bool']).copy()

# Merging numerical data and one-hot-encoded categorical data
df_final = pd.concat([df_num,df_cat_dm],axis = 1)

# Deleting unnecessary dataframes to save memory
del df_num
del df_cat_dm
del df
del df_cat

# Splitting the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(df_final, labels, test_size=0.2, random_state=42)

# Split the remaining 20% into 50% testing and 50% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#to save memory
del df_final

# Columns to normalize
columns_to_normalize = ['WhiteElo', 'BlackElo'] + [f'C{i}' for i in range(384)]

# Scaling the data using StandardScaler
scaler = StandardScaler().fit(X_train.loc[:, columns_to_normalize].astype('float32'))
X_train_scale = X_train.copy()
X_train_scale.loc[:, columns_to_normalize] = scaler.transform(X_train.loc[:, columns_to_normalize].astype('float32'))

# Transform the selected columns in the test set
X_test_scale = X_test.copy()
X_test_scale.loc[:, columns_to_normalize] = scaler.transform(X_test.loc[:, columns_to_normalize].astype('float32'))

# Transform the selected columns in the validation set
X_val_scale = X_val.copy()
X_val_scale.loc[:, columns_to_normalize] = scaler.transform(X_val.loc[:, columns_to_normalize].astype('float32'))

'''
That was a test using another type of data, it is more efficient but then we have problems 
to create the SHAP analysis.

dtrain = xgb.DMatrix(X_train, label=y_train, nthread=8,  feature_names=list(X_train.columns))
dtest = xgb.DMatrix(X_test, label=y_test, nthread=8,  feature_names=list(X_test.columns))
dval = xgb.DMatrix(X_val, label=y_val, nthread=8, feature_names=list(X_val.columns))
'''

'''
Another test: This include normalizing the bool data

Normalizing data with the train data.
scaler = StandardScaler().fit(X_train.astype('float64'))
X_train_scale = scaler.fit_transform(X_train.astype('float64'))
X_test_scale = scaler.fit_transform(X_test.astype('float64'))
X_val_scale = scaler.fit_transform(X_val.astype('float64'))

#Setting as dataframe
X_train_scale = pd.DataFrame(X_train_scale, columns = X_train.columns)
X_test_scale = pd.DataFrame(X_test_scale, columns = X_test.columns)
X_val_scale = pd.DataFrame(X_test_scale, columns = X_val.columns)
'''

# Deleting unnecessary dataframes to save memory

del X_train
del X_test
del X_val
del X_temp
del y_temp
del labels
del scaler

#Converting the dataframe into 'float32' to save memory. Also, this is to convert
#the bool data into 1s and 0s due to make simplier the SHAP analysis.
X_val_scale = X_val_scale.astype('float32')
X_test_scale = X_test_scale.astype('float32')
X_train_scale = X_train_scale.astype('float32')

#dataframe to series
y_test = y_test.squeeze()
y_train = y_train.squeeze()
y_val = y_val.squeeze()

# Initializing an XGBoost model for multiclasification problem
xgb_model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', early_stopping_rounds=3, njobs = 8)

# Fitting the model on the training set and evaluating on the validation set
xgb_model.fit(X_train_scale, y_train, eval_set=[(X_val_scale, y_val)])
# Making predictions on the training set and evaluating performance
preds_train = xgb_model.predict(X_train_scale)
labels_cm = ['0-1', '1-0', '1/2-1/2']
print('accuracy:', accuracy_score(y_train, preds_train))
print('precision:', precision_score(y_train, preds_train, average='weighted'))
print('recall:', recall_score(y_train, preds_train, average='weighted'))

'''
accuracy: 0.6573650712830957
precision: 0.6582049484594383
recall: 0.6573650712830957
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
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()

'''
accuracy: 0.6346639511201629
precision: 0.6246219137542931
recall: 0.6346639511201629

'''
#xgb_model.save_model('trained_model_binary.json')
# Save the model to a file
dump(xgb_model, 'trained_model_R.joblib')

explainer = shap.TreeExplainer(xgb_model)
explanation = explainer(X_test_scale)
shap_values = explanation.values
shap.summary_plot(shap_values, X_test_scale, class_names = ['0-1', '1-0', '1/2-1/2'])
shap.decision_plot(explainer.expected_value, shap_values, X_test_scale, ignore_warnings = True)
shap.force_plot(explainer.expected_value, shap_values, X_test_scale)
explainer = shap.TreeExplainer(xgb_model, X_train_scale, feature_names=X_test_scale.columns)
shap_values = explainer(X_test_scale)
shap.plots.scatter(shap_values[:,"WhiteElo"],color=shap_values)
'''
#  load the saved model back into your code
loaded_model = load('first_trained_model.joblib')
'''


'''
# Bayesian Optimization using Hyperopt for XGBoost parameters

'''
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 0,9),
        'eta': hp.uniform('eta',0,1),
        'reg_lambda': hp.uniform('reg_lambda', 0,5),
        'min_child_weight' : hp.quniform('min_child_weight',0,10,1),
    }

# Objective function to optimize the hyperparameters
'''
We could define a K-cross-validation with the next function. However,
we should do a different splitting of the data, like 85 for train and 
15 % test because the cross_val_score function performs the split between train and 
validation.

from sklearn.model_selection import cross_val_score
def objective(space):
    clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', #early_stopping_rounds=3,
        max_depth=int(space['max_depth']), gamma=space['gamma'],
        eta=space['eta'], reg_lambda = space['reg_lambda'], min_child_weight = space['min_child_weight'])
    accuracy = cross_val_score(clf, X_train_scale, y_train, cv=5).mean()
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}
'''

# HoldOut

def objective(space):
    clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', #early_stopping_rounds=3,
        max_depth=int(space['max_depth']), gamma=space['gamma'],
        eta=space['eta'], reg_lambda = space['reg_lambda'], min_child_weight = space['min_child_weight'])

    evaluation = [(X_train_scale, y_train), (X_val_scale, y_val)]

    clf.fit(X_train_scale, y_train, eval_set=evaluation, verbose=False)

    pred = clf.predict(X_val_scale)
    accuracy = accuracy_score(y_val, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}
trials = Trials()
# Using Hyperopt to perform Bayesian optimization
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 6,
                        trials = trials)

# Creating a new XGBoost classifier with the best hyperparameters
clf = clone(xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', early_stopping_rounds=5,
        n_estimators=180, eta = best_hyperparams['eta'], max_depth=int(best_hyperparams['max_depth']), gamma=best_hyperparams['gamma'], reg_lambda=best_hyperparams['reg_lambda']))

# Fitting the model on the training set and evaluating on the validation set
clf.fit(X_train_scale,y_train,eval_set=[(X_val_scale, y_val)])

# Making predictions on the training set and evaluating performance
preds_train = clf.predict(X_train_scale)
print('accuracy:', accuracy_score(y_train, preds_train))
print('precision:', precision_score(y_train, preds_train, average='weighted'))
print('recall:', recall_score(y_train, preds_train, average='weighted'))
labels_cm = ['0-1', '1-0', '1/2-1/2']

cm = confusion_matrix(y_train, preds_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()

''' Train
accuracy: 0.633733407079646
precision: 0.6233146164982641
recall: 0.633733407079646
'''

'''
K-fold

accuracy: 0.6447428716904277
precision: 0.6393395613530649
recall: 0.6447428716904277
'''
# Making predictions on the test set and evaluating performance
preds_test = clf.predict(X_test_scale)
print('accuracy:', accuracy_score(y_test, preds_test))
print('precision:', precision_score(y_test, preds_test, average='weighted'))
print('recall:', recall_score(y_test, preds_test, average='weighted'))
# Plotting the confusion matrix for the test set
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
disp.plot()
'''Test
accuracy: 0.6264159292035398
precision: 0.617906600622388
recall: 0.6264159292035398


5-fold-cv
accuracy: 0.6347250509164969
precision: 0.6246422502340854
recall: 0.6347250509164969
'''
# Saving the trained model to a file
dump(xgb_model, 'trained_model_opt.joblib')

# Creating a SHAP explainer for the XGBoost model
explainer = shap.TreeExplainer(clf)
explanation = explainer(X_test_scale)
shap_values = explanation.values
shap.summary_plot(shap_values, X_test_scale, class_names = ['0-1', '1-0', '1/2-1/2'])
shap.decision_plot(explainer.expected_value, shap_values, X_test_scale, ignore_warnings = True)
shap.force_plot(explainer.expected_value, shap_values, X_test_scale)
explainer = shap.TreeExplainer(clf, X_train_scale, feature_names=X_test_scale.columns)
shap_values = explainer(X_test_scale)
shap.plots.scatter(shap_values[:,"WhiteElo"],color=shap_values)


'''
# GridSearchCV with k-fold cross validation for Hyperparameter Tuning with XGBoost. However, 
this is very slow due to the amount of data. 
'''

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

# Defining the search space for hyperparameters
param_grid = {
    'max_depth': list(range(3, 10)),
    'gamma': list(range(10)),
    'eta': [i/10.0 for i in range(11)],
    'reg_lambda': list(range(6)),
    'min_child_weight': list(range(11)),
}

# Create XGBClassifier with fixed parameters
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    early_stopping_rounds=3
)

# Create StratifiedKFold cross-validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Manually split your data into batches
batch_size = 39280  # Set your desired batch size
num_batches = X_train_scale.shape[0] // batch_size

#We create batches to increase the performance and due to lack of memory.
for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = (batch_num + 1) * batch_size

    # Extract the current batch
    X_batch = X_train_scale[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1)

    # Fit the GridSearchCV object to the current batch
    grid_search.fit(X_batch, y_batch)

    # Get the best hyperparameters from the grid search
    best_hyperparams = grid_search.best_params_

    # Print the best hyperparameters for the current batch
    print(f"Best Hyperparameters (Batch {batch_num + 1}):", best_hyperparams)

    # Create XGBClassifier with the best hyperparameters
    best_xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        early_stopping_rounds=3,
        max_depth=best_hyperparams['max_depth'],
        gamma=best_hyperparams['gamma'],
        eta=best_hyperparams['eta'],
        reg_lambda=best_hyperparams['reg_lambda'],
        min_child_weight=best_hyperparams['min_child_weight']
    )

    # Fit the model with the best hyperparameters on the full training set
    best_xgb_clf.fit(X_train_scale, y_train, eval_set=[(X_val_scale, y_val)])

    # Make predictions on the validation set
    pred = best_xgb_clf.predict(X_test_scale)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, pred > 0.5)
    print(f"Validation Accuracy (Batch {batch_num + 1}):", accuracy)

# Fit the model with the best hyperparameters
best_xgb_clf.fit(X_train_scale, y_train, eval_set=[(X_val_scale, y_val)])
