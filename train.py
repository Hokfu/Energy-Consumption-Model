import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

max_depth = 10
n_estimators = 90
min_samples_leaf = 3
output_file = f'model_1.bin'

df = pd.read_csv('Steel_industry_data.csv')
df.columns = df.columns.str.lower()
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in string_columns:
    df[c] = df[c].str.lower()
    
df.date = pd.to_datetime(df.date, format="%d/%m/%Y %H:%M")
df.date = df.date.dt.to_period('M')
df.date = df.date.apply(str)
del df['day_of_week']

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.load_type
y_val = df_val.load_type
y_test = df_test.load_type

feature_columns = list(df.dtypes.index)[:-1]

train_dict = df_train[feature_columns].to_dict(orient = 'records')
val_dict = df_val[feature_columns].to_dict(orient = 'records')
test_dict = df_test[feature_columns].to_dict(orient = 'records')

dv = DictVectorizer(sparse = False)

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

print("training model..")
rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, 
                                min_samples_leaf = min_samples_leaf, random_state = 1)
rf.fit(X_train,y_train)

def f1_score_and_auc_score(y_val, y_pred):
    scores = []
    t = 0.5
    light_load_actual_positive = (y_val == 'light_load')
    light_load_actual_negative = (y_val != 'light_load')

    medium_load_actual_positive = (y_val == 'medium_load')
    medium_load_actual_negative = (y_val != 'medium_load')

    maximum_load_actual_positive = (y_val == 'maximum_load')
    maximum_load_actual_negative = (y_val != 'maximum_load')

    light_load_predictive_positive = (y_pred[:,0] >= t)
    light_load_predictive_negative = (y_pred[:,0] < t)

    medium_load_predictive_positive = (y_pred[:,2] >= t)
    medium_load_predictive_negative = (y_pred[:,2] < t)

    maximum_load_predictive_positive = (y_pred[:,1] >=t)
    maximum_load_predictive_negative = (y_pred[:,1] < t)

    l_l_tp = (light_load_predictive_positive & light_load_actual_positive).sum()
    l_l_tn = (light_load_predictive_negative & light_load_actual_negative).sum()
    l_l_fp = (light_load_predictive_positive & light_load_actual_negative).sum()
    l_l_fn = (light_load_predictive_negative & light_load_actual_positive).sum()

    med_l_tp = (medium_load_predictive_positive & medium_load_actual_positive).sum()
    med_l_tn = (medium_load_predictive_negative & medium_load_actual_negative).sum()
    med_l_fp = (medium_load_predictive_positive & medium_load_actual_negative).sum()
    med_l_fn = (medium_load_predictive_negative & medium_load_actual_positive).sum()

    max_l_tp = (maximum_load_predictive_positive & maximum_load_actual_positive).sum()
    max_l_tn = (maximum_load_predictive_negative & maximum_load_actual_negative).sum()
    max_l_fp = (maximum_load_predictive_positive & maximum_load_actual_negative).sum()
    max_l_fn = (maximum_load_predictive_negative & maximum_load_actual_positive).sum()

    l_l_precision = l_l_tp/(l_l_tp + l_l_fp)
    l_l_recall = l_l_tp/(l_l_tp + l_l_fn)
        
    med_l_precision = med_l_tp/(med_l_tp + med_l_fp)
    med_l_recall = med_l_tp/(med_l_tp + med_l_fn)
        
    max_l_precision = max_l_tp/(max_l_tp + max_l_fp)
    max_l_recall = max_l_tp/(max_l_tp + max_l_fn)
        
    l_l_f1_score = (2 * l_l_precision * l_l_recall)/(l_l_precision + l_l_recall)
    med_l_f1_score = (2 * med_l_precision * med_l_recall)/(med_l_precision + med_l_recall)
    max_l_f1_score = (2 * max_l_precision * max_l_recall)/(max_l_precision+ max_l_recall)
    
    light_load_auc = roc_auc_score(light_load_actual_positive, light_load_predictive_positive)
    medium_load_auc = roc_auc_score(medium_load_actual_positive, medium_load_predictive_positive)
    maximum_load_auc = roc_auc_score(maximum_load_actual_positive, maximum_load_predictive_positive)
    
    macro_avg_f1_score = (l_l_f1_score + med_l_f1_score + max_l_f1_score)/3
    macro_avg_auc = (light_load_auc + medium_load_auc + maximum_load_auc)/3
    return macro_avg_f1_score, macro_avg_auc
y_pred = rf.predict_proba(X_val)
macro_avg_f1_score, macro_avg_auc = f1_score_and_auc_score(y_val, y_pred)
print('validation result..')
print('macro_avg_f1_score = %.3f, macro_avg_auc = %.3f' %(macro_avg_f1_score, macro_avg_auc))
feature_columns = list(df.dtypes.index)[:-1]
def train(df_train, y_train, max_depth, n_estimators, min_samples_leaf):
    dicts = df_train[feature_columns].to_dict(orient = 'records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, 
                                min_samples_leaf = min_samples_leaf, random_state = 1)
    rf.fit(X_train, y_train)
    return dv, rf

df_full_train = df_full_train.reset_index(drop = True)
y_full_train = df_full_train.load_type
dv, model = train(df_full_train, y_full_train, max_depth, n_estimators, min_samples_leaf)

with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
print(f'The model is saved to {output_file}')



