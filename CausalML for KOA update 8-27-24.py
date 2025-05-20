#pip install --upgrade pip --user
#pip install --upgrade causalml --user
#pip install --upgrade numpy --user
#pip install --upgrade scipy --user
#pip install Numpy==1.23.5 --user
#pip install --upgrade statsmodels --user
#pip install MissForest --user
#pip install missingpy --user
#pip install missforest --user
#pip install scikit-learn --user

#from missforest.missforest import MissForest

#pip install causalml==0.14.1
#pip install causalml --upgrade
#pip freeze | findstr causalml
import causalml

#the following dependencies copied from: https://github.com/uber/causalml/blob/master/examples/meta_learners_with_synthetic_data_multiple_treatment.ipynb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

# Visualization
import seaborn as sns

# from causalml.inference.meta import XGBTLearner, MLPTLearner
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from causalml.inference.meta import LRSRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *

#from causalml.dataset import make_uplift_classification
from causalml.metrics import plot_gain
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

from sklearn.model_selection import train_test_split
import causalml
#causalml.__version__

from IPython.display import Image

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# imports from package
import logging
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from copy import deepcopy

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

%matplotlib inline

#####################
# Multiple Trx Case #
#####################

# example provided here: https://causalml.readthedocs.io/en/latest/examples/meta_learners_with_synthetic_data_multiple_treatment.html

import pandas as pd
from pandas import read_csv
import os
import numpy as np
from pandasql import *

# changing working directory
os.chdir('//v06.med.va.gov/DUR/HSRD/HTE_KOA_IRB_1725969/8_Statistician (sensitive data)/Statistician/Data/')

# importing data
group_df = pd.read_csv('KOAHTE_dataset.csv')

#create new three-level variable for intervention type
group_df['treatment'] = np.where(group_df['GroupPTArm']==0,"IndividualPT", 
                          np.where(group_df['GroupPTArm']==1,"GroupPT",
                                   np.where(group_df['SKOAArm']==0,"Control",
                                            np.where(group_df['SKOAArm']==1,"SKOA",None))))

group_df['WomacTotal_Delta'] = group_df['use_WomacTotal_Post']-group_df['use_WomacTotal_Pre']

treatment_name = ['treatment']
y_name = ['WomacTotal_Delta']
X_names = ['YearsWithArthritis_Pre','SelfRatedHealth_Pre','BMI_Pre','Age_Pre',
             'Education_Pre','WorkStatus_Pre','Gender_Pre','Ethnicity_Pre',
             'White1Nonwhite0KOAHTE_Pre','OpioidMed_Pre','NonOpioidPainMed_Pre',
             'AssistiveDevBrace_Pre','AssistiveDevCaneStick_Pre','StrengthPASECHAMPS_Pre',
             'ComorbidityHeartDisease_Pre','ComorbidityHighBloodPressure_Pre',
             'ComorbidityDiabetes_Pre','ComorbidityDepression_Pre',
             'ComorbidityBackPain_Pre','SelfEfficacyExerciseTotal_Pre',
             'TotalJointsArthritis_Pre','LowerBodyJointsArthritis_Pre',
             'LowBackArthritis_Pre','Bilateral1Unilateral0KOA_Pre','use_WomacTotal_Pre']

# dropping 'ComorbidityOA_Pre' because no variability
vars = treatment_name + y_name + X_names

#dropping missing values from treatment group
complete_df = group_df.dropna(subset=vars)

temp = complete_df[vars]

#creating arrays for analyses
treatment = complete_df[treatment_name].to_numpy()
treatment.shape = (621,)
y = complete_df[y_name].to_numpy()
y.shape = (621,)
X = complete_df[X_names].to_numpy()

IDs = complete_df['STUDYID'].to_numpy()

############################
# Split into Training/Test #
############################

# define RMSE function
def rmse(predictions, targets):
    return round(np.sqrt(((predictions - targets) ** 2).mean()),2)

# define MAPE function
def smape(predictions, targets):
    targets, predictions = np.array(targets), np.array(predictions)
    smape = 100*np.mean(np.abs((targets - predictions)/
                                 ((np.abs(targets)+np.abs(predictions))/2)))
    return round(smape,2)

# define MAE function
def mae(predictions, targets):
    targets, predictions = np.array(targets), np.array(predictions)
    mae = np.mean(np.abs((targets - predictions)))
    return round(mae,2)

# Create dataframe for holding predictions
yhat = pd.DataFrame()
yhat['treatment'] = treatment
yhat['observed'] = y

# Create dataframe for holding prediction summaries
model_summaries = pd.DataFrame()

baselearners = ['S','T','X']
algos = ['XGB','lightGB','RF']

coln=0
counter=2

#setting up cross-validation
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234)

for i in baselearners: 
    
    rown=0
    
    for j in algos:
        
        if j=='XGB':
            algo = XGBRegressor()
        elif j=='lightGB':
            algo = LGBMRegressor()
        else:
            algo = RandomForestRegressor()

        if i=='S':
            learner = BaseSRegressor(algo, control_name='Control')
        elif i=='T':
            learner = BaseTRegressor(algo, control_name='Control')
        elif i=='X':
            learner = BaseXRegressor(algo, control_name='Control')

        #creating arrays to store results from repeated k folds
        treatment_preds = pd.DataFrame()

        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            t_train, t_test = treatment[train_index], treatment[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ID_train, ID_test = IDs[train_index], IDs[test_index]
            
            # fit model on training set
            learner.fit(X=X_train, treatment=t_train, y=y_train)
            
            # Predict the TE for test, and request the components (predictions for each of the 3 treatments (relative to control))
            te_test_preds, yhat_c, yhat_t = learner.predict(X_test, t_test, return_components=True)
            
            # set yhat values, read in observed and predicted values
            # predicted control values are all the same for each treatment
    
            output = pd.DataFrame(zip(ID_test,t_test,y_test), columns=['STUDYIDS','treatment','observed'])
               
            output['pred'] = np.where(output['treatment'] == 'GroupPT', list(yhat_t.items())[0][1], 
                np.where(output['treatment']=='IndividualPT', list(yhat_t.items())[1][1],
                     np.where(output['treatment']=='SKOA',list(yhat_t.items())[2][1],np.nan))) 

            treatment_preds = treatment_preds.append(output)
            
        # now need to aggregate across repetitions and folds
        yhat = sqldf('select STUDYIDS, treatment, observed, avg(pred) as pred from treatment_preds group by STUDYIDS, treatment, observed')
        
        yhat2 = yhat.dropna()
        
        varname = 'predicted_'+i+'_'+j
        
        model_summaries.loc[rown,coln] = str(rmse(yhat2.iloc[:,3],yhat2.iloc[:,2])) + "/" + str(mae(yhat2.iloc[:,3],yhat2.iloc[:,2]))
        
        counter = counter+1
        rown= rown+1
        
    coln = coln+1

model_summaries.columns =['S','T','X']
model_summaries.index = ['XG Boost', 'Light GB', 'Random Forest']

# S-learner with random forest implementation fits the best #

# fit model on full set
learner = BaseSRegressor(RandomForestRegressor(), control_name='Control')
learner.fit(X=X, treatment=treatment, y=y)

pred_ML = learner.predict(X_test, t_test, return_components=True)
pred_ML_control = pd.DataFrame.from_dict(pred_ML[1])
pred_ML_trx = pd.DataFrame.from_dict(pred_ML[2])

# estimate ATE
learner.estimate_ate(X, treatment, y, return_ci=True)

# feature importance
learner_tau = learner.fit_predict(X, treatment, y)
learner.get_importance(X=X, 
                       tau=learner_tau, 
                       normalize=True, 
                       method='auto',
                       features=X_names)

# Shapley values
shap_learner = learner.get_shap_values(X=X, tau=learner_tau)
learner.plot_shap_values(X=X, tau=learner_tau, features=X_names)

summary(complete_df['WomacTotal_Delta'])

#####################
# Train Uplift Tree #
#####################

from causalml.inference.tree import CausalRandomForestRegressor, CausalTreeRegressor
from causalml.dataset import make_uplift_classification
from causalml.metrics import plot_gain
from causalml.inference.tree import UpliftTreeClassifier


clf = UpliftTreeClassifier(max_depth=3, min_samples_leaf=10, #evaluationFunction='KL',
                           control_name='Control')
clf.fit(X_train, t_train, y_train)

#clf.fit(X, treatment, y)


pred = clf.predict(X_test)

df_res = pd.DataFrame(pred, columns=clf.classes_)
df_res.head()

result = uplift_tree_string(clf.fitted_uplift_tree, X_names)
