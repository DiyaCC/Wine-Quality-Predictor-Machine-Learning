import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

############################################################################
#reading dataset

data = pd.read_csv('winequality-red.csv')

features = list(data.drop(['quality'], axis = 1).columns)

############################################################################
# Classes of target/y-values

classes = set(data.quality)

############################################################################
# DESCRIBING THE DATASET. ARE THERE NULL VALUES?

print(data.describe())
print(data.info())
print(data.isnull().sum())


############################################################################
# MAKING A CORRELATION MATRIX WITH ALL FEATURES IN DATASET

plt.figure(figsize = (50, 75))
corr = data.corr()
sb.heatmap(corr, annot = True)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.show()

# there are no highly correlated features. There is no need to remove a specific feature.

# DETECTING OUTLIERS USING Z-SCORE

from scipy import stats

z = np.abs(stats.zscore(data))

# remove those records whose z > 3 (outlier)

new_data = data[(z < 3).all(axis = 1 )]

# SPLITTING DATA INTO FEATURES (X) AND TARGET (Y)

features = new_data.drop(['quality'], axis = 1, inplace = False)
target = new_data['quality'].values


# SPLITTING TRAINING VERSUS TESTING DATA

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 22)

    
# DATA MODELING USING RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

rf_classificationModel = RandomForestClassifier(n_estimators = 100)

rf_classificationModel.fit(x_train, y_train)

# MAKE PREDICTIONS USING RANDOM FOREST MODEL

y_pred = rf_classificationModel.predict(x_test)

predictionTable = pd.DataFrame()

predictionTable['Actual Y-Values'] = y_test
predictionTable['Predicted Y-Values'] = y_pred

print(predictionTable)


# DATA EVALUATION

from sklearn import metrics

print('Accuracy Score' , metrics.accuracy_score(y_test, y_pred))



