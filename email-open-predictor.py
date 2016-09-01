"""
Created on Tue Aug 30 17:14:05 2016

@author: Eric Leung

Email open predictor.  Given a set of historic email stats, 
predict if an unseen set of emails will be opened by the recipients.
"""

import numpy as np
import pandas as pd
import re

# Helper functions

# convert true/false to 1/0
def convertBool(x):
    if x == "true":
        return 1
    else:
        return 0

# convert user_id and mail_id to a hash value
def convertId(x):
    return hash(x)

# extract the last digits in a string
# used for columns mail_category and mail_type
# e.g. mail_category_1 becomes 1, mail_type_5 becomes 5, etc
def convertMailType(x):
    return getLastDigitsFromString(x)

def convertMailCategory(x):
    return getLastDigitsFromString(x)

def getLastDigitsFromString(x):
    result = re.match('.*?([0-9]+)$', x)
    if result is None:
        return -9999
    else:
        return int(result.group(1))
########

# Read training data set
f = open("./hackerrank-predict-email-opens-dataset/training_dataset.csv")
df = pd.read_csv(f,
                 converters={"opened": convertBool,
                             "unsubscribed": convertBool,
                             "clicked": convertBool,
                             "hacker_confirmation": convertBool,
                             "mail_type": convertMailType,
                             "mail_category": convertMailCategory,
                             "user_id": convertId,
                             "mail_id": convertId})
f.close
df = df.fillna(value=-9999)
print('DF size before dropping invalid records='+str(len(df)))
df = df.drop(df[np.logical_and(df['opened'] != 1, df['clicked'] == 1)].index)
print('DF size after dropping invalid records='+str(len(df)))
#df = df[np.logical_and(df['opened'] != 0, df['clicked'] != 1)]

# Examine positive and negative samples to see if there is class imbalance
X_positive = df.loc[df['opened'] == 1]
X_negative = df.loc[df['opened'] != 1]
print('Original positive sample size=' + str(len(X_positive))) #161347
print('Original negative sample size=' + str(len(X_negative))) #324701

# There are twice as many negatives samples as positive ones, cut half of the negatives
X_negative_half = X_negative[:161347] #136179

df = pd.concat([X_positive, X_negative_half])
print('Tweaked positive sample size=' + str(len(df.loc[df['opened'] == 1]))) #161347
print('Tweaked negative sample size=' + str(len(df.loc[df['opened'] != 1]))) #324701

#Shuffle
#df.reindex(np.random.permutation(df.index))

# ignorin mail_cateory and mail_type improved accuracy
X = df.drop([
                    'user_id',
                    'mail_id',
                    'click_time', 
                    'clicked', 
                    'open_time', 
                    'opened', 
                    'unsubscribe_time', 
                    'unsubscribed',
                    'mail_category', 'mail_type',
                    'sent_time'], axis=1)
y = df['opened']

# normalize data
#from sklearn import preprocessing
#X = preprocessing.scale(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .05)

# create classifier, at this point it is just an empty box of rules
from sklearn import tree
clf = tree.DecisionTreeClassifier(class_weight="balanced")

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm="ball_tree")

# train the classifier, fit() will find pattern in data
clf = clf.fit(X_train, y_train) 

print('prediction=')
y_pred = clf.predict(X_test)
print(y_pred)
print('actual=')
print(y_test.values)

from sklearn.metrics import f1_score, accuracy_score
print('f1_score='+str(f1_score(y_test, y_pred)))  
print('accuracy='+str(accuracy_score(y_test, y_pred)))

# test set
f = open("./hackerrank-predict-email-opens-dataset/test_dataset.csv")
testDf = pd.read_csv(f,
                 converters={"opened": convertBool,
                             "unsubscribed": convertBool,
                             "clicked": convertBool,
                             "hacker_confirmation": convertBool,
                             "mail_type": convertMailType,
                             "mail_category": convertMailCategory,
                             "user_id": convertId,
                             "mail_id": convertId},)
f.close
testDf = testDf.fillna(value=-9999)

X2 = testDf.drop([
                  'user_id',
                  'mail_id',
                  'mail_category', 'mail_type',
                    'sent_time'], axis=1)

final_pred = clf.predict(X2)
print('final predictions')
print(final_pred)
np.savetxt('prediction.csv',final_pred, fmt='%i')