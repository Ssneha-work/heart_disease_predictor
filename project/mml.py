#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv("heart.csv")

#feature and target variables
X=df.iloc[:,:13].values
Y=df.iloc[:,13].values


model=LogisticRegression(max_iter=1000)

#fitting model with training data
model.fit(X,Y)

# Saving model to current directory
pickle.dump(model, open('model.pkl','wb'))
