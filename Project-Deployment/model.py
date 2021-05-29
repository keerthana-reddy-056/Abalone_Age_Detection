import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('abalone.csv')

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
data['Sex'] = labelencoder_y.fit_transform(data['Sex'])

x=data.iloc[:,0:8]
y=data.iloc[:,-1]


# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.tree import DecisionTreeRegressor

#Create the Decision Tree regressor object 
regressor = DecisionTreeRegressor(random_state=0)

#Fit the regressor object to the dataset.
regressor.fit(x_train,y_train)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,0.455,0.365,0.095,0.5140,0.2245,0.1010,0.1500]]))