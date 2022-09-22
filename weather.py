weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes']

from heapq import merge
from sklearn import preprocessing
#creating labelEncoder
le=preprocessing.LabelEncoder()
#converting string labels into numbers
weather_encoded=le.fit_transform(weather)
print("weather:",weather_encoded)
#Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Temp:",temp_encoded)
print("Play:",label)
#combining weather and temp into single list of tuples
features=list(zip(weather_encoded,temp_encoded))
print(features)
#import Guaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(features,label)
predicted=model.predict([[0,2]])
print("Predicted Value:",predicted) 
