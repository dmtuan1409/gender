from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('../Data/gender.csv')
df = df[0:len(df)]
df_feature = df['Name']
df_sex = df['sex']
cv = CountVectorizer()
X = cv.fit_transform(df_feature)
print(cv.get_feature_names())
x_train, x_test, y_train, y_test = train_test_split(X, df_sex, test_size=0.25, random_state=50)
NB = MultinomialNB()
NB.fit(x_train, y_train)
result = NB.predict(x_test)
print(accuracy_score(y_test, result))
