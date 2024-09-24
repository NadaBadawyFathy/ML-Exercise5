import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('HR_comma_sep.csv')

# print(df.head())

left1 = df[df.left ==1]
print(left1.shape)

left0 = df[df.left ==0]
print(left0.shape)

# medium = df.groupby('left').mean()

pd.crosstab(df.salary ,df.left).plot(kind='bar')
pd.crosstab(df.Department ,df.left).plot(kind='bar')

# plt.show()

allDataAn= df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salaryDummies = pd.get_dummies(allDataAn.salary, prefix='Salary')

concatData = pd.concat([allDataAn,salaryDummies] , axis='columns')

x = concatData.drop(['salary'],axis='columns')

y = df['left']

X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.3)

model = LogisticRegression()

model.fit(X_train,y_train)

print(model.predict(X_test))
print(model.score(X_test,y_test))
