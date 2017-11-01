from sklearn import datasets
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.5)

from sklearn.neighbors import KNeighborsClassifier
sarthak=KNeighborsClassifier()

sarthak.fit(x_train,y_train)

predictions=sarthak.predict(x_test)

print (accuracy_score(y_test,predictions))
