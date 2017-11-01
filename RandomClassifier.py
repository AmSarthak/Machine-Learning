import random

class RandomClassifier():
    def fit(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
    def predict(self, x_test):
        predictions=[]
        for row in x_test:
            label=random.choice(self.y_train)
            predictions.append(label)
        return predictions

from sklearn import datasets
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.5)

#from sklearn import tree
sarthak=RandomClassifier()

sarthak.fit(x_train,y_train)

predictions=sarthak.predict(x_test)

print("The Accuracy Score is:")
print (accuracy_score(y_test,predictions))         #Answer must be near to 33% since 1/3=0.33
