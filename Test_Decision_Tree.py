import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
test_id=[0,50,100]

train_target=np.delete(iris.target,test_id)
train_data=np.delete(iris.data,test_id,axis=0)

test_target=iris.target[test_id]
test_data=iris.data[test_id]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(train_data,train_target)

print("The Original Value is:")
print(test_target)
print("The Predicted value is")
print(clf.predict(test_data))
