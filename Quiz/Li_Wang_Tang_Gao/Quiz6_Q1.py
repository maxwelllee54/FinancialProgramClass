
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.datasets import load_iris
import time
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import svm

iris=load_iris()
iris.data[iris.target!=2]
iris.target[iris.target!=2]

df = pd.DataFrame(iris.data[iris.target!=2], iris.target[iris.target!=2], columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
df.to_csv('iris_binary_data.csv')

X = iris.data[iris.target!=2][:, :2]  # we only take the first two features. We could                                     # avoid this ugly slicing by using a two-dim dataset
y = iris.target[iris.target!=2]

h = .02  # step size in the mesh

svc = svm.SVC(kernel='linear').fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X, y)
sigmoid_svc=svm.SVC(kernel='sigmoid').fit(X,y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with sigmoid',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, sigmoid_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()


# In[7]:

from sklearn.model_selection import train_test_split
from sklearn import svm

data=pd.read_csv('/Users/tangyang1176/iris_binary_data.csv')
datax=data[[1,2,3,4]]
datay=data[[0]]
test_percent=0.45
training_data,test_data,training_data_label,test_data_label = train_test_split(datax,datay,test_size=test_percent,random_state=42) 

svm_learning_machine_rbf=svm.SVC(kernel='rbf', gamma=0.7)
svm_learning_machine_rbf.fit(training_data,training_data_label)

svm_learning_machine_linear=svm.SVC(kernel='linear')
svm_learning_machine_linear.fit(training_data,training_data_label)

svm_learning_machine_poly=svm.SVC(kernel='poly', degree=3)
svm_learning_machine_poly.fit(training_data,training_data_label)

svm_learning_machine_sigmoid=svm.SVC(kernel='sigmoid')
svm_learning_machine_sigmoid.fit(training_data,training_data_label)

svm_predict_rbf=svm_learning_machine_rbf.predict(test_data)
svm_predict_linear=svm_learning_machine_linear.predict(test_data)
svm_predict_poly=svm_learning_machine_poly.predict(test_data)
svm_predict_sigmoid=svm_learning_machine_sigmoid.predict(test_data)
print("\n the result with rbf kernel" )
print(svm_predict_rbf)
print("\n the result with linear kernel" )
print(svm_predict_linear)
print("\n the result with poly kernel" )
print(svm_predict_poly)
print("\n the result with sigmoid kernel" )
print(svm_predict_sigmoid)


# In[8]:

import math
import numpy as np
def compute_measure(predicted_label, true_label):
    t_idx=(predicted_label == true_label)
    f_idx=np.logical_not(t_idx)
    
    p_idx=(true_label >0)
    n_idx=np.logical_not(p_idx)
    
    tp=np.sum(np.logical_and(t_idx,p_idx))
    tn=np.sum(np.logical_and(t_idx,n_idx))
    
    fp=np.sum(n_idx)-tn
    fn=np.sum(p_idx)-tp
    
    tp_fp_tn_fn_list=[]
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list=np.array(tp_fp_tn_fn_list)
    
    tp=tp_fp_tn_fn_list[0]
    fp=tp_fp_tn_fn_list[1]
    tn=tp_fp_tn_fn_list[2]
    fn=tp_fp_tn_fn_list[3]
    
    with np.errstate(divide='ignore'):
        sen= (1.0 *tp)/(tp+fn)
    with np.errstate(divide='ignore'):
        spc= (1.0 *tn)/(tn+fp)
    with np.errstate(divide='ignore'):
        ppr= (1.0 *tp)/(tp+fp)
    with np.errstate(divide='ignore'):
        npr= (1.0 *tn)/(tn+fn)
        
    acc=(tp+tn)*1.0/(tp+fp+tn+fn)
    ans=[]
    ans.append(acc)
    ans.append(sen)
    ans.append(spc)
    ans.append(ppr)
    ans.append(npr)
    
    return ans


# In[9]:

predicted_label=svm_predict_rbf
clm_list = []
for i in test_data_label:
    clm_list.append(i)
true_label = test_data_label[clm_list[0]].values
ans1=compute_measure(predicted_label, true_label)
print("\n check the following classication measures: accuracy, sen, spec, ppr, npr")
print("\n with rbr kernel")
print("{}".format(ans1))

predicted_label=svm_predict_linear
ans2=compute_measure(predicted_label, true_label)
print("\n with linear kernel")
print("{}".format(ans2))

predicted_label=svm_predict_poly
ans3=compute_measure(predicted_label, true_label)
print("\n with poly kernel ")
print("{}".format(ans3))

predicted_label=svm_predict_sigmoid
ans4=compute_measure(predicted_label, true_label)
print("\n with sigmoid kernel ")
print("{}".format(ans4))
#m


# In[5]:

#muiltiple classification

import numpy as np
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris=datasets.load_iris()
data= iris.data 
label=iris.target
test_percent=0.45
training_data, test_data, training_data_label, test_data_label = train_test_split(data, label, test_size=test_percent, random_state=42)

svm_mul_rbf=svm.SVC(decision_function_shape='ovo',kernel='rbf')
svm_mul_rbf.fit(training_data, training_data_label)

svm_mul_linear=svm.SVC(decision_function_shape='ovo',kernel='linear')
svm_mul_linear.fit(training_data, training_data_label)

svm_mul_poly=svm.SVC(decision_function_shape='ovo',kernel='poly')
svm_mul_poly.fit(training_data, training_data_label)

svm_mul_sigmoid=svm.SVC(decision_function_shape='ovo',kernel='sigmoid')
svm_mul_sigmoid.fit(training_data, training_data_label)

svm_mul_predict_rbf=svm_mul_rbf.predict(test_data)
svm_mul_predict_linear=svm_mul_linear.predict(test_data)
svm_mul_predict_poly=svm_mul_poly.predict(test_data)
svm_mul_predict_sigmoid=svm_mul_sigmoid.predict(test_data)

print("\n the multi-classification result with rbf kernel" )
print(svm_mul_predict_rbf)
print("\n the multi-classification result with linear kernel" )
print(svm_mul_predict_linear)
print("\n the multi-classification result with poly kernel" )
print(svm_mul_predict_poly)
print("\n the multi-classification result with sigmoid kernel" )
print(svm_mul_predict_sigmoid)


# In[6]:

predicted_label=svm_mul_predict_rbf
true_label = test_data_label
ans5=compute_measure(predicted_label, true_label)
print("\n check the following multi-classication measures: accuracy, sen, spec, ppr, npr")
print("\n with rbr kernel")
print("{}".format(ans5))

predicted_label=svm_mul_predict_linear
ans6=compute_measure(predicted_label, true_label)
print("\n with linear kernel")
print("{}".format(ans6))

predicted_label=svm_mul_predict_poly
ans7=compute_measure(predicted_label, true_label)
print("\n with poly kernel ")
print("{}".format(ans7))

predicted_label=svm_mul_predict_sigmoid
ans8=compute_measure(predicted_label, true_label)
print("\n with sigmoid kernel ")
print("{}".format(ans8))

