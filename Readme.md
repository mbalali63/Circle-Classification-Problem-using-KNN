# Project 1 -Circle Classification Problem using KNN #

## Abstract ##

This project investigates how different parameters affect model accuracy in a geometric classification task. This is a solid foundational project that can yield interesting insights about model behavior on simple geometric problems. The problem is a simple circle inside an square. the task is to create a model that can predict if any point inside the square is inside the circle or not.
The results show that the number of neighbors have an inverse effect on the accuracy, recall anf f1 score, but increase the precision.
The minkowski (and Euclidian) methods of calculating the distance are the best and cosine method is the worst, among those studied here. We have not found any considerable effect by the chosen algorithm on the performance of the model.
Finally,the project revealed unexpected relationships between parameters and metrics, prompting further study.

# Table of Contents
1. [Introduction](#Introduction)
2. [Methodology](#Methodology)
   - 2.1 [Data Generation](#Data-Generation)
   - 2.2 [ML Model](#ML-Model)
   - 2.3 [Parameter Study](#Parameter-Study)
   - 2.4 [Evaluation Metrics](#Evaluation-Metrics)
3. [Results](#Results) 
   - 3.1 [Effect of Number of neighbors](#Effect-of-Number-of-neighbors) 
   - 3.2 [Distance estimation method](#Distance-estimation-method) 
   - 3.3 [The algorithm](#The-algorithm) 
   - 3.4 [The weights](#The-weights)
4. [Discussion](#Discussion)
5. [References](#References)


## Introduction ##

KNN is one of the most famous machine learning models, used in classification problems. The main working principle of this method is to check the class according to the calculated distance of each data point, with the training labeled points, and decide about its class. 
<br/>
In this project we have implemented a simple KNN model and tyied to evaluate the effect of different parameters on it. The parameters, we have studied are the number of neighobrs, the method of calculating distance, the weighting method and the main algorithm. 




## Methodology ##

### Data Generation ###

To have a simple, controllable problem for investigating the effect of different parameters on the model performance, we have created a hypothetical model and do the training and test of the KNN models based on that. The model is a circle inside a square, and the data points are (x,y) points inside the square, that could be inside or outside the circle. The points are classified if they are inside the circle (labeled 1) or outside (labeled 0). For this reason the following code snippet will create cirlce inside the square bounds, and also create a bunch of points inside it, which are generated randomlly, and could be inside or outside the circle. <br>


```python
import numpy as np
import matplotlib.pyplot as plt


def generate_points(radii,x_bounds,y_bounds,number_of_points):   
    np.random.seed(42)
    number_of_points = number_of_points
    x_min_bound, x_max_bound = x_bounds[:]
    y_min_bound, y_max_bound = y_bounds[:]

    x_points = np.random.uniform(x_min_bound,x_max_bound,number_of_points)
    y_points = np.random.uniform(y_min_bound,y_max_bound,number_of_points)
    return [x_points,y_points]

def plot_points(points):
    x_points = points[0]
    y_points = points[1]

    theta = np.linspace(0,2*np.pi,100);
    radii = 5;
    x = radii * np.cos(theta);
    y = radii * np.sin(theta);
    
    plt.figure(figsize=(6,6));
    plt.plot(x,y,'b-',linewidth=2);
    plt.scatter(x_points,y_points,color='red',s=50,label="Points")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()



def plot_points2(points):
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    
    theta = np.linspace(0,2*np.pi,100);
    radii = 5;
    x = radii * np.cos(theta);
    y = radii * np.sin(theta);
    
    plt.figure(figsize=(6,6));
    plt.plot(x,y,'b-',linewidth=2);
    plt.scatter(x_points,y_points,color='red',s=50,label="Points")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


radii = 5.0
x_bounds = [-10,10]
y_bounds = [-10,10]
number_of_points = 300
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
plot_points([x_points,y_points])
```


    
![png](output_9_0.png)
    


After generating the points in the domain, they must be labeled for if they are inside or outside the circle. For this reason, we have defined a function called is_inside with the following deinition:


```python
def is_inside(x0,y0,r0):
    return x0**2.+y0**2 <= radii**2
```

This function returns true if the point is inside or on the circle, and false if it is outside the circle. Then the following code snippet will loop through all of the generated points, and label them with 1, if they are inside and 0 if they are outside the circle:


```python
def labeling(xypoints,radii):
    x_points = xypoints[0]
    y_points = xypoints[1]
    Points = []
    Labels = []
    for index, x0 in enumerate(x_points):
        y0 = y_points[index]
        Points.append((x0,y0))
        if (is_inside(x0,y0,radii)):
            Labels.append(1)
        else:
            Labels.append(0)
    return [Points,Labels]

[Points,Labels] = labeling([x_points,y_points],radii)
```

### ML Model ###

Now, the points are all labeled, we have to prepare the machine learning model. We will use the scikit-learn[1] package functionalities to prepare the model and data. For this reason, the generated data must be splitted into train and test groups. Then, we will use KNeighborsClassifier model, as it is the target of this project. Here you can find a code snippet, which implements these, with default parameters:


```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)
clf = KNeighborsClassifier(n_neighbors=4,weights='uniform',algorithm='auto')
clf.fit(X_train,y_train)
print("Test set accuracy = {}".format(clf.score(X_test,y_test)))
```

    Test set accuracy = 0.9333333333333333
    

As you may consider the accuracy of the model on the test data is 92%. 

### Parameter Study ###

In this project, we will check and evaluate the effect of different parameters on the model performance. The parameters to be studied are as follows:
<ul>
    <li>Number of neighbors</li>
    <li>Distance estimation method</li>
    <li>The algorithm</li>
    <li>The weights</li>
    <li>Number of training points</li>
</ul>

### Evaluation Metrics ###

To have a quantitative metric for evaluation, we will use the following parameters:
<ul>
    <li>Accuracy</li>
    <li>Precision</li>
    <li>Recall</li>
    <li>F1-Score</li>
</ul>

## Results ##

### Effect of Number of neighbors ###

To evaluate the effect of number of neighbors on the model performance, we will keep all of the other parameters as their default values, and only change this parameter. 


```python
n_neighbors_list = np.unique(np.linspace(1, 75, 75).astype(int))
scores_list = []
for n in n_neighbors_list:
    clf1 = KNeighborsClassifier(n_neighbors=n)
    clf1.fit(X_train,y_train)
    scores_list.append(clf1.score(X_test,y_test))

```

So, the scores are:


```python
plt.plot(n_neighbors_list,scores_list)
plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.show()
```


    
![png](output_27_0.png)
    


According to the above chart, the accuracy will decrease if the number of neighbors increase too much. It will be as low as 80%. The maximum value for accuracy took place at n=1. We can also find few local maximums in the curve.
This could be highly influenced by the number of points, used for training. so the following code, will change the number of generated points, and then create the model and compare the effect of number of neighbors.



```python


for number_of_points in [100,400,800,1200]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors_list = np.unique(np.linspace(1, int(0.25*number_of_points),int(0.25*number_of_points)).astype(int))
    scores_list = []
    for n in n_neighbors_list:
        clf1 = KNeighborsClassifier(n_neighbors=n)
        clf1.fit(X_train,y_train)
        scores_list.append(clf1.score(X_test,y_test))
    plt.plot(n_neighbors_list,scores_list,label=f'N={number_of_points}')
plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.legend(title="Number of points",bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_29_0.png)
    


According to the above chart, increasing the number of points, do not affect the general behaviour fo the accuracy-n curve. in the other words, in all of the curves, the general trend is downward, with some local maximums. But, the range of variation, is increased by increasing the number of points. The following curve shows the Accuracy-N variation for only n=1200


```python
for number_of_points in [1200]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors_list = np.unique(np.linspace(1, int(0.25*number_of_points),int(0.25*number_of_points)).astype(int))
    scores_list = []
    for n in n_neighbors_list:
        clf1 = KNeighborsClassifier(n_neighbors=n)
        clf1.fit(X_train,y_train)
        scores_list.append(clf1.score(X_test,y_test))
    plt.plot(n_neighbors_list,scores_list,label=f'N={number_of_points}')
plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.legend(title="Number of points",bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_31_0.png)
    


The maximum value is:


```python
max0 = max(scores_list)
max_index = scores_list.index(max0)
max_index
```




    2



After evaluating the effect of number of neighbors on the accuracy, we have to find out more about its effect on other model performance parameters, such as precision, Recall, and F1-score. So, the following code will do it:


```python
from sklearn.metrics import confusion_matrix


def calc_metrics(tn, fp, fn, tp,parameter):
    if (parameter == 'accuracy'):
        return (tn+tp)/(tn+tp+fp+fn);
    elif (parameter == 'precision'):
        if (tp+fp != 0):
            return tp/(tp+fp);
        else:
            return 0;
    elif (parameter == 'recall'):
        return tp/(tp+fn);
    elif (parameter == 'f1'):
        precision = (tn+tp)/(tn+tp+fp+fn);
        recall = tp/(tp+fn);
        return (2*(precision*recall)/(precision+recall));
    return 0;
        
    

for number_of_points in [1000]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors_list = np.unique(np.linspace(1, int(0.1*number_of_points),int(0.1*number_of_points)).astype(int))
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for n in n_neighbors_list:
        clf1 = KNeighborsClassifier(n_neighbors=n)
        clf1.fit(X_train,y_train)
        tn, fp, fn, tp = confusion_matrix(y_test,clf1.predict(X_test)).ravel()
        accuracy.append(calc_metrics(tn, fp, fn, tp,'accuracy'))
        precision.append(calc_metrics(tn, fp, fn, tp,'precision'))
        recall.append(calc_metrics(tn, fp, fn, tp,'recall'))
        f1.append(calc_metrics(tn, fp, fn, tp,'f1'))
    plt.plot(n_neighbors_list,accuracy,label=f'Accuracy')
    plt.plot(n_neighbors_list,precision,label=f'Precision')
    plt.plot(n_neighbors_list,recall,label=f'Recall')
    plt.plot(n_neighbors_list,f1,label=f'F1')

plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_35_0.png)
    


Surprisingly, all other performance parameters, decrease with increasing the number of points for this problem, except percision, which is an upward curce. the following chart shows the precision curve, only.


```python
plt.plot(n_neighbors_list,precision,label=f'Precision')
plt.xlabel('number of neighbors')
plt.ylabel('Precision')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_37_0.png)
    


So, by increasing the number of neighobrs, we will have a much higher precision of predicting, in the other words, if the model predict the point is inside the circle, it is (with high probability, because the precision is high). but, the recall and accuracy falls down, for higher number of points. So, it means that altough the points which are guessed to be true, are correctly inside the circle (for higher number of points), but the points which are wrongly predicted to be outside the circle are also increased in higher number of points.  

Now, we can also check if the total number of points has a major effect on this result:


```python
from sklearn.metrics import confusion_matrix

for number_of_points in [600,1200]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors_list = np.unique(np.linspace(1, int(0.1*number_of_points),int(0.1*number_of_points)).astype(int))
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for n in n_neighbors_list:
        clf1 = KNeighborsClassifier(n_neighbors=n)
        clf1.fit(X_train,y_train)
        tn, fp, fn, tp = confusion_matrix(y_test,clf1.predict(X_test)).ravel()
        accuracy.append(calc_metrics(tn, fp, fn, tp,'accuracy'))
        precision.append(calc_metrics(tn, fp, fn, tp,'precision'))
        recall.append(calc_metrics(tn, fp, fn, tp,'recall'))
        f1.append(calc_metrics(tn, fp, fn, tp,'f1'))
    plt.plot(n_neighbors_list,precision,label=f'N={number_of_points}')

plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.legend(title="number of points",bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_40_0.png)
    


So, the general trend is confirmed.

### Distance estimation method ###

There are different methods available for distance estimation in KNN model. Some of the most important and famous methods are:
<ul>
    <li>minkowski (with p=3)</li>
    <li>euclidean</li>
    <li>cityblock</li>
    <li>cosine</li>
</ul>
We will check the effect of choosing any of the mentioned methods on the performance of our KNN model.


```python
import seaborn as sns

for number_of_points in [300]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)

    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors = 2
    methods_list = ["minkowski","euclidean","cityblock","cosine"]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for method in methods_list:
        if method == 'minkowski':
            clf1 = KNeighborsClassifier(n_neighbors=n_neighbors,metric=method,p=3)
        else:
            clf1 = KNeighborsClassifier(n_neighbors=n_neighbors,metric=method)
        clf1.fit(X_train,y_train)
        tn, fp, fn, tp = confusion_matrix(y_test,clf1.predict(X_test)).ravel()
        accuracy.append(calc_metrics(tn, fp, fn, tp,'accuracy'))
        precision.append(calc_metrics(tn, fp, fn, tp,'precision'))
        recall.append(calc_metrics(tn, fp, fn, tp,'recall'))
        f1.append(calc_metrics(tn, fp, fn, tp,'f1'))


metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
methods = ["minkowski", "euclidean", "cityblock", "cosine"]
data = np.array([accuracy, precision, recall, f1])

fig, ax = plt.subplots()
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=methods, yticklabels=metrics_names)
ax.set_title('Performance Metrics by Distance Metric')
plt.tight_layout()
plt.show()
```


    
![png](output_44_0.png)
    


According to this chart, the minkowski method, shows the best results, and cosie method the worst. The reason of poor performace for cosine metric, relates to its definition. Actuall, the cosine metric is something related to the cosine of the angle between two vecotrs. In the other words, this will check the cosine of the angle between two vectors, which are drawn from the origin to each point. So, it is not providing any information about the actual distance of the points. or, in the other words, two points on the same line, may be considered too close, while tow points inside the circle but having 40degree aparat, may be assumed not in a same domain. So, the poor performace is expected. and this metric is not suitable for such problems. cosine method is useful for:
<ul>
    <li>The magnitude of the vectors is not important, but their direction matters.</li>
    <li>The dataset consists of text data (e.g., TF-IDF representations of words).</li>
    <li>The data is high-dimensional and sparse (e.g., document classification, recommendation systems).</li>
</ul>



### The algorithm ###

The algorithms implemented in scikit-learn KNN model are:
<ul>
    <li>ball_tree</li>
    <li>kd_tree</li>
    <li>brute</li>
    <li>auto</li>
</ul>
the auto algorithm will attempt to decide the most appropriate algorithm based on the value passed to fit method. We will compare the results based on all of them.


```python
for number_of_points in [300]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)

    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors = 2
    algorithms_list = ["ball_tree","kd_tree","brute","auto"]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for algorithm in algorithms_list:
        clf1 = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm = algorithm)
        clf1.fit(X_train,y_train)
        tn, fp, fn, tp = confusion_matrix(y_test,clf1.predict(X_test)).ravel()
        accuracy.append(calc_metrics(tn, fp, fn, tp,'accuracy'))
        precision.append(calc_metrics(tn, fp, fn, tp,'precision'))
        recall.append(calc_metrics(tn, fp, fn, tp,'recall'))
        f1.append(calc_metrics(tn, fp, fn, tp,'f1'))


metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
methods = ["ball_tree","kd_tree","brute","auto"]
data = np.array([accuracy, precision, recall, f1])

fig, ax = plt.subplots()
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=methods, yticklabels=metrics_names)
ax.set_title('Performance Metrics by Algorithm')
plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


According to the chat, there is no special difference on the results, based on the algorithm. Or, in the other words, all of the algorithms lead to the same results and predictions. To be sure, that the number of points is not a major parameter, we will recalculate the chart using 1000 points:


```python
for number_of_points in [1000]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

    n_neighbors = 2
    algorithms_list = ["ball_tree","kd_tree","brute","auto"]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for algorithm in algorithms_list:
        clf1 = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm = algorithm)
        clf1.fit(X_train,y_train)
        tn, fp, fn, tp = confusion_matrix(y_test,clf1.predict(X_test)).ravel()
        accuracy.append(calc_metrics(tn, fp, fn, tp,'accuracy'))
        precision.append(calc_metrics(tn, fp, fn, tp,'precision'))
        recall.append(calc_metrics(tn, fp, fn, tp,'recall'))
        f1.append(calc_metrics(tn, fp, fn, tp,'f1'))


metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
methods = ["ball_tree","kd_tree","brute","auto"]
data = np.array([accuracy, precision, recall, f1])

fig, ax = plt.subplots()
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=methods, yticklabels=metrics_names)
ax.set_title('Performance Metrics by Algorithm')
plt.tight_layout()
plt.show()
```


    
![png](output_50_0.png)
    


So, the performance parameter are still independent of the algorithm, in high number of points, too. But it could affect the running time of the code. so we may be able to compare them all, from this point of view:


```python
import timeit
number_of_points = 1000
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
[Points,Labels] = labeling([x_points,y_points],radii)
X_train,X_test,y_train,y_test = train_test_split(Points,Labels,train_size=0.25,shuffle=True)

def do_KNN(algorithm):
    n_neighbors = 2
    clf1 = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm = algorithm)
    clf1.fit(X_train,y_train)



algorithms_list = ["ball_tree","kd_tree","brute","auto"]
algorithm_times = []
for algorithm in algorithms_list:
    algorithm_times.append(timeit.timeit(lambda: do_KNN(algorithm),number=1))
algorithm_times_ms = [time * 1000 for time in algorithm_times]

plt.bar(algorithms_list, algorithm_times_ms, color='skyblue')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (mili seconds)')
plt.title('Execution Time for KNeighborsClassifier with Different Algorithms')
plt.show()
```


    
![png](output_52_0.png)
    


According to this chart, the brute method is about 1.7 times faster than ball_tree method. kd_tree method and ball_tree seems to work as fast as each other.

### The weights ###

We will compare the performance metrics for two types of weights:
<ul>
    <li>uniform</li>
    <li>distance</li>
</ul>
In uniform weighting, all of the points in each neighborhood are weighted equally, but in the distance method, the weight of each point is directly dependent to its distance inverse.


```python
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
[Points,Labels] = labeling([x_points,y_points],radii)

X_train, X_test, y_train, y_test = train_test_split(Points, Labels, train_size=0.25, shuffle=True)

# Evaluate KNN with different weight methods
n_neighbors = 2
weight_list = ["uniform", "distance"]
metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

for weight in weight_list:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics['Accuracy'].append(calc_metrics(tn, fp, fn, tp,'accuracy'))
    metrics['Precision'].append(calc_metrics(tn, fp, fn, tp,'precision'))
    metrics['Recall'].append(calc_metrics(tn, fp, fn, tp,'recall'))
    metrics['F1-Score'].append(calc_metrics(tn, fp, fn, tp,'f1'))

# Plotting
x = np.arange(len(metrics))  # 4 groups (Accuracy, Precision, Recall, F1)
width = 0.35  # Width of each bar
colors = ['#1f77b4', '#ff7f0e']  # Colors for uniform & distance

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for uniform and distance side by side in each group
for i, weight in enumerate(weight_list):
    values = [metrics[metric][i] for metric in metrics]
    ax.bar(x + i * width, values, width, label=weight, color=colors[i])

# Customize plot
ax.set_xlabel('Performance Metric')
ax.set_ylabel('Score')
ax.set_title('KNN Performance: uniform vs. distance (n_neighbors=2)')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metrics.keys())
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.1)  # Adjust y-axis limit if needed

plt.tight_layout()
plt.show()

```


    
![png](output_56_0.png)
    


According to this chart, the accuracy is not much affected ny the weighting method, but Precision and recall show change in their values. And their changes are in opposite direction. In the other words, by changing the weighting method, from uniform to distance, the precision will decrease, but recall will increase.


## Evaluating the distribution ##

As we are using a uniform distribution to generate the points inside the square, and as the circle area is much smaller than the rectangle (about 5%), so the generated points distribution must be mostly out of the circle. To check this, we can evaluate the number of unique labels:


```python
number_of_points = 1000;
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
[Points,Labels] = labeling([x_points,y_points],radii)
X_train, X_test, y_train, y_test = train_test_split(Points, Labels, train_size=0.25, shuffle=True)
result =  np.unique(y_train, return_counts=True)
ratio = result[1][1]/result[1][0]
print("Number of points inside circle/Number of points outside circle= {:.2f} ".format(ratio))
```

    Number of points inside circle/Number of points outside circle= 0.25 
    

In the other words, only less than 25% of the generated points are inside the circle. So, it may have an improper effect on the model behavior. To fix this issue, first we try the stratify argument of the train_test_split method.


```python
number_of_points = 1000;
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
[Points,Labels] = labeling([x_points,y_points],radii)
X_train, X_test, y_train, y_test = train_test_split(Points, Labels, train_size=0.25, shuffle=True,stratify= Labels)
result =  np.unique(y_train, return_counts=True)
ratio = result[1][1]/result[1][0]
print("Number of points inside circle/Number of points outside circle= {:.2f} ".format(ratio))
```

    Number of points inside circle/Number of points outside circle= 0.24 
    

as you may consider the ratio does not change. We have only a stratified distribution in train and test sets. So we may need to use another method to resolve this issue.


```python
from imblearn.over_sampling import SMOTE
number_of_points = 1000;
[x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
[Points,Labels] = labeling([x_points,y_points],radii)
smote = SMOTE(sampling_strategy=0.5,random_state=42)
X_resampled, y_resampled = smote.fit_resample(Points,Labels)
result =  np.unique(y_resampled, return_counts=True)
ratio = result[1][1]/result[1][0]
print("Number of points inside circle/Number of points outside circle= {:.2f} ".format(ratio))
plot_points2(X_resampled)
```

    Number of points inside circle/Number of points outside circle= 0.50 
    


    
![png](output_64_1.png)
    


As it is clear the ratio necomes 50% and the graph shows that points are equally distributed in and out of circle. Now, we will check the effect of number of points and also number of neighbors on the accuracy.


```python
for number_of_points in [100,400,800,1200]:
    [x_points,y_points] = generate_points(radii,x_bounds,y_bounds,number_of_points)
    [Points,Labels] = labeling([x_points,y_points],radii)
    smote = SMOTE(sampling_strategy=0.5,random_state=42)
    X_resampled, y_resampled = smote.fit_resample(Points,Labels)
    X_train, X_test, y_train, y_test = train_test_split(Points, Labels, train_size=0.25, shuffle=True,stratify= Labels)

    n_neighbors_list = np.unique(np.linspace(1, int(0.25*number_of_points),int(0.25*number_of_points)).astype(int))
    scores_list = []
    for n in n_neighbors_list:
        clf1 = KNeighborsClassifier(n_neighbors=n)
        clf1.fit(X_train,y_train)
        scores_list.append(clf1.score(X_test,y_test))
    plt.plot(n_neighbors_list,scores_list,label=f'N={number_of_points}')
plt.xlabel('number of neighbors')
plt.ylabel('Accuracy score')
plt.legend(title="Number of points",bbox_to_anchor=(1.05,1),loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_66_0.png)
    


### Discussion ###

We have compared the effect of different parameters on the performance of a KNN model, implemented on a simple, hypothetical problem. The number of points show a decreasing effect on most of the performance metrics, except precision. Algorithm does not change the results, and the best distance calculation method was minkowski. The weighting method has an effect on Precision and Recall, but not any one the accuracy and F1-score.
After completing this project, I think there is something ambigues about the distribution of points and its effect on the model performance metrics. This is more emphsaized when checking the effect of weighting function, that shows special effect on precision and recall. or when we consider the effect of number_of_neighbors on the precision, which shows an increase.
This ambiguty must be clarified by furthre study and reasoning.

**References**  
1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.



### Revision ###

In this version I have implemented the followings:
<ol>
    <li>Create functions that encapsulate repetitive codes</li>
    <li>Solve the division by zero while calculating precision</li>
    <li>Resolve the problem of unbalance dataset. I used SMOTE to resample new data, with balance distribution.</li>
    <li>I add explaniation about the poor perfromacne of cosine metric</li>
    <li>I add analysis for comparison of running time comparsion for algorithms</li>
</ol>



## Future Improvments ##

The reason of increases and decreases must be explained in more technical detail


```python

```
