# Iris Flower Data Set Project

## Introduction
The goal of this project is to investigate the famous Iris flower dataset and find interesting observations through the use of Python
and related tools. The Iris flower data set is a classic dataset that is traditionally used for classification and prediction i.e. to see
which features of the Iris can identify the flower to be a specific type of Iris. The data set is also frequently used in the world of computing and data science.

The data set was first introduced by the well-known statistician and biologist Ronald Fischer in 1936, it consists of 50 samples each from
the following 3 species of Iris flower: Iris setosa, Iris versicolor and Iris virginica. Each sample has 4 features measured in centimetres:
petal width, petal length, sepal width and sepal length.

## Tools and Libraries
I investigated the data set using Python 3.7.3 and some Python libraries, these libraries are:
* [pandas](https://pandas.pydata.org/)
* [numpy](https://www.numpy.org/)
* [matplotlib](https://matplotlib.org/)

Pandas is a library that is used for data manipulation and analysis. It allowed me to read the dataset from the csv file directly into a DataFrame, pandas also
has the capability of filtering the data into different sections e.g. analysing only Iris setosa data.

Numpy is a library that provides easy to use functions for mathematical operations such as getting the maximum value of an array i.e. a column of data.

Matplotlib is a library which is capable of producing various types of graphs, charts and plots. I use it to visualise some of the data and plot some features against others. The plots are found further down.

## Running iris.py
To run the script located in iris.py, you must have pandas, numpy and matplotlib installed. Guides to installing these libraries are found on their websites. The script can be run from the python shell or even a code editor with a python interpreter. For the dataset to be read correctly, the iris.csv file must be located in the same directory as iris.py.

## Investigating the Dataset
To analyse this dataset, I wrote a python script iris.py that generates tables of data which show the maximum, minimum, the mean, and the standard deviation of each feature (petal and sepal length/width). Numpy is used to get all the values for the table and pandas creates the table itself.

The script also generates 3 graphs, using matplotlib to plot the data framed by pandas. 
The first graph is a scatter plot which plots sepal length vs petal length for all 3 species.
The second graph is also a scatter plot which plots sepal width vs petal width for all 3 species. 
The final graph is a bar plot, which plots the mean values of all 4 features for each of the 3 species.

## Results and Observations
**The following scatter plot generated by the script shows sepal length vs petal length.**
![Figure 1](https://github.com/rituteval/Iris-Data-Set-project/blob/master/figures/sepal_v_petal_length.png)

It's clear from the above plot (figure 1), that the Iris setosa species has a small petal length compared to the versicolor and virginica species. The virginica species has the largest petal and sepal length out of all the species though the versicolor is almost equal to the virginica in a lot of samples. 

This suggests that the virginica and versicolor are less distinguishable from each other based on sepal and petal lengths, whereas the setosa is clearly different from the other two.

**The following scatter plot generated by the script shows sepal width vs petal width.**
![Figure 2](https://github.com/rituteval/Iris-Data-Set-project/blob/master/figures/sepal_v_petal_width.png)

In the above plot (figure 2), theres more evidence to suggest that the setosa is more distinguishable than the versicolor and virginica. The setosa has a much smaller petal width at about 0.1 to 0.5 cm in width. The virginica also has the largest petal width meaning that the virginica species is probably the largest in general out of the 3 species. 

The setosa has the longer sepal width in some samples but a lot of the samples have roughly the same sepal width for all 3 species, this indicates that the sepal width is not the best feature for predicting the species of the iris flower in this dataset. 

There also appears to be one sample of the Iris setosa which has a much smaller sepal width, this is perhaps a flower that has not grown to the average width or maybe an error in measurement.

**The following bar plot generated by the script shows the mean values of features by species.**
![Figure 3](https://github.com/rituteval/Iris-Data-Set-project/blob/master/figures/barplot.png)

As we can see from the above bar plot, the Iris setosa is the smallest of the 3 species and the Iris virginica is the largest. It appears that a the petal width and the petal length are both features that clearly distinguish the setosa from the other 2 species. 

The Iris setosa has a larger mean for sepal width but not by much, again this shows that the sepal width is the least relevant feature when it comes to classifying species in this dataset. This also show that it is more difficult to distinguish between the versicolor and the virginica species, i.e. they are more similar to each other than they are to the setosa.

### **The following tables are copies of the tables generated by the python script**

#### Petal Length Table
Set | max | min | mean | std. dev
--- | --- | --- | ---- | --------
Iris setosa | 1.9cm | 1.0cm | 1.464cm | 0.1717cm
Iris versicolor | 5.1cm | 3.0cm | 4.26cm | 0.4652cm
Iris virginica | 6.9cm | 4.5 cm | 5.552cm | 0.5464cm
All Data | 6.9cm | 1.0cm | 3.758cm | 1.7582cm 

The above table shows the iris setosa species is the smallest petal length by far, but it also has small range of length from the max to min. The mean values are roughly the middle of the range for all of the species. 

The versicolor and virginica have bigger maximum values and also have bigger ranges of about 2 to 2.5cm. There is a relatively small standard deviation for all 3 species, which shows there is not a large amount of variance for petal length, especially with the iris setosa.

#### Petal Width Table
Set | max | min | mean | std. dev
--- | --- | --- | ---- | --------
Iris setosa | 0.6cm | 0.1cm | 0.244cm | 0.1061cm
Iris versicolor | 1.8cm | 1.0cm | 1.32cm | 0.1958cm
Iris virginica | 2.5cm | 1.4 cm | 2.026cm | 0.2719cm
All Data | 2.5cm | 0.1cm | 1.1987cm | 0.7606cm 

From the above table, we can see that petal width is generally much smaller than the length, and the difference in sizes between the setosa and the other 2 species is not nearly as large as it is with the petal length. 

The virginica has the biggest standard deviation out of the 3 species but it is still a relatively small number indicating that there most samples are clustered around the mean value.

#### Sepal Length Table
Set | max | min | mean | std. dev
--- | --- | --- | ---- | --------
Iris setosa | 5.8cm | 4.3cm | 5.006cm | 0.3489cm
Iris versicolor | 7.0cm | 4.9cm | 5.936cm | 0.5110cm
Iris virginica | 7.9cm | 4.9 cm | 6.588cm | 0.6295cm
All Data | 7.9cm | 4.3cm | 5.843cm | 0.8253cm

From the sepal length table, we see that the sepal length is the largest or longest part of the flower in each species as it has the largest maximums across the board as well as the largest minium values in comparison with the other tables. 

The values are also closer together which shows that sepal length is also a poor classification feature of the species.
The standard deviation indicates the samples are less clustered around the mean than the petal length/width, however it is still a small spread.

#### Sepal Width Table
Set | max | min | mean | std. dev
--- | --- | --- | ---- | --------
Iris setosa | 4.4cm | 2.3cm | 3.418cm | 0.3772cm
Iris versicolor | 3.4cm | 2.0cm | 2.770cm | 0.3106cm
Iris virginica | 3.8cm | 2.2 cm | 2.974cm | 0.3193cm
All Data | 4.4cm | 2.0cm | 3.054cm | 0.4322cm

In the above table we can see that the sepal width is unique in this dataset because its the only feature where the iris setosa has the largest max/min/mean/std dev of all 3 species. However, although they are larger they are still quite similar to the values of the Iris virginica and the Iris versicolor.

This shows that the sepal width is yet again the worst classification feature, closely followed by the sepal length. Therefore, the sepal width/length is generally the same across the 3 species of the dataset.

## Conclusion
In conclusion, the evidence suggests that out of the 3 species in the Iris dataset, the Iris setosa is the smallest species. The Iris versicolor is the second largest and the Iris virginica is the largest of the dataset. The versicolor and the virginica 