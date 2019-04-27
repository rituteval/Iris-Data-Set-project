import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading iris dataset into dataframe df
df = pd.read_csv('iris.csv')

#filtering the dataset to seperate the 3 species for plotting and calculations
setosa = df[df['species'] == 'Iris-setosa']
versicolor = df[df['species'] == 'Iris-versicolor']
virginica = df[df['species'] == 'Iris-virginica']

#petal length table using numpy to get the max/min/mean/standard deviation petal length of each species and the overall dataset
print("\n----- Petal Length Table ------")
petal_length_table = pd.DataFrame(np.array([[np.max(setosa['petal_length']), np.min(setosa['petal_length']), np.mean(setosa['petal_length']), np.std(setosa['petal_length'])],
                     [np.max(versicolor['petal_length']), np.min(versicolor['petal_length']), np.mean(versicolor['petal_length']), np.std(versicolor['petal_length'])],
                     [np.max(virginica['petal_length']), np.min(virginica['petal_length']), np.mean(virginica['petal_length']), np.std(virginica['petal_length'])],
                     [np.max(df['petal_length']), np.min(df['petal_length']), np.mean(df['petal_length']), np.std(df['petal_length'])]]), 
                     columns=['max', 'min', 'mean','std. dev'], index=['Iris setosa','Iris versicolor','Iris virginica','All Data'])
print(petal_length_table) #print table to console

#petal width table using numpy to get the max/min/mean/standard deviation petal width of each species and the overall dataset
print("\n----- Petal Width Table -------")
petal_width_table = pd.DataFrame(np.array([[np.max(setosa['petal_width']), np.min(setosa['petal_width']), np.mean(setosa['petal_width']), np.std(setosa['petal_width'])],
                     [np.max(versicolor['petal_width']), np.min(versicolor['petal_width']), np.mean(versicolor['petal_width']), np.std(versicolor['petal_width'])],
                     [np.max(virginica['petal_width']), np.min(virginica['petal_width']), np.mean(virginica['petal_width']), np.std(virginica['petal_width'])],
                     [np.max(df['petal_width']), np.min(df['petal_width']), np.mean(df['petal_width']), np.std(df['petal_width'])]]),
                     columns=['max', 'min', 'mean','std. dev'], index=['Iris setosa','Iris versicolor','Iris virginica','All Data'])
print(petal_width_table)

#sepal length table using numpy to get the max/min/mean/standard deviation sepal length of each species and the overall dataset
print("\n------ Sepal Length Table ------")
sepal_length_table = pd.DataFrame(np.array([[np.max(setosa['sepal_length']), np.min(setosa['sepal_length']), np.mean(setosa['sepal_length']), np.std(setosa['sepal_length'])],
                     [np.max(versicolor['sepal_length']), np.min(versicolor['sepal_length']), np.mean(versicolor['sepal_length']), np.std(versicolor['sepal_length'])],
                     [np.max(virginica['sepal_length']), np.min(virginica['sepal_length']), np.mean(virginica['sepal_length']), np.std(virginica['sepal_length'])],
                     [np.max(df['sepal_length']), np.min(df['sepal_length']), np.mean(df['sepal_length']), np.std(df['sepal_length'])]]),
                     columns=['max', 'min', 'mean', 'std. dev'], index=['Iris setosa','Iris versicolor','Iris virginica','All Data'])
print(sepal_length_table)

#sepal width table using numpy to get the max/min/mean/standard deviation sepal width of each species and the overall dataset
print("\n------ Sepal Width Table ------")
sepal_width_table = pd.DataFrame(np.array([[np.max(setosa['sepal_width']), np.min(setosa['sepal_width']), np.mean(setosa['sepal_width']), np.std(setosa['sepal_width'])],
                     [np.max(versicolor['sepal_width']), np.min(versicolor['sepal_width']), np.mean(versicolor['sepal_width']), np.std(versicolor['sepal_width'])],
                     [np.max(virginica['sepal_width']), np.min(virginica['sepal_width']), np.mean(virginica['sepal_width']), np.std(virginica['sepal_width'])],
                     [np.max(df['sepal_width']), np.min(df['sepal_width']), np.mean(df['sepal_width']), np.std(df['sepal_width'])]]),
                     columns=['max', 'min', 'mean', 'std. dev'], index=['Iris setosa','Iris versicolor','Iris virginica','All Data'])
print(sepal_width_table)

#scatter plot for sepal length vs petal length for the 3 species in the dataset
plt.scatter(setosa['sepal_length'], setosa['petal_length'], color="red", alpha=0.5, label="Iris setosa")
plt.scatter(versicolor['sepal_length'], versicolor['petal_length'], color="green", alpha=0.5, label="Iris versicolor")
plt.scatter(virginica['sepal_length'], virginica['petal_length'], color="blue", alpha=0.5, label="Iris virginica")

#title and axis labels
plt.title("sepal length vs petal length")
plt.xlabel("sepal length (cm)")
plt.ylabel("petal length (cm)")
plt.legend(loc="lower right")

#show scatter plot
plt.show()

#scatter plot for sepal width vs petal width for the 3 species in the dataset
p2 = plt
p2.scatter(setosa['sepal_width'], setosa['petal_width'], color="red", alpha=0.5, label="Iris setosa")
p2.scatter(versicolor['sepal_width'], versicolor['petal_width'], color="green", alpha=0.5, label="Iris versicolor")
p2.scatter(virginica['sepal_width'], virginica['petal_width'], color="blue", alpha=0.5, label="Iris virginica")

#title and axis labes
p2.title("sepal width vs petal width")
p2.xlabel("sepal width (cm)")
p2.ylabel("petal width (cm)")
p2.legend(loc="best")

#show scatter plot
p2.show()

#getting mean values of all 4 features using pandas .mean()
#Iris setosa mean values
petal_length_setosa = setosa['petal_length'].mean()
petal_width_setosa = setosa['petal_width'].mean()
sepal_length_setosa = setosa['sepal_length'].mean()
sepal_width_setosa = setosa['sepal_width'].mean()

#Iris versicolor mean values
petal_length_versicolor = versicolor['petal_length'].mean()
petal_width_versicolor = versicolor['petal_width'].mean()
sepal_length_versicolor = versicolor['sepal_length'].mean()
sepal_width_versicolor = versicolor['sepal_width'].mean()

#Iris virginica mean values
petal_length_virginica = virginica['petal_length'].mean()
petal_width_virginica = virginica['petal_width'].mean()
sepal_length_virginica = virginica['sepal_length'].mean()
sepal_width_virginica = virginica['sepal_width'].mean()

#lists of mean values for bar plot
petal_lengths = [petal_length_setosa, petal_length_versicolor, petal_length_virginica]
petal_widths = [petal_width_setosa, petal_width_versicolor, petal_width_virginica]
sepal_lengths = [sepal_length_setosa, sepal_length_versicolor, sepal_length_virginica]
sepal_widths = [sepal_width_setosa, sepal_width_versicolor, sepal_width_virginica]
index = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

#dataframe for bar plot
bp = pd.DataFrame({'petal length': petal_lengths,'sepal length': sepal_lengths,'petal width': petal_widths, 'sepal width': sepal_widths}, index=index)

#plot the dataframe
ax = bp.plot.bar(rot=0)

#title for the bar plot and axis label
plt.title('Mean Values of Iris flower features by species')
plt.ylabel('Centimetres')

#show bar plot
plt.show()