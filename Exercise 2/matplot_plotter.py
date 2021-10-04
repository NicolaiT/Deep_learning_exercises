import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/clean_auto.csv')

x = df.mpg
y = df.cylinders

print('Plotting mpg x cylinders')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('cylinders')
plt.savefig('data/graph/cylinders.png')
plt.clf()

y = df.displacement

print('Plotting mpg x displacement')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('displacement')
plt.savefig('data/graph/mpg_displacement.png')
plt.clf()

y = df.horsepower

print('Plotting mpg x horsepower')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('horsepower')
plt.savefig('data/graph/mpg_horsepower.png')
plt.clf()

y = df.weight

print('Plotting mpg x weight')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('weight')
plt.savefig('data/graph/mpg_weight.png')
plt.clf()

y = df.acceleration

print('Plotting mpg x acceleration')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('acceleration')
plt.savefig('data/graph/mpg_acceleration.png')
plt.clf()

y = df.model_year

print('Plotting mpg x model_year')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('model_year')
plt.savefig('data/graph/mpg_model_year.png')
plt.clf()

y = df.origin

print('Plotting mpg x origin')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('origin')
plt.savefig('data/graph/mpg_origin.png')
plt.clf()

y = df.name

print('Plotting mpg x name')
plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('name')
plt.savefig('data/graph/mpg_name.png')
plt.clf()
