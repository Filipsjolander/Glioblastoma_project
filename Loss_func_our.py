### This version of the model utilizes our own loss function.
### Filip Sjölander - filip0917@gmail.com

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import pandas as pd
#from sklearn.decomposition import PCA

#### Setup #############################

# Collected data points
filnamn = 'Test_glio.csv' ### These are the imported data points from the MySQL database. They should be of type CSV(; separated).
df=pd.read_csv(filnamn, sep=';',header=None)
a = df.values
b=a[1:,1:]
data_set = b.tolist()

for point in range(len(data_set)):
	for value in range(3):
		data_set[point][value] = int(data_set[point][value])
		data_set[point][value] -= 1

#print((data_set))

# Number of cell lines handled in the data set
number_of_images = 25

# The desired number of dimensions in the projection
dimensions = 2

# The strength of the L2 factorization. 0 => no regularization.
lam = 0

# The names of the cell lines used in the plot. It is important that the cell lines have the same order as in the database, so that they are correctly plotted.
picture_names = ['3005','3013','3021','3024','3028','3031','3033','3051','3054','3082','3086','3110','3117','3118','3123','3167','3179','3180','3202','3220','3230','3275','3279','3289','3291']

# The number of times we 'train' the model, i.e. how many iterations of minimization we do.
cycles = 10000

########################################


picture_list = list(range(number_of_images))

X = tf.Variable(initial_value=tf.random.uniform([number_of_images, dimensions], -10, 10,dtype='float64'), name='X' )

def Heaviside(argument, slope):
	e = 2.718281828
	Exponential = e**((-1)*slope*argument)
	return 1/(1+Exponential)


#### Here we create our loss function ####
I = np.array(data_set)

A = np.array([I[:,0]]).T
B = np.array([I[:,1]]).T
C = np.array([I[:,2]]).T

A = A.tolist()
B = B.tolist()
C = C.tolist()


X_A = tf.gather_nd(X, A)
X_B = tf.gather_nd(X, B)
X_C = tf.gather_nd(X, C)


#AB is the distance between A and B 
AB = tf.reduce_sum((X_A - X_B)**2.0,axis=1)**0.5
AC = tf.reduce_sum((X_A - X_C)**2.0,axis=1)**0.5
BC = tf.reduce_sum((X_C - X_B)**2.0,axis=1)**0.5


loss = tf.reduce_sum(Heaviside(AB-AC,1) + Heaviside(AB-BC,1))

loss += lam * tf.reduce_sum(X**2.0)
########################################


### This is where we minimize the loss function, projecting the images as coordinates in n-dimensional space as efficiently as possible.
opt = tf.train.AdamOptimizer(0.05)

train = opt.minimize(loss)

sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

for step in range(cycles): 
	sess.run(train)
	if step % 500 == 0:
		print(step, sess.run(loss))


### Plotting the data in up to three dimensions.

class Annotation3D(Annotation):
	'''Annotate the point xyz with text s'''

	def __init__(self, s, xyz, *args, **kwargs):
		Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
		self._verts3d = xyz        

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.xy=(xs,ys)
		Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
	'''add anotation text s to to Axes3d ax'''

	tag = Annotation3D(s, *args, **kwargs)
	ax.add_artist(tag)


if dimensions == 1:

	x = []
	y = []
	for i in range(len(picture_list)):
		x.append(sess.run(X[i,0]))
		y.append(0)



	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.set_axis_off()  ### This removes the axes, making the plot easier to read.

	for i, txt in enumerate(picture_names):
	    ax.annotate(txt, (x[i], y[i]))

	plt.show()
	print(x)

if dimensions == 2:

	x = []
	y = []
	for i in range(len(picture_list)):
		x.append(sess.run(X[i,0]))
		y.append(sess.run(X[i,1]))



	fig, ax = plt.subplots()
	ax.scatter(x, y)#, c=colours, s=50 ,cmap='viridis')
	#ax.scatter(x, y)
	ax.set_axis_off()  ### This removes the axes, making the plot easier to read.

	for i, txt in enumerate(picture_names):
	    ax.annotate(txt, (x[i], y[i]))

	plt.show()


	print(x,y)
	#PCA_values = np.array([x,y]).T.tolist()
	#print(PCA_values)
	#pca = PCA(n_components = 2)
	#pca.fit(PCA_values)
	#print(pca.singular_values_)



if dimensions == 3:
	x = []
	y = []
	z = []
	for i in range(len(picture_list)):
		x.append(sess.run(X[i,0]))
		y.append(sess.run(X[i,1]))
		z.append(sess.run(X[i,2]))

	xyz = zip(x, y, z)
              

	# create figure        
	fig = plt.figure(dpi=60)
	ax = fig.gca(projection='3d')
	ax.set_axis_off()  ### This removes the axes, making the plot easier to read.

	ax.scatter(x,y,z, marker='o', c = [0]*number_of_images, s = 64)    

	for j, xyz_ in enumerate(xyz): 
		annotate3D(ax, s=picture_names[j], xyz=xyz_, fontsize=10, xytext=(-3,3), textcoords='offset points', ha='right',va='bottom')    

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

	print(x,y,z)
else:
	print(sess.run(X))
