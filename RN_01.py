#!/usr/bin/env python
# coding: utf-8

# # PREPARATION

# ## import librairies

# In[2]:


import laspy
#import pcl

#print('PCL version: %s' % pcl.__version__)
print('LASPY version: %s' % laspy.__version__)


# In[233]:


# Removing ANOYING WARNINGS
import warnings
#warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# see it ONCE
warnings.filterwarnings(action='once')


# In[231]:


from __future__ import print_function

import tensorflow as tf

import keras
### from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation     #LSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import History
#
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
#
from keras.optimizers import SGD
from keras.regularizers import l2
#
print('TensorFlow: %s' % tf.__version__)
print('Keras: %s' % keras.__version__)
# for plots
import pydot
import pydotplus
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import model_to_dot


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Indispensables
import pandas as pd
import numpy as np
#import math
print('Panda version: %s' % pd.__version__)
print('Numpy version: %s' % np.__version__)

import sklearn
print('SCIKIT LEARN version: %s' % sklearn.__version__)
# pip install scikit-plot
import scikitplot as skplt
print('SCIKIT PLOT version: %s' % skplt.__version__)
# Machine Lerning
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

# general
import os
from os import listdir
#from os.path import isfile, join
#from os import walk
import sys
import io
import glob
import platform
#
print(os.name, "--", platform.system(), "--", platform.release())
#
import argparse
#from datetime import datetime
#import h5py
#import importlib

# VISU
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#from matplotlib.colors import LogNorm


# In[18]:


# ann_visualizer
import graphviz   # need ALL *graphv* from ANACONDA
# package custominsé à partir de GITHUB https://github.com/Prodicode/ann-visualizer
from ann_visualizer.visualize import ann_viz;
#
import itertools
#
import hvplot.pandas
import selenium
#import phantomJS

# IMAGES
#from IPython.display import Image, IFrame
#from PIL import Image
#import rasterio
import imageio

import colorsys

print('MATPLOTLIB version: %s' % matplotlib.__version__)
print('Seaborn version: %s' % sns.__version__)

import geopandas as gpd
print('Geo Panda version: %s' % gpd.__version__)

import xlwt    # needed for writing pd.to_Excel
import xlrd

get_ipython().run_line_magic('matplotlib', 'inline')

# import configparser


# In[6]:


from shapely.geometry import Point, LineString, MultiLineString
from shapely.geometry import shape


# ## Définitions de fonctions

# In[7]:


def make_confusion_matrix(cf_matrix, group_names="labels", categories="categories", cmap='binary'):
    """This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to bedisplayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix.Default is True.
    normalize:     If True, show the proportions for each category.Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix. Default is True
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure.Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
    """
    pass
    return None


# In[8]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:





# ## Déclaration des directories et variables

# In[9]:


#. initialisation des variables utilisées
df_RES = None
BASE_DIR = os.getcwd()
#
separe = "/"

# Laptop
if platform.system() == 'Darwin' and platform.release() == '18.7.0':
    DATA_DIR  = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/datasets/"
    DATA_DIR1 = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/DATA1/"
    DATA_DIR2 = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/DATA2/"
    DATA_DIR3 = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/DATA3/"
    DATA_DIR4 = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/DATA_new/"
    DATA_DIR9 = "/Users/pierreleisy/Data_Science/NOTEBOOK/STAGE/"
    DATA_DIR10= "/Users/pierreleisy/Data_Science/NOTEBOOK/"
    DATA_DIR8 = "/Users/pierreleisy/Data_Science/STAGE/RESULTATS/"
# sur IMAC
else:   #if platform.system() == 'Darwin' and platform.release() == '18.0.0':
    DATA_DIR  = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/datasets/"
    DATA_DIR1 = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/DATA1/"
    DATA_DIR2 = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/DATA2/"
    DATA_DIR3 = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/DATA3/"
    DATA_DIR4 = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/DATA_new/"
    DATA_DIR9 = "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/"
    DATA_DIR10= "/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/"
    DATA_DIR8 = "/Users/pl/Desktop/DATA_SCIENCE/STAGE/RESULTATS/"
## Eurométropole
if platform.system() == 'Windows':
    DATA_DIR  = "S:\\Commun\\SIG3D\\2020\\PROJETS_2020\\20039_MachineLearning\\DATA\\"
    DATA_DIR1 = "S:\\Commun\\SIG3D\\2020\\PROJETS_2020\\20039_MachineLearning\\DATA\\"
    DATA_DIR2 = "S:\\Commun\\SIG3D\\2020\\PROJETS_2020\\20039_MachineLearning\\DATA2\\"
    DATA_DIR3 = "S:\\Commun\\SIG3D\\2020\\PROJETS_2020\\20039_MachineLearning\\DATA3\\"
    DATA_DIR9 = "C:\\Users\\STG3841\\Mes Documents (local)\\travail\\"
    separe = "\\"   
 #
#for n in range(1,4):
DALLE_NUM  = DATA_DIR1 + "DallesNumPoints5m" + separe
DALLE_ECA  = DATA_DIR1 + "DallesEcartZ5m" + separe
DALLE_POS  = DATA_DIR1 + "DallesPosition5m" + separe
#
DALLE_NUM2 = DATA_DIR2 + "DallesNumPoints5m" + separe
DALLE_ECA2 = DATA_DIR2 + "DallesEcartZ5m" + separe
DALLE_POS2 = DATA_DIR2 + "DallesPosition5m" + separe
#
DALLE_NUM3 = DATA_DIR3 + "DallesNumPoints5m" + separe
DALLE_ECA3 = DATA_DIR3 + "DallesEcartZ5m" + separe
DALLE_POS3 = DATA_DIR3 + "DallesPosition5m" + separe
#
DALLE_NUM4 = DATA_DIR4 + "DalleLAS_5m" + separe
#
print("Working directory:",BASE_DIR)
print("Data ROOT directory:",DATA_DIR)

sys.path.append(DATA_DIR)
sys.path.append(os.path.join(DATA_DIR,'DallesNumPoints5m\\'))
sys.path

print("test4:", DALLE_NUM4)


# ## Zone considérée

# In[138]:


zone = [2042500,7272000,0,0]
pas = [200,200,5]
extensionX = pas[0]*pas[2]
extensionY = pas[1]*pas[2]
zone[2] = zone[0]+extensionX
zone[3] = zone[1]+extensionY
print("zone considérée :", zone[0] ,"-", zone[1]," à", 
      zone[2],"-", zone[3], "d'extension :", extensionX, "*", extensionY," m")


# ## création des listes d'images

# In[10]:


#liste1 = [f for f in listdir(DALLE_NUM) if isfile(join(DALLE_NUM, f))]
liste_11 = [f for f in glob.glob(DALLE_NUM  + separe + "*.tif", recursive=True)]#   #only first N for test
liste_12 = [f for f in glob.glob(DALLE_ECA  + separe + "*.tif", recursive=True)]#
liste_13 = [f for f in glob.glob(DALLE_POS  + separe + "*.tif", recursive=True)]#
#
liste_21 = [f for f in glob.glob(DALLE_NUM2 + separe + "*.tif", recursive=True)]#
liste_22 = [f for f in glob.glob(DALLE_ECA2 + separe + "*.tif", recursive=True)]#
liste_23 = [f for f in glob.glob(DALLE_POS2 + separe + "*.tif", recursive=True)]#
#
liste_31 = [f for f in glob.glob(DALLE_NUM3 + separe + "*.tif", recursive=True)]#
liste_32 = [f for f in glob.glob(DALLE_ECA3 + separe + "*.tif", recursive=True)]#
liste_33 = [f for f in glob.glob(DALLE_POS3 + separe + "*.tif", recursive=True)]#
#
liste_41 = [f for f in glob.glob(DALLE_NUM4 + separe + "*.tif", recursive=True)]#
#
print("DATA1:", len(listdir(DALLE_NUM)), len(liste_11), len(liste_12), len(liste_13))
print("DATA2:", len(liste_21), len(liste_22), len(liste_23))
print("DATA3:", len(liste_31), len(liste_32), len(liste_33))
print("DATA4:", len(liste_41))

liste_NUM = os.listdir(DALLE_NUM)
liste_ECA = os.listdir(DALLE_ECA)
liste_POS = os.listdir(DALLE_POS)
print(len(liste_NUM), len(liste_ECA), len(liste_POS))


# In[ ]:





# # DONNEES

# ## LASPY ou autre librairies NUAGES de POINTS
https://towardsdatascience.com/point-cloud-data-simple-approach-f3855fdc08f5

### enormous amount of RAM for the storage of really sparse data
5030868^3 =  1,27 10e20
# In[11]:


import laspy

# Open a file in read mode:
inFile = laspy.file.File(DATA_DIR9 + "2043000_7272500_Lidar_15-16.las")
# Grab a numpy dataset of our clustering dimensions:
dataset = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()
dataset.shape


# In[23]:


get_ipython().run_cell_magic('time', '', "def frange(start, stop, step):\n    i = start\n    while i < stop:\n        yield i\n        i += step\n        \n#ground points grid filter\nn = 100 #grid step\n\ndataset_Z_filtered = dataset[[0]]\n\nzfiltered = (dataset[:, 2].max() - dataset[:, 2].min())/10 \n\n#setting height filtered from ground\nprint('zfiltered =', zfiltered)\nxstep = (dataset[:, 0].max() - dataset[:, 0].min())/n\nystep = (dataset[:, 1].max() - dataset[:, 1].min())/n\n\nfor x in frange (dataset[:, 0].min(), dataset[:, 0].max(), xstep):\n    for y in frange (dataset[:, 1].min(), dataset[:, 1].max(), ystep):\n        datasetfiltered = dataset[(dataset[:,0] > x)\n                             &(dataset[:, 0] < x+xstep)\n                             &(dataset[:, 1] > y)\n                             &(dataset[:, 1] < y+ystep)]\n    if datasetfiltered.shape[0] > 0:\n        datasetfiltered = datasetfiltered[datasetfiltered[:, 2]\n                        >(datasetfiltered[:, 2].min()+ zfiltered)]\n        if datasetfiltered.shape[0] > 0:\n            dataset_Z_filtered = np.concatenate((dataset_Z_filtered,\n                                             datasetfiltered))\nprint('dataset_Z_filtered shape', dataset_Z_filtered.shape)")


# In[24]:


print("Examining Point Format: ")
pointformat = inFile.point_format
for spec in inFile.point_format:
    print(spec.name)

During my experiments
I try to use the 4D representation of data (X, Y, Z and intensity) but the results 
do not improve over 3D (X, Y, Z) so let’s stick to the latter subset of data
# In[12]:


print('Z range =', dataset[:, 2].max() - dataset[:, 2].min())
print('Z max   =', dataset[:, 2].max(), 'Z min =', dataset[:, 2].min())
print('Y range =', dataset[:, 1].max() - dataset[:, 1].min())
print('Y max   =', dataset[:, 1].max(), 'Y min =', dataset[:, 1].min())
print('X range =', dataset[:, 0].max() - dataset[:, 0].min())
print('X max   =', dataset[:, 0].max(), 'X min =', dataset[:, 0].min())


# In[13]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path

dataset = preprocessing.normalize(dataset)
dataset_Z_filtered = preprocessing.normalize(dataset_Z_filtered)


# In[ ]:


clustering = DBSCAN(eps=2, min_samples=5, leaf_size=30).fit(dataset)
#clustering = DBSCAN(eps=2, min_samples=5, leaf_size=30).fit(dataset_Z_filtered)


# In[ ]:


core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# In[ ]:


# Black removed and is used for noise instead.
fig = plt.figure(figsize=[100, 50])
ax = fig.add_subplot(111, projection=’3d’)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)

for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
    # Black used for noise.
            col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
        xyz = dataset[class_member_mask & core_samples_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=”.”)
        
plt.title(‘Estimated number of cluster: %d’ % n_clusters_)
plt.show()


# ## Creation dataframe lampes des lampadaires présents dans la dalle X*X km

# In[219]:


def header_lampadaire(file, ):
    head = pd.read_csv(file, sep=" ", header=None, skiprows=0, nrows=5, names=["noms","val"])
    n_row = head['val'][0]
    n_col = head['val'][1]
    xlow  = head['val'][2]
    ylow  = head['val'][3]
    cell  = head['val'][4]
    print("Bord supérieur gauche 1:", xlow,ylow,n_row,n_col,cell)
    return (xlow,ylow,n_row,n_col,cell)

def create_lampadaire(file, bordX, bordY, nX, nY, taille):
    lamp = pd.read_csv(file, sep=" ", header=None, skiprows=6)
    #lampadaires1.sum().head(4)
    #print(lamp.nunique().agg(['mean','count','sum']))
    # reconstruit le tableau dans le bon schéma usuel par le calcul de la transposee
    lamp = lamp.T
    # inversion des colonnes (pas de soucis car nombre pair de colonnes ... nécessaire ou pas ?)
    lamp = lamp[lamp.columns[::-1]]
    lamp2 = lamp.copy()
    lamp2.loc['Total',:] = lamp2.sum(axis=0)
    lamp2.loc[:,'Total'] = lamp2.sum(axis=1)
    lamp2.sum(axis=0)
    print(lamp.shape, "somme totale:",lamp.sum().sum(), 
          "somme col:", lamp2.iloc[:-1,-1].sum(), "somme row:", lamp2.iloc[-1,:-1].sum())
    #lamp['Somm2'] = lamp.sum().sum()
    # calcul d'une table de nX*nY lignes
    lampe = pd.DataFrame(np.reshape(lamp.to_numpy(), (nX * nY)))
    lampe['NumI'] = lampe.index
    # calcul des dalles ... indexage des lignes et colonnes
    lampe['NumY'] = lampe['NumI'].mod(other=nY)
    lampe['NumX'] = ((lampe['NumI'] - nX + 1) / nX).apply(np.ceil)
    #lampes = lampes.drop(['Num'], axis=1)
    lampe.rename(columns={0: 'present'}, inplace=True)
    list_col = {'present': 'int8','NumI': 'int32','NumX': 'int32','NumY': 'int32'}
    lampe = lampe.astype(list_col, copy=False)
    #print(lampe.dtypes, lampe.describe(include='all'))
    lampe['X'] = bordX + lampe['NumX'] * taille
    lampe['Y'] = bordY + lampe['NumY'] * taille
    lampe['nom_NUM'] = lampe['X'].map(str) + "_" + lampe['Y'].map(str) +  "_NumPoints5m.tif"
    lampe['nom_ECA'] = lampe['X'].map(str) + "_" + lampe['Y'].map(str) + "_EcartZ5m.tif"
    lampe['nom_POS'] = lampe['X'].map(str) + "_" + lampe['Y'].map(str) +  "_Position5m.tif"
    # extraction des lampadaires
    # lampeB = lampe[lampe['present'] > 0]
    lampeB = lampe[lampe['present'] != 0]
    print("Dalle:", lampe.shape, "Nbre lampadaires:", lampeB.shape)
    lampe.head()
    return (lamp2,lampe,lampeB)

# Intialisation des variables de la dalle
fich_lamp1  = DATA_DIR1 + "LampadairePresence_0_1_5m.asc"   # DATA1
fich_lamp2a = DATA_DIR2 + "LampadairePresence_0_1_5m.asc"   # DATA2
fich_lamp2b = DATA_DIR2 + "listepositif.xlsx"

(xlow,ylow,n_row,n_col,cell) = header_lampadaire(fich_lamp1)
_ = header_lampadaire(fich_lamp2a)
##################################
(lamp1,lampes1,lampes1b) = create_lampadaire(fich_lamp1, xlow, ylow, n_row, n_col, cell)
(lamp2,lampes2,lampes2b) = create_lampadaire(fich_lamp2a, xlow+2, ylow+2, n_row, n_col, cell)
##################################
#print(lampadaires1.describe())
lampes2a = pd.read_excel(fich_lamp2b, header=0, skiprows=0)
print(lampes1.shape,lampes1b.shape,lampes2.shape,lampes2a.shape,lampes2b.shape)
lampes1b.head(5)
lampes2b.head(5)
lampes2a.head(5)


# In[ ]:





# ## Lecture autres fichiers (CSV et/ou SHP)

# ### Bancs publics

# In[135]:


# BANCS PUBLICS
f_mobi = DATA_DIR  + "mobilier_amenagement2.csv"
mobilier = pd.read_csv(f_mobi, sep=",", header=0, encoding = "utf-8")
mobilier2 = mobilier[mobilier['type_entite'] == 'banc_public']
print("Tout Mobilier:", mobilier.shape, " seulement bancs publics:", mobilier2.shape)
mobilier2.head(2)

gdf_mobi = gpd.read_file(DATA_DIR  + "shapes/mobilier_amenagement.shp")
gdf_mobi.head(2)
gdf_mobi2 = gdf_mobi[gdf_mobi['type_entit'] == 'banc_public']
# Type MULTILINESTRING
gdf_mobi2.head(4)

# Formes des fichiers des bancs publics ... caractérisation
gdf_mobi3 = gdf_mobi2.head(80)
print(gdf_mobi2.shape, gdf_mobi3.shape)

#for boundary in gdf_mobi3['geometry']:
#    print(boundary.xy)

maxi = 0
nb = 0
#for i, row in gdf_mobi2.iterrows():
for i, row in gdf_mobi3.iterrows():
    ligne = row['geometry']
    nbre = 0
    if ligne.geom_type == "LineString":
        nbre = len(ligne.coords)
        #x, y = ligne.centroid.x, ligne.centroid.y
        #print(i, x, y)
    elif ligne.geom_type == "MultiLineString":
        nb += 1
        for line in ligne:
            nbre += len(line.coords)
    #print(i, nbre, forme)
    if nbre > maxi:
        maxi = nbre
    #for pt in list(row['geometry'].coords):
        
print("Nombre maximum de points:", maxi, " avec ", nb, "MultiLineStrings")

#def getXY(pt):
#    return (pt.x, pt.y)
#centroidseries = zones['geometry'].centroid
#x,y = [list(t) for t in zip(*map(getXY, centroidseries))]

gdf_mobi3["x"] = gdf_mobi3.centroid.x
gdf_mobi3["y"] = gdf_mobi3.centroid.y
print(gdf_mobi3.shape)

gdf_mobi3.tail()


# ### Lampadaires

# In[232]:



# LAMPADAIRES CSV
f_lamp = DATA_DIR  + "lampadaires2.csv"
lampad = pd.read_csv(f_lamp, sep=",",header=0, encoding = "utf-8")
#lampad.head(3)
# LAMPADAIRES SHP
gdf_lamp = gpd.read_file(DATA_DIR  + "shapes/lampadaires.shp")                           
# extraction des coordonnées (x,y)
gdf_lamp["x"] = gdf_lamp["geometry"].x
gdf_lamp["y"] = gdf_lamp["geometry"].y
#gdf_lamp.head(3)
print("Tailles des fichiers originels:", lampad.shape, gdf_lamp.shape)

decal = 2
colonnes0 = ['gid','x','y','z_sol','date_leve','date_reco','date_creat','date_maj']
colonnes2 = ['gid','x','y','z_sol','x2','y2']
colonnes3 = ['gid','x','y','z_sol','x2','y2','intX','intY','nom_NUM']

def arrondi(df, col, c1=10, c2=2):
    cols = [col+str(c1),col+str(c2)]
    df[cols[0]] = round((df[col]/10),0)*10
    index1 = df[df[col] - df[cols[0]] >= 0].index
    index2 = df[df[col] - df[cols[0]] < 0].index
    df.loc[index1, cols[1]] = df[cols[0]]
    df.loc[index2, cols[1]] = df[cols[0]] - 5
    df[cols[1]] = df[cols[1]].astype(int)
    return df

def create_dalles(df1, df2, zone, pas, col):
    df1['intX'] = ((df1['x2']-zone[0]+1)/pas[2]).apply(np.floor)   # .apply(np.ceil)
    df1['intY'] = ((df1['y2']-zone[1]+1)/pas[2]).apply(np.floor)
    df1[['intX','intY']] = df1[['intX','intY']].astype(int)
    df1['nom_NUM'] = df1['x2'].map(str) + '_' + df1['y2'].map(str) + '_NumPoints5m.tif'
    #df1['nom_ECA'] = df1['x2'].map(str) + '_' + df1['y2'].map(str) + '_EcartZ5m.tif'
    #df1['nom_POS'] = df1['x2'].map(str) + '_' + df1['y2'].map(str) + '_Position5m.tif'
    df_b = df1[col].sort_values(['intX','intY'])
    df_c = df_b.merge(df2, how='left', left_on=['intX','intY'], right_on=['NumX','NumY'])
    df_c['X_diff'] = df_c['intX'] - df_c['NumX']
    df_c['Y_diff'] = df_tous1['intY'] - df_c['NumY']
    df_d = df_c[(df_c['X_diff'] != 0) & (df_c['Y_diff'] != 0)]
    return (df1,df_b,df_c,df_d)

dalle1 = gdf_lamp[(gdf_lamp['x'] >= zone[0]) & (gdf_lamp['x'] <= zone[2]) &
                  (gdf_lamp['y'] >= zone[1]) & (gdf_lamp['y'] <= zone[3])]

dalle2 = gdf_lamp[(gdf_lamp['x'] >= zone[0]+decal) & (gdf_lamp['x'] <= zone[2]+decal) &
                  (gdf_lamp['y'] >= zone[1]+decal) & (gdf_lamp['y'] <= zone[3]+decal)]
print(dalle1.shape, dalle2.shape)
#dalle1[colonnes0].head(3)

# Calcul des coordonnées du début de dalle pour les objets
dalle1 = arrondi(dalle1, 'x', 10, 2)
dalle1 = arrondi(dalle1, 'y', 10, 2)
(dalle1, dalle1b, dalle1c, dalle1d) = create_dalles(dalle1, lampes1b, zone, pas, colonnes3)
dalle1[colonnes3].head(2)
dalle1d
print(dalle1c.shape , dalle1d.shape)
dalle2 = arrondi(dalle2, 'x', 10, 2)
dalle2 = arrondi(dalle2, 'y', 10, 2)
(dalle2, dalle2b, dalle2c, dalle2d) = create_dalles(dalle2, lampes1b, zone, pas, colonnes3)
dalle2b[colonnes3].head(3)
dalle2d
print(dalle2c.shape , dalle2d.shape)


# In[ ]:





# In[ ]:





# ## Extraction de(s) image(s) dans un dataframe unique (à partir de imageio)

# In[23]:



off = 200000
if off > 40000:
    off = len(liste_11)
NN = 0
NM = NN + off

dal11 = ['nom_NUM', 'nom_ECA', 'nom_POS', ]
dal12 = [DALLE_NUM, DALLE_ECA, DALLE_POS,]
listes1 = [liste_11, liste_12, liste_13, ]
listes2 = [liste_21, liste_22, liste_23, ]
listes3 = [liste_31, liste_32, liste_33, ]
data = ['data1', 'data2', 'data3']
#### Pour eviter les fichiers manquants et ordonner tous les NON au début du fichier ... puis les OUI
# Compare les deux SETS (impossible avec des listes)
#for m in range(1,4):
for m in range(4,5):
    set_A = set(dal12[m-1] + lampes1b[dal11[m-1]])
    set_B1 = set(listes1[m-1])
    set_B2 = set(listes2[m-1])
    set_B3 = set(listes3[m-1])
    set_C1 = set_A & set_B1
    set_C2 = set_B1 - set_A
    liste_C = list(set_C2)
    liste_extraite0 = list(liste_C[NN:NM]) + list(set_C1) + list(set_B2) + list(set_B3)
    if m == 1:
        mult = int(len(set_C2) / (len(liste_extraite0) - len(set_C2)))
        print("Coef mutiplicateur devrait-être de:", mult, len(set_C2), (len(liste_extraite0) - len(set_C2)))
        mult = 4    # pas mettre mult = 0   ==> utiliser     liste_extraite0
        print("Coef mutiplicateur utilisé:", mult)
    liste_extraite  = list(liste_C[NN:NM]) + mult * (list(set_C1) + list(set_B2) + list(set_B3))
    #liste_extraite = list(set(list(liste_11[NN:NM]) + list(set_C1)))
    print("M:", m)
    print(len(set_A), len(set_C1), len(set_C2), len(liste_11))
    print(len(liste_extraite0), (len(liste_extraite0) - len(set_C2)), mult, len(liste_extraite))

    # creation du tenseur resultant ... merge tous les fichiers images
    fichier = DATA_DIR9 + "data_LAS_" + str(dal11[m-1]) + "_" + str(off) + "_" + str(mult) + ".npy"
    #test = False
    print("Utilisera le fichier:", fichier)
    if not(os.path.exists(fichier)):
    #if test:
    #  Ne fonctionne pas np.array(load_img(fname)) for fname in list_lamp0
        data[m-1] = np.array([np.array(imageio.imread(fname)) for fname in liste_extraite])
        data[m-1].shape
        np.save(fichier, data[m-1])
    if m == 1:
        liste_y1 = [ 0 for x in range(len(liste_C[NN:NM]))]
        # liste_y2 = [ 1 for x in range(mult * len(list(set_C1)))]
        liste_y2 = [ 1 for x in range(len(list(set_C1) + list(set_B2) + list(set_B3)))]
        liste_y = liste_y1 + mult * liste_y2
        print(len(liste_y1), len(liste_y2), len(liste_y))
    
print(sum(liste_y1),sum(liste_y2),sum(liste_y))


# ## Travailler avec PDAL

# In[61]:


### PDAL
#pdal --version
#python ./pdal_test.py

fichLAS = DATA_DIR9+"pdal_test.py"
print(fichLAS)
#!python "fichLAS"
# %run -i fichLAS

# Récupérer les directories et les PATH
# os.getcwd() == /Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/
#   ./EMS/*las  ou *.laz
#   ./EMS/DATA_new/DalleLAS_5m/dalle_#.tif

# calculs sur 1 dalle unique de 500m X 500m .... 1mn06s
get_ipython().system("python '/Users/pl/Desktop/DATA_SCIENCE/NOTEBOOK/EMS/pdal_test_4.py'")


# In[ ]:





# In[ ]:





# # KERAS - TENSORFLOW
# 
# ## Lecture du fichier de données

# In[216]:


print(tf.keras.datasets)
if platform.system() == 'Darwin' and platform.release() == '18.0.0':
    df4 = pd.read_csv(DATA_DIR9 + "donnees_IMAC_LAS_1k_1k.csv")
else:
    df4 = pd.read_csv(DATA_DIR9 + "donnees_LAS_1k_1k.csv")
df4.head(2)


# ### Rajouter des lampadaires pour améliorer l'entrainement

# In[217]:


df4b = df4[df4['label'] != 0]

mult1d = int(df4.shape[0] / df4b.shape[0])
mult1d = 5

print("On rajoute: ", mult1d-1, " fois", df4b.shape[0], " nombre de lampadaires (", mult1d*df4b.shape[0], ")")
print(df4.shape, df4.shape[0], df4b.shape[0], mult1d)

#df = pd.concat([df] * a + [df.iloc[[-1]]] * b).sort_values('col1').reset_index(drop=True)

# 
df5 = pd.concat([df4b] * (mult1d-1) + [df4]).sort_values('label').reset_index(drop=True)
print(df5.shape)
df5.head(3)


# ### Changement de la classe en categorial

# In[218]:


# Charge et split les données
X1 = df5.iloc[:,:-1]
y1 = df5.iloc[:,-1]
y1_binary = to_categorical(y1)


# In[219]:


X1.describe()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_scale = pd.DataFrame(sc.fit_transform(X1))
X1_scale.describe()


# ### Séparation Train - Test

# In[220]:


train_1d0, test_1d0, train_1d_label, test_1d_label = train_test_split(
    X1_scale, y1_binary, test_size=0.2, random_state=42, stratify=y1_binary)   #  stratify=y

print(train_1d0.shape, train_1d_label.shape, test_1d0.shape, test_1d_label.shape, y1_binary.shape, y1_binary.shape)


# In[ ]:





# ## Création d'un Modèle Réseau de Neurone "SIMPLE"

# In[228]:


#Nomb_Input = 200
#fil_M = 32    # 512-256-128-64-32 ?
filtre = 8
kernel = 3
epochs = 10   #. 10-20-30-50
batch_size = 10  # 128  nombre d'échantillon à chaque cycle
n_dens = 50  # 50 - 100 ou 200?

num_class = 2   # ou ce
max_pool1 = 2  # 2 en général   params = 158k (2)   et =   (1)

# redimensionnement nécesaire pour la convolution
train_1d = np.expand_dims(train_1d0, axis=2)
# ou ? train_data.reshape(train_1d.shape[0], train_1d.shape[1], 1)  et input_shape=(train_1d.shape[1], 1)
test_1d = np.expand_dims(test_1d0, axis=2)

def build_cnn_model_1D():
    model = Sequential()
    # kernel_initializer = "uniform" , input_dim=N
    model.add(Conv1D(filtre, kernel, activation='relu', input_shape=(train_1d.shape[1],train_1d.shape[2])))
    model.add(MaxPooling1D(pool_size=max_pool1))
    #model.add(Dropout(0.5))
    #model.add(Conv1D(64, kernel, activation='relu'))
    #model.add(MaxPooling1D(pool_size=(2)))
    # model.add(BatchNormalization())
    model.add(Flatten())        
    model.add(Dense(n_dens, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    return model

model_1d = build_cnn_model_1D()

model_1d.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
###
### model.save_weights('file.h5')  avec le model construit + les poids    model.load_weights('file.h5')
model_1d.summary()
#history0 = model_1d.fit(train_1d, train_1d_label, validation_split=0.33, verbose=2, epochs=epochs, batch_size=batch_size)
pxls = str(model_1d.layers[0].input_shape).split(',')

### from keras.utils.vis_utils import plot_model
### from keras.utils import plot_model
keras.utils.vis_utils.pydot = pydot
plot_model(model_1d)

# direct plot
SVG(model_to_dot(model_1d).create(prog='dot', format='svg'))
# to file
plot_model(model_1d, to_file=DATA_DIR8 + "model_1D.png", show_shapes=True,
        show_layer_names=True, expand_nested=False, dpi=100)

print(pxls, int(pxls[1][1:]))
ann_viz(model_1d, filename=DATA_DIR8 + "model_1D.gv", title="modele: 1D")


# ##### classifier = KerasClassifier(build_fn = model_1d, batch_size=128, nb_epoch=1)
# print(classifier)
# accuracies = cross_val_score(estimator = classifier,X = train_1d,y = train_1d_label,cv = 10,n_jobs = -1)
# mean = accuracies.mean()
# variance = accuracies.var()
# print(mean, variance)

# In[ ]:





# ### Figures

# In[71]:


#history0 = history0b
#history0.history
# plot metrics
test = False
fig = plt.figure(figsize=(8,6))
if test:
    _ = plt.plot(history0.history['msle'])
    _ = plt.plot(history0.history['mean_squared_error'])
    _ = plt.plot(history0.history['binary_accuracy'])
    _ = plt.plot(history0.history['categorical_accuracy'])
_ = plt.plot(history0.history['accuracy'])
_ = plt.ylabel('Taux de succès',fontsize=20)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)
plt.savefig(DATA_DIR8 +"fig1D_hist_" + str(mult1d) + ".png")
_ = plt.show()


# In[76]:


print(history0.history.keys())
#coul = ["blue", "lightblue", "orange", "red"]
coul = ["orange", "orange", "blue", "blue"]    
print(history0.history['val_accuracy'][-1])
print(history0.history['accuracy'][-1])
print(history0.history['val_loss'][-1])
print(history0.history['loss'][-1])

fig = plt.figure(figsize=(20,5))
_ = plt.subplot(1,2,1)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.ylabel('Erreur',fontsize=20)
for i in range(0,4,2):
    _ = plt.plot(list(history0.history.values())[i],'k-o', color=coul[i])
    _ = plt.legend(['test','train'], loc='center right', fontsize="xx-large")
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)
# plot.legend(loc=2, prop={'size': 6})
#plt.legend(fontsize=20) # using a size in points
#plt.legend(fontsize="x-large")                   
# using a named size xx-small x-small small medium large x-large xx-large
                   
_ = plt.subplot(1,2,2)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.ylabel('Taux de succès',fontsize=20)
for j in range(1,4,2):
    _ = plt.plot(list(history0.history.values())[j],'k-o', color=coul[j])
#    _ = plt.legend(['test','train'], loc='upper left', fontsize="xx-large")
    _ = plt.legend(['test','train'], loc='lower right', fontsize="xx-large")
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)

plt.savefig(DATA_DIR8 +"fig1D_LossAcc1_" + str(mult1d) + ".png")


# In[73]:


# RESULTATS
score = model_1d.evaluate(test_1d, test_1d_label, verbose=0, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# predictions
classes = model_1d.predict(test_1d, batch_size=batch_size)
predictions = model_1d.predict(test_1d[:10])
print(np.argmax(predictions, axis=1))
print(test_1d_label[:10,1])

# PLOTS
fig = plt.figure(figsize=(12,6))
_ = plt.subplot(2,1,1)
_ = plt.plot(history0.history['accuracy'])
_ = plt.plot(history0.history['val_accuracy'])
_ = plt.title('model accuracy')
_ = plt.ylabel('Taux de succès',fontsize=20)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)
_ = plt.legend(['train','test'], loc='lower right')
#
_ = plt.subplot(2,1,2)
_ = plt.plot(history0.history['loss'])
_ = plt.plot(history0.history['val_loss'])
_ = plt.title('model loss')
_ = plt.ylabel('Erreur',fontsize=20)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)
_ = plt.legend(['train','test'], loc='upper right')
_ = plt.tight_layout()
_ = fig

plt.savefig(DATA_DIR8 +"fig1D_LossAcc2_" + str(mult1d) + ".png")


# ### Matrice de Confusion

# In[74]:


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

def tab_result(df_a, df_b, model):
    # Extrait les matrices binaires en 1 seule colonne
    y_pred = pd.DataFrame(model.predict(df_b)[:,1:])
    y_true = pd.DataFrame(df_a[:,1:])
    # création de la table
    dy = y_true.merge(y_pred, left_index=True, right_index=True, suffixes=('_true', '_pred'))
    dy['0_true'] = dy['0_true'].astype(int)
    dy['0_pred2'] = dy['0_pred'].round(0)   #arrondi à l'entier le + proche
    dy['diff'] = dy['0_true'] - dy['0_pred2']
    return dy

def lampe_results(df1, df2):
    df_10 = df1[df1['0_true'] == 0]
    df_11 = dy1[dy1['0_true'] == 1]
    df_20 = df2[df2['0_true'] == 0]
    df_21 = dy2[dy2['0_true'] == 1]
    return (df_10, df_11, df_20, df_21)

#####################################
# sensitivity-recall : TPR = TP/P = TP / (TP + FN)
# specificity-select : TNR = TN/N = TN / (TN + FP)
# balance : = (TPR + TNR) / 2
# precision          : TP / (TP + FP)

# accuracy    : ACC = (TP + TN) / total
# FI score    :  2 * TP / (2 TP + FP + FN) == 2 / (1/precision + 1/recall)


# In[75]:


# transforme les matrice en listes d'entiers binaires (0 ou 1)
Y11 = pd.DataFrame(model_1d.predict(test_1d)[:,1:])[0].round(0).tolist()
Y10 = pd.DataFrame(test_1d_label[:,1:])[0].tolist()
print("\n Echantillon Test:", len(Y11))
print(confusion_matrix(Y10, Y11))
Y01 = pd.DataFrame(model_1d.predict(train_1d)[:,1:])[0].round(0).tolist()
Y00 = pd.DataFrame(train_1d_label[:,1:])[0].tolist()
print("\n Echantillon Train:", len(Y01))
print(confusion_matrix(Y00, Y01))
y_reel = pd.Series(Y10, name='Reel')
y_pred = pd.Series(Y11, name='Prédit')
df_confusion = pd.crosstab(y_reel, y_pred)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
print("\n Normalisé:")
print(df_conf_norm)
print("\n testing1:")

df_confusion = pd.crosstab(y_reel, y_pred, rownames=['Reel'], colnames=['Predit'], margins=True)
print(df_confusion)
_ = sns.heatmap(df_confusion, annot=True)
plt.show()

df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis=0)
print("\n testing2:")
print(df_conf_norm)
_ = sns.heatmap(df_conf_norm, annot=True)
plt.show()
#skplt.metrics.plot_confusion_matrix(y_reel, y_pred, figsize=(5,5))

dy1 = tab_result(test_1d_label, test_1d, model_1d)
dy2 = tab_result(train_1d_label, train_1d, model_1d)
#dy1.head(2)
#dy2.head(2)
df_test_1, df_test_0, df_train_1, df_train_0 = lampe_results(dy1, dy2)

#print(dy1.shape, dy2.shape)
#print(df_test_1.shape, df_test_0.shape, df_train_1.shape, df_train_0.shape)
#print(df_test_0.describe())
true_pos1 = df_test_1[(df_test_1['0_pred2'] == 1) & (df_test_1['0_true'] == 1)].sum()
fals_pos1 = df_test_1[(df_test_1['0_pred2'] == 1) & (df_test_1['0_true'] == 0)].sum()
fals_neg1 = df_test_1[(df_test_1['0_pred2'] == 0) & (df_test_1['0_true'] == 1)].sum()
true_neg1 = df_test_1[(df_test_1['0_pred2'] == 0) & (df_test_1['0_true'] == 0)].sum()
#print("\n ERREURS:", true_pos1, fals_pos1, fals_neg1, true_neg1)
#####################################
# calcul pour LOSS du Mean Squared Error
print("\n LOSS", mse_loss(y_reel, y_pred))
######################################################################
#### Essai pour obtenir une belle matrice de confusion 
#cm_fig = interp.plot_confusion_matrix(return_fig=True)
#ax = cm_fig.gca()
#ax.set_ylim(interp.data.c - .5, - .5)
######################################################################
cf_matrix = confusion_matrix(y_reel, y_pred)
print(cf_matrix.flatten())

ax1 = plt.subplot()
_ = skplt.metrics.plot_confusion_matrix(y_reel, y_pred, ax = ax1, figsize=(5,5))
_ = ax1.set_title('Matrice de confusion')
_ = ax1.set_xlabel('Valeurs Prédites')
_ = ax1.set_ylabel('Valeurs Réelles'); 
_ = ax1.xaxis.set_ticklabels(['Rien', 'Lampadaires'])
_ = ax1.yaxis.set_ticklabels(['Rien', 'Lampadaires'])
plt.savefig(DATA_DIR8 +"fig1D_matConf1_" + str(mult1d) + ".png")
plt.show()

ax2 = plt.subplot()
#, xticks_rotation='vertical'
_ = skplt.metrics.plot_confusion_matrix(y_reel, y_pred, ax = ax2, figsize=(5,5), normalize=True)
#_ = sns.heatmap(cf_matrix, ax = ax2)
_ = ax2.set_title('Normalisée')
_ = ax2.set_xlabel('Valeurs Prédites')
_ = ax2.set_ylabel('Valeurs Réelles'); 
_ = ax2.xaxis.set_ticklabels(['Rien', 'Lampadaires'])
_ = ax2.yaxis.set_ticklabels(['Rien', 'Lampadaires'])
plt.savefig(DATA_DIR8 +"fig1D_matConf2_" + str(mult1d) + ".png")
plt.show()

group_names = ["TN","FP","FN","TP"]
#group_names = ["Vrai Neg","Faux Pos","Faux Neg","Vrai Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.0%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
# labels, title and ticks
ax3 = plt.subplot()
_ = sns.heatmap(cf_matrix, annot=labels, ax = ax3, fmt="", cmap='Blues')
_ = ax3.set_xlabel('Valeurs Prédites')
_ = ax3.set_ylabel('Valeurs Réelles'); 
_ = ax3.xaxis.set_ticklabels(['Rien', 'Lampadaires'])
_ = ax3.yaxis.set_ticklabels(['Rien', 'Lampadaires'])
plt.savefig(DATA_DIR8 +"fig1D_matConf3_" + str(mult1d) + ".png")
plt.show()         


# In[ ]:





# # KERAS :  images bi-dimensionelles

# In[196]:



def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

def lampe_results(df1, df2):
    df_10 = df1[df1['0_true'] == 0]
    df_11 = dy1[dy1['0_true'] == 1]
    df_20 = df2[df2['0_true'] == 0]
    df_21 = dy2[dy2['0_true'] == 1]
    return (df_10, df_11, df_20, df_21)


# ## Lecture des 3 fichiers de données

# In[197]:


# mult = 1  # 0 ou autres ???
fichier1  = DATA_DIR9 + "data_LAS_nom_NUM_" + str(off) + "_" + str(mult) + ".npy"
data1     = np.load(fichier1)
fichier2  = DATA_DIR9 + "data_LAS_nom_ECA_" + str(off) + "_" + str(mult) + ".npy"
data2     = np.load(fichier2)
fichierY  = DATA_DIR9 + "data_LAS_nom_POS_" + str(off) + "_" + str(mult) + ".npy"
data9     = np.load(fichierY)
#print(type(data1), type(data2))
#if ((data1.shape[0] - data2.shape[0] != 0):
if ((data1.shape[0] - data2.shape[0] != 0) and (data1.shape[0] - data34.shape[0] != 0)):
    print(data1.shape, data2.shape, data9.shape)
else:
    print(data1.shape)
# concatenation de NUM avec ECA et POS
data14 = tf.expand_dims(data1, 3)
data24 = tf.expand_dims(data2, 3)
data34 = tf.expand_dims(data9, 3)
#print(data14.shape,data24.shape, data34.shape)

data_2d = tf.concat([data14, data24], 3)
#data_2d = tf.concat([data14, data24, data34], 3)
taille = data_2d.shape[0]
data_y   = np.load(fichierY).reshape(taille,100)
print("\n Fichiers utilisés:\n", fichier1, "\n", fichier2, "\n", fichierY)
print("Classification utiliseé:\n", fichierY, "\n")
print(data_2d.shape, taille, data_y.shape)


# In[198]:


# somme sur toutes les rangées ou colonnes   
# présence de valeurs SUPERIEURES A 1 (3 valeurs à 2 ! pourquoi ?????)
df_Y = pd.DataFrame(data_y)
df_Y['lamp'] = df_Y.sum(axis=1)
df_Y['lamp2'] = df_Y[:-1].sum(axis=0)
zeros = df_Y[df_Y['lamp'] == 0]
uns   = df_Y[df_Y['lamp'] == 1]
plus  = df_Y[df_Y['lamp'] > 1]
print(df_Y.shape, df_Y['lamp'].sum())
print("Valeurs à:  0   1   ou plus)")
print(df_Y.shape[0], len(zeros), len(uns), len(plus))
#  Ecrase les valeurs de 2  ... pour avoir des 0 ou des 1
df_Y.loc[df_Y['lamp'] > 1] = 1
uns2   = df_Y[df_Y['lamp'] == 1]
plus2  = df_Y[df_Y['lamp'] > 1]
print("Valeurs à: 0 ou 1)")
print(df_Y.shape[0], len(zeros), len(uns2), len(plus2))
#df_Y[df_Y['lamp'] == 2]


# In[199]:


print(df_Y.shape)
df_Y.head(1)


# In[200]:


print(lampes1.shape)
lampes1b.head(1)


# In[201]:


# PAS LE MEME INDEXAGE (sur 40000 et 29384 + )
# lampes1 autre DataFrame avec les indice de présence des lampadaires ... CORRECTS ? ... pas cohérent
df_tot = df_Y.merge(lampes1, left_index=True, right_index=True)
df_tot['diff'] = df_tot['lamp'] - df_tot['present']
print(df_tot.shape, lampes1.shape,df_Y.shape )
test = True
if test:
    print("diff =  2:", df_tot[df_tot['diff'] ==  2].count(axis=0)[0])
    print("diff =  1:", df_tot[df_tot['diff'] ==  1].count(axis=0)[0])
    print("diff =  0:", df_tot[df_tot['diff'] ==  0].count(axis=0)[0])
    print("diff = -1:", df_tot[df_tot['diff'] == -1].count(axis=0)[0])
    print("diff = -2:", df_tot[df_tot['diff'] == -2].count(axis=0)[0])
df_tot['diff'].describe()


# In[202]:


df_tot[df_tot['diff'] == -1]


# In[203]:


df_Y.describe()


# In[ ]:





# ### Séparation Train - Test 

# In[204]:


y = df_Y[["lamp"]]
y_binary = to_categorical(y)

len0 = df_Y[df_Y['lamp'] == 0].shape[0]
len1 = df_Y.shape[0] - len0

len01 = int(len0*0.8)
len11 = int(len1*0.8)
len02 = len0+len11

df_test = pd.DataFrame(y_binary)
print(len0, len1, y.shape, y_binary.shape, df_test.shape)

print(0, len01, " -", len0, len02, len01, len02-len0)
print(len01, len0, " -", len02, len0+len1, len0-len01, len0+len1-len02)


# In[205]:


#########   problemes d'arrondis   ????  #######
#### pas le meme nombre de lignes pour les 2 manières différentes d'extraire les lignes  ????
#y0 = df_testY.iloc[:int(len0*0.8),len0:len0+int(len1*0.8)]
#y1 = df_testY.iloc[int(len0*0.8):len0,len0+int(len1*0.8):len0+len1]
df_testY = pd.DataFrame(y_binary)
#
y0a = df_test.iloc[:len01,:]
y0b = df_test.iloc[len0:len02,:]
y0  = pd.concat([y0a, y0b])  # Dimension 3  #y0  = pd.concat(y0a, y0b)
#
y1a = df_test.iloc[len01:len0,:]   
y1b = df_test.iloc[len02:len0+len1,:]   
y1  = pd.concat([y1a, y1b])  # Dimension 3  #y1  = pd.concat(y1a, y1b)

# Probleme de dimensions des Y à prédire
print(y.shape, y0.shape, y1.shape)
#y0 = to_categorical(y0)
#y1 = to_categorical(y1)
print(len(y0[1]), len(y1[1]))

data_2d0a = data_2d[     :len01]
data_2d0b = data_2d[len0 :len02]
data_2d1a = data_2d[len01:len0 ]
data_2d1b = data_2d[len02:]
print(data_2d0a.shape, data_2d0b.shape,data_2d1a.shape, data_2d1b.shape)

#data_x0 = np.concatenate(data_x0a, data_x0b, axis=1)
data_2d0 = tf.keras.backend.concatenate((data_2d0a, data_2d0b), axis=0)
data_2d1 = tf.keras.backend.concatenate((data_2d1a, data_2d1b), axis=0)
print(data_2d0.shape,data_2d1.shape)


# In[206]:


# Charge et split les données
###############   TODO      ###############
####### fonctionne avec les 2 cubes/images 
X2d = data_2d
####### Si ne fonctionne pas avec les 2 cubes/images 
# X = data1
print("Dimensions X:", X2d.shape)
##############################################
#####   calcul de la position de la lampe dans le carré N
###  Passage mode binaire à une liste de 100
############
# Manière 1 de 
y2d = liste_y   # y2d = data_y
# Manière 2 ... directe à partir des 3 mêmes fichiers 10x10
y2d = df_Y[["lamp"]]
y2d_binary = to_categorical(y2d, dtype='int')
#y_binary = to_categorical(y, num_classes=2, dtype='int')
# print(y2d.shape)
#############################
#y2d_binary

print("Dimensions Y:", y2d_binary.shape, len(y), y0.shape, y1.shape)


# In[207]:




# tfds   tensorflow dataset splitting
# test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
# PYTORCH   torchtext.data.Dataset
# split(split_ratio=0.7, stratified=False, strata_field='label', random_state=None

methode = 'test'
#methode = 'SKLEARN'

if methode == 'SKLEARN':
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
#        X, y, test_size=0.2, random_state=42, stratify=y)
    
if methode == 'test':
    train_data, test_data     = (data_2d0, data_2d1)
    train_labels, test_labels = (y0, y1)

#print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
print("Train et Test:", train_data.shape, len(train_labels), test_data.shape, len(test_labels))

#train_data = np.expand_dims(train_data, axis=3)
#test_data = np.expand_dims(test_data, axis=3)

#### Ce n'est plus nécessaire si X est un Tenseur 4D
#train_data = train_data.reshape(train_data.shape[0], 10, 10, 1)
#test_data  = test_data.reshape(test_data.shape[0], 10, 10, 1)

#print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
print(train_data.shape, len(train_labels), test_data.shape, len(test_labels))
print(y0.shape, y1.shape)


# ## Modèles CNN 2D

# ### tensorboard plot diagramme   ou tikz in latex

# In[208]:


print(train_data.shape, train_labels.shape, df_Y.shape, y_binary.shape, len(y))

# 2) Train CONVNET on the MNIST dataset
dim = len(keras.backend.int_shape(train_data))
if dim == 2:
    input_shape = (10, 10)
if dim == 4:
    input_shape = (10, 10, 2)
    # input_shape = (10, 10, 1)
    # steps PROBLEMES si différent de 1 pour les derniers plots
    
batch_size  = 1    # 64 
ratio       = len(liste_y1) / len(liste_y2)
ratio       = 10 * 20000 / 70
poids_class = {0: 1., 1: ratio}

##############################################################
modele = 1    #  1 .... 4
##############################################################
# (epoque*steps) = Cte ???  à plus de 800 pour le moment
#  Meilleur si les 2 sont comparables
##############################################################
epochs = 200    # 30 à 50 ou 120-150-200 ???
steps  = 40     # 2-5-10-20-40
print("Dimension:", dim, " et modèle:", modele, epochs, steps)
##############################################################
filt      = 3    # 5
filters   = (filt, filt)
strides   = (1,1)
pool_size = (2,2)   # pour MaxPooling2D, stride=NONE ==> strides=2,2
cv2d  = 16          # 6-16 ; 16-32 - ????
drop  = 0.1         # 0.5
dens1 = 128         # 100-120
dens2 = 64         # 100
activ0 = "relu"     #  'relu' , 'sigmoid', softmax' 
if modele >= 5:
    activ0 = "sigmoid"
# Couches cachées:         sigmoid ou Relu
# REGRESSION:              lineaire Identite
# CLASSIFICATION binaire:  sigmoid
# CLASSIFICATION multi:    softmax
activ  = "sigmoid"   #  ou softmax'  
############################################################## 
def build_cnn_model(mod):
    cv2d0 = 6; m1=1; m2=1; 
    if mod >= 2:
        m1=2
#        if mod > 2:
#            cv2d0=cv2d; 
    if mod == 11: cv2d0 = 10
    if mod == 3: m2=2
    if mod == 4: m2=3
    print("Modele:", mod,mult,cv2d0,cv2d, drop, dens1, dens2, filt, strides, pool_size)
    model = Sequential();
    for n in range(1,m1+1):   # 2x  modeles 2 - 3 - 4
#        model.add(Conv2D(cv2d0 , filters, padding="same", strides=strides, input_shape=input_shape, activation=activ0))
        model.add(Conv2D(cv2d0, filters, padding="same", strides=strides, input_shape=input_shape, activation=activ0))
    model.add(MaxPooling2D(pool_size=pool_size))
    for n in range(1,m2+1):
#        model.add(Conv2D(m1*cv2d, filters, padding="same", strides=strides, activation=activ0))
        model.add(Conv2D(cv2d, filters, padding="same", strides=strides, activation=activ0))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dens1, activation=activ0))
    model.add(Dropout(drop))
    model.add(Dense(dens2, activation=activ0))
    model.add(Dropout(drop))
    model.add(Dense(2, activation=activ))
    return model
    
model2 = build_cnn_model(modele)
    
#model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
model2.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(),
#model2.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

###  si TENSOR = 2D
if dim == 2:
    history2 = model2.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, 
                          class_weight=poids_class,  # ou   'balanced'
          verbose=1, validation_data=(test_data, test_labels))   # validation_split=0.25
    score = model2.evaluate(test_data, test_labels, verbose=0)
elif dim == 4:
###    history2 = model2.fit_generator(train_data, train_labels, verbose=1)      
    history2 = model2.fit(train_data, train_labels, epochs=epochs, verbose=1, 
###                          # batch_size, class_weight=poids_class,   #  ValueError:  NOT supported for 3+ dimensional targets
                        validation_freq=.33, steps_per_epoch=steps)   #steps_per_epoch=2
    score = model2.evaluate(test_data, test_labels, steps=3, verbose=0)
else:
    print("WRONG tensor dimension inputs ....")

print("score total:", score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

ann_viz(model2, filename=DATA_DIR8 + "model_2D.gv", title="modele: 2D"+str(modele))


# In[ ]:





# ### Sommaire

# In[186]:


model2.summary()

# from keras.utils.vis_utils import plot_model
#from keras.utils import plot_model
keras.utils.vis_utils.pydot = pydot
plot_model(model2)

# direct plot
SVG(model_to_dot(model2).create(prog='dot', format='svg'))
# to file
plot_model(model2, to_file='model_test.png', show_shapes=True,
        show_layer_names=True, expand_nested=False, dpi=100)
# model.save_weights('file.h5')
# avec le model construit + les poids
# model.load_weights('file.h5')


# In[187]:


#creating a mapping of layer name ot layer details 
#we will create a dictionary layers_info which maps a layer name to its charcteristics
#here the layer_weights dictionary will map every layer_name to its corresponding weights
layers_info = {}
layer_weights = {}
for i in model2.layers:
    layers_info[i.name] = i.get_config()
    layer_weights[i.name] = i.get_weights()

# print(layers_info['block5_conv1'])
print(layers_info)
print(layer_weights)

layers = model2.layers
layer_ids = [1,2,3,4,5]
for i in range(1):
    print("Couche", i, layers[layer_ids[i]].get_weights()[0][:,:,:,0][:,:,0])

#plot the filters
fig,ax = plt.subplots(nrows=1,ncols=5)
for i in range(5):
    _ = ax[i].imshow(layers[layer_ids[i]].get_weights()[0][:,:,:,0][:,:,0],cmap='gray')
    ax[i].set_title('block'+str(i+1))
    _ = ax[i].set_xticks([])
    _ = ax[i].set_yticks([])


# ### Figures

# In[209]:


#history0.history
# plot metrics
fig = plt.figure(figsize=(8,6))
#_ = plt.plot(history2.history['binary_accuracy'])
_ = plt.plot(history2.history['accuracy'])
_ = plt.ylabel('Taux de succès',fontsize=20)
_ = plt.xlabel('Epoque',fontsize=20)
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)
plt.savefig(DATA_DIR8 +"fig2D_history_" + str(mult) + ".png")
_ = plt.show()


# ### history

# In[210]:


#print(history2.history.keys())
coul = ["red", "orange",  "blue", "lightblue"]

fig = plt.figure(figsize=(20,4))
_ = plt.subplot(1,2,1)
for i in range(0,2,2):
    print(i,coul[i], list(history2.history.values()))
    #_ = plt.plot(list(history2.history.values())[i],'k-o', color=coul[i])
    _ = plt.plot(list(history2.history.values())[0],'k-o', color=coul[i])
    _ = plt.ylabel('Erreur')
    _ = plt.xlabel('Epoque')
    _ = plt.xticks(fontsize=16)
    _ = plt.yticks(fontsize=16)
#    _ = plt.legend(['test'], loc='center right', fontsize="xx-large")
    
_ = plt.subplot(1,2,2)
for j in range(1,2,2):
    _ = plt.plot(list(history2.history.values())[1],'k-o', color=coul[j])
    _ = plt.ylabel('Taux de succès')
    _ = plt.xlabel('Epoque')
    _ = plt.xticks(fontsize=16)
    _ = plt.yticks(fontsize=16)
#    _ = plt.legend(['test'], loc='center right', fontsize="xx-large")

plt.savefig(DATA_DIR8 +"fig2D_LossAcc_" + str(mult) + ".png")

# RESULTATS
if dim == 2:
    score = model2.evaluate(test_data, test_labels, verbose=0, batch_size=batch_size)
    classes = model2.predict(test_data, batch_size=batch_size)
    predictions = model2.predict(test_data[:10])
if dim == 4:
    score = model2.evaluate(test_data, test_labels, verbose=0, steps=batch_size)
    classes = model2.predict(test_data, steps=batch_size)
    predictions = model2.predict(test_data[:100], steps=batch_size)
    
print('Test loss:', score[0])
print('Test accuracy:', score[1])

 #predictions = model2.predict_classes(test_data[:10])
print(np.argmax(predictions, axis=1))
print(test_labels.shape, type(test_labels))

######## TypeError: '(slice(None, 100, None), slice(0, 1, None))' is an invalid key
#print(test_labels[:100,0:1])
# PLOTS training & validation accuracy values
fig = plt.figure(figsize=(12,6))
_ = plt.subplot(2,1,1)
_ = plt.plot(history2.history['accuracy'])
if dim == 2:
    _ = plt.plot(history2.history['val_acc'])
_ = plt.title('Itérations')
_ = plt.ylabel('Taux de succès')
_ = plt.xlabel('Epoque')
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)
_ = plt.legend([ 'Test','Train'], loc='upper left', fontsize="xx-large")

_ = plt.subplot(2,1,2)
_ = plt.plot(history2.history['loss'])
if dim == 2:
    _ = plt.plot(history2.history['val_loss'])
_ = plt.title('Itérations ')
_ = plt.ylabel('Erreur')
_ = plt.xlabel('Epoque')
_ = plt.xticks(fontsize=16)
_ = plt.yticks(fontsize=16)
_ = plt.legend(['Test','Train'], loc='upper right', fontsize="xx-large")

_ = plt.tight_layout()
_ = fig

#plt.savefig(DATA_DIR8 +"fig2D_LossAcc1_" + str(mult1d) + ".png")


# ### Matrice de confusion

# In[211]:


print(data_2d0.shape, y0[:].shape, y0[1:].shape)
#y0[:]
print(test_data.shape, test_labels[:].shape)


# In[212]:


# transforme les matrice en listes d'entiers binaires (0 ou 1)
if dim == 2:
    print("start matrice 2D")
    Y01 = pd.DataFrame(model2.predict(train_data, steps=batch_size)[:,1:])[0].round(0).tolist()
    Y11 = pd.DataFrame(model2.predict(test_data, steps=batch_size)[:,1:])[0].round(0).tolist()
    Y00 = pd.DataFrame(train_labels[:,1:])[0].tolist()
    Y10 = pd.DataFrame(test_labels[:,1:])[0].tolist()
    y_reel = pd.Series(Y10, name='Reel_test')
    y_pred = pd.Series(Y11, name='Prédit_test')
    #df_confusion = pd.crosstab(y_reel, y_pred, rownames=['Reel_test'], colnames=['Predit_test'], margins=True)
    print("\n", df_confusion)
    
if dim == 4:
    print("start tensor 4D")
    Y01 = model2.predict(train_data, steps=batch_size)[:].round(0).tolist()
    Y11 = model2.predict(test_data, steps=batch_size)[:].round(0).tolist()
    Y00 = train_labels[:]
    Y10 = test_labels[:]
#    Y10 = test_labels[1:].numpy().tolist()
#    y_reel = test_labels[1:].numpy()
    y_reel0 = train_labels[1]
    y_pred0 = model2.predict(train_data, steps=batch_size).round(0)[:,1:]
    y_reel = test_labels[1]
    y_pred = model2.predict(test_data, steps=batch_size).round(0)[:,1:]

print("Echantillon Train:", train_data.shape, len(Y00), len(Y01))
#confusion_matrix(Y01, Y00)

print("Echantillon Test:", len(Y10), len(Y11))
#confusion_matrix(Y11, Y10)

print(y_reel0.shape, y_pred0.shape, y_reel.shape, y_pred.shape)
print("SOMME réelle: ", y_reel0.sum(), y_reel.sum())
print("SOMME prédite:", y_pred0.sum(), y_pred.sum())
#

ax1 = plt.subplot()
_ = skplt.metrics.plot_confusion_matrix(y_reel, y_pred, ax = ax1, figsize=(5,5), hide_counts=False)
ax1.set(xlim=(-1, 2), ylim=(2, -1))
_ = ax1.set_title('Matrice de confusion')
_ = ax1.set_xlabel('Valeurs Prédites')
_ = ax1.set_ylabel('Valeurs Réelles'); 
_ = ax1.xaxis.set_ticklabels(['Rien', 'Lampadaires'])
_ = ax1.yaxis.set_ticklabels(['Rien', 'Lampadaires'])
plt.savefig(DATA_DIR8 +"fig2D_matConf1_" + str(mult) + ".png")
plt.show()

print("Pourcentages:")
print("Train:", y_pred0.sum(), y_reel0.sum(), round(y_pred0.sum() / y_reel0.sum()* 100, 2),"%")
print("Test :", y_pred.sum(),  y_reel.sum(),  round(y_pred.sum() / y_reel.sum()* 100, 2),"%")

ax2 = plt.subplot()
#, xticks_rotation='vertical'
_ = skplt.metrics.plot_confusion_matrix(y_reel, y_pred, ax = ax2, figsize=(5,5), normalize=True)
#_ = sns.heatmap(cf_matrix, ax = ax2)
_ = ax2.set_title('Normalisée')
_ = ax2.set_xlabel('Valeurs Prédites')
_ = ax2.set_ylabel('Valeurs Réelles'); 
_ = ax2.xaxis.set_ticklabels(['Rien', 'Lampadaires'])
_ = ax2.yaxis.set_ticklabels(['Rien', 'Lampadaires'])
plt.savefig(DATA_DIR8 +"fig2D_matConf2_" + str(mult) + ".png")
plt.show()


# In[213]:


#  en cas de probleme
#df_RES_T['True-Test']   = df_RES_T['TP']  + df_RES_T['FP']
#df_RES_T['False-Test']  = df_RES_T['TN']  + df_RES_T['FN']
#df_RES_T['True-Train']  = df_RES_T['TP0'] + df_RES_T['FP0']
#df_RES_T['False-Train'] = df_RES_T['FP0']  + df_RES_T['FN0']

#df_RESULTATS = pd.read_csv(DATA_DIR9+"results_2.txt", sep=",")
#df_RESULTATS['ACC'] = round((df_RESULTATS['TP'] + df_RESULTATS['TN']) / (df_RESULTATS['TP'] + df_RESULTATS['TN'] + df_RESULTATS['FP'] + df_RESULTATS['FN']),3)
#df_RESULTATS.rename({"Unnamed: 0":"id"})
#df_RESULTATS.index = (list(df_RESULTATS["Unnamed: 0"]))
#df_RES.drop(['Unnamed: 0','True-Test','True-Train','False-Test','False-Train'],inplace=True)
#df_RES_T = df_RES.T
#df_RES.drop(['Unnamed: 0'])
#df_RES_T = df_RESULTATS.drop(['Unnamed: 0'], axis=1)
#df_RES = df_RES_T.T


# In[214]:


# Calcul et CHECK des pourcentages et des valeurs
print(y_reel0.shape, y_pred0.shape, y_reel.shape, y_pred.shape,)
print("SOMME réelle: ", y_reel0.sum(), y_reel.sum())
print("SOMME prédite:", y_pred0.sum(), y_pred.sum())

print("Pourcentages:")
print("Train:", y_pred0.sum(), y_reel0.sum(), round(y_pred0.sum() / y_reel0.sum()* 100, 2),"%")
print("Test :", y_pred.sum(),  y_reel.sum(),  round(y_pred.sum() / y_reel.sum()* 100, 2),"%")

df_res0 = pd.DataFrame(y_reel0).reset_index().rename(columns={1: "reel"})
df_res0['pred'] = pd.DataFrame(y_pred0)
df_res0['diff'] = df_res0['pred'] - df_res0['reel']
TN0  = len(df_res0[(df_res0['pred'] == 0) & (df_res0['reel'] == 0)])
FP0  = len(df_res0[(df_res0['pred'] == 0) & (df_res0['reel'] == 1)])
FN0  = len(df_res0[(df_res0['pred'] == 1) & (df_res0['reel'] == 0)])
TP0  = len(df_res0[(df_res0['pred'] == 1) & (df_res0['reel'] == 1)])
if (TP0 + FN0) != 0:
    TPR0 = round(TP0 / (TP0 + FN0),3)   # TP / P
else:
    TPR0 = int(0)
FPR0 = round(FP0 / (TN0 + FP0),3)   # FP / N
ACC0 = round((TP0 + TN0) / (TN0 + FP0 + FN0 + TP0),3)  # TP + TN / (P + N)
PUR0 = round(TP0 / (TP0 + FP0),3) # PUR=PRECIS = TP / (TP + FP)

print("\n Pourcentages:")
print('Valeurs:', TN0, FP0, FN0, TP0)
print("Train 0:", round(TN0 / (TN0+FN0)* 100, 2),"%")
print("Train 1: ", round(TP0 / (FP0+TP0)* 100, 2),"%")

df_res = pd.DataFrame(y_reel).reset_index().rename(columns={1: "reel"})
df_res['pred'] = pd.DataFrame(y_pred)
df_res['diff'] = df_res['pred'] - df_res['reel']
TN = len(df_res[(df_res['pred'] == 0) & (df_res['reel'] == 0)])
FP = len(df_res[(df_res['pred'] == 0) & (df_res['reel'] == 1)])
FN = len(df_res[(df_res['pred'] == 1) & (df_res['reel'] == 0)])
TP = len(df_res[(df_res['pred'] == 1) & (df_res['reel'] == 1)])
if (TP + FN) != 0:
    TPR = round(TP / (TP + FN),3)   # TP / P
else:
    TPR = int(0)
FPR = round(FP / (TN + FP),3)   # FP / N
ACC = round((TP + TN) / (TP + TN + FP + FN),3)  # TP + TN / (P + N)
PUR = round(TP / (TP + FP),3) # PUR=PRECIS = TP / (TP + FP)
MCC1 = (TP * TN) - (FP * FN)
MCC2 = (TP+FP)*(TP+FN)*(TN*FP)*(TN+FN)
MCC  = round(MCC1 / np.sqrt(MCC2),4)
TrueTest   = TP  + FP
FalseTest  = TN  + FN
TrueTrain  = TP0 + FP0
FalseTrain = FP0 + FN0

print("Pourcentages:")
print('Valeurs:', TN, FP, FN, TP)
print("Test 0:", round(TN / (TN+FN)* 100, 2),"%")
print("Test 1: ", round(TP / (FP+TP)* 100, 2),"%")

print("True Pos Rate:", TPR, " False Pos Rate:", FPR)
print("Accuracy:", ACC, " Purity:", PUR, " Matthews:", MCC)
print("Accuracy0:", ACC0, " Purity:", PUR0)


# In[215]:


# Delete first the dataframe 
#del df_RES; df_RES = None
# df_RES = None et resN=0 ....  EN DEBUT de programme
#
fileRES = DATA_DIR9 + "results_" + str(0) + ".txt"
print(fileRES)
if (os.path.exists(fileRES)):
    maxi = 0
    for file in glob.glob(DATA_DIR9 +"results_*.txt"):    
        fileN = os.path.basename(file)
        subs = int(fileN.split("_")[1].split(".")[0])
        if subs >= maxi:
            maxi = subs
        resN = maxi
        print(fileN, maxi, resN)
    resN += 1
    fileRES = DATA_DIR9 + "results_" + str(resN) + ".txt"
#
indexRES = ['TN', 'FP', 'FN', 'TP', 'TPR', 'FPR', 'ACC', 'PUR','MCC',
            'TN0','FP0','FN0','TP0','TPR0','FPR0','ACC0','PUR0','True-Test',
            'True-Train','False-Test','False-Train']
listeRES = [TN,FP,FN,TP,TPR,FPR,ACC,PUR,MCC,TN0,FP0,FN0,TP0,TPR0,FPR0,ACC0,
            PUR0,TrueTest,FalseTest,TrueTrain,FalseTrain]
fich_res = DATA_DIR9 +"results-TOUT.txt"
list_col = ['TN','FP','FN','TP','TN0','FP0','FN0','TP0','True-Test','True-Train','False-Test','False-Train']
list_col1 = ['modele'] + list_col
list_col2 = ['mod','mult','epoch','step','n_filt','drop','dens1','idens2','filt']
#
colRES1 = str(modele) + "_" + str(mult) + "_" + str(epochs) + "_" + str(steps) + "_" 
colRES2 = str(cv2d) + "_" + str(drop)+ "_" + str(dens1) + "_" + str(dens2) + "_" + str(filt)
colRES  = colRES1 + colRES2
if df_RES is None:
    df_RES = pd.DataFrame(listeRES, index=indexRES, columns=[colRES])
else:
    if not colRES in df_RES.columns:
        print(colRES, "\n", listeRES)
        #df_RES = df1.assign(e=e.values)  #df_RES.insert(listeRES, colonne)
        df_RES[str(colRES)] = listeRES
# transpose la table
df_RES_T = df_RES.T
# converti en entiers certaines colonnes
df_RES_T[list_col] = df_RES_T[list_col].astype(int)
resultats = df_RES_T.to_csv(fileRES, sep=",", index=True, header=True)
#df_RES.drop(['2_5_20_2_16_0.1'], axis=1, inplace=True)

df_RES_T
#df_RES_T.describe()


# In[ ]:





# In[ ]:





# In[162]:



def plot_snsHeat(fileIN,fileOUT,cmap='viridis',fig_size=(12,4),f_scale=1.0,rot=0,xf_size=20,yf_size=13):
    _ = plt.figure(figsize=fig_size,facecolor='w',edgecolor='k')
    _ = sns.set(font_scale=f_scale)
    ax1 = sns.heatmap(fileIN,cmap=cmap,cbar=True,square= False)    #df_TP.iloc[:,1:]
    _ = ax1.set_xticklabels(ax1.get_xticklabels(),fontsize=xf_size)
    _ = ax1.set_yticklabels(ax1.get_yticklabels(),rotation=rot,fontsize=xf_size)
    _ = plt.savefig(fileOUT+".png")
    
    return plt.show()

def compil_resul(fileIN, col, col2, fileOUT):
    df = pd.read_csv(fileIN, sep=",").replace(np.inf, 1).fillna(0)
    # nettoyage des données FLOAT qui sont approximés à 10-8 près
    df_0  = df.replace(np.inf, 1).fillna(0).rename(columns={'Unnamed: 0': "modele"})
    df_2 = df_0[col]
    df_1 = df_0.drop(col, axis=1)
    df_1 = df_1 * 1000
    df_1 = df_1.astype(int)
    df_1 = df_1 / 1000
    df   = df_2.merge(df_1, left_index=True, right_index=True)
    df.to_csv(fileOUT, sep=",", index=True, header=True)
    df['False-Test'] = df['TN'] + df['FN']
    df['TN'] = df['TN'] / df['False-Test']
    df['FP'] = df['FP'] / df['True-Test']
    df['FN'] = df['FN'] / df['False-Test']
    df['TP'] = df['TP'] / df['True-Test']
    df[col2] = df['modele'].str.split("_",expand=True,)
    df = df.drop(['modele'], axis=1)
    col2b = col2[1:5] + col2[6:]
    df[col2b] = df[col2b].astype(int)
    df['iteration'] = df[col2[2]]*df[col2[3]]
    df['model'] = df[col2[0]]
    df['multi'] = df[col2[1]]
    df['epoq']  = df[col2[2]]
    df['pas']   = df[col2[3]]
    df['itera'] = df[col2[2]] * df[col2[3]]
    df.set_index([col2[0],col2[1],col2[2],col2[3],'iteration'],inplace=True,drop=True,append=False)
    df.sort_index(inplace=True, ascending=True)
    return df


# In[169]:


#1 fileRES = DATA_DIR9 + "results_" + str(resN) + ".txt"
# multi = "3-4"
# _ = sns.relplot(x="itera",y="TN",hue="multi",size='model',data=df_RESU)   # size="pas"
# _ = sns.relplot(x="drop",y="TN",hue="multi",size='model',data=df_RESU)   # size="pas"
### variation du DropOut
fileRES = DATA_DIR9 + "results-dropout.txt"
###  1 TOUT
multi = "1"   # fig_size=(24,10), f_scale=2.0, rot=0, xf_size=20, yf_size=20
###  3 TOUT
multi = "3"  # fig_size=(20,12), f_scale=2.0, rot=0, xf_size=20, yf_size=20
###  4 TOUT
multi = "4"
fileRES = DATA_DIR9 + "results-" + str(multi) + "fois.txt"
print(fileRES)
name_fois = "results-" + str(multi)
fich_res1 = DATA_DIR9 + name_fois + "_TOUT.txt"
print(multi) #, nom_fichier)
print(fich_res)
df_RESULTATS = compil_resul(fileRES, list_col1, list_col2, fich_res1)
df_RESU  = df_RESULTATS.sort_values(['mod','mult','iteration'], ascending=True)
#df_TPi.to_csv(DATA_DIR9 +"results-3-4sort.txt", sep="\t", index=True, header=True)
df_RESU
col_MatConf_1 = ['TN','TP','FN','FP']
col_MatConf_2 = ['TPR','TPR0','FPR','ACC','PUR','PUR0']
# col_MatConf_2 = ['TPR','TPR0','FPR','FPR0','ACC','ACC0','PUR','PUR0','MCC']


fich_out1 = DATA_DIR8 + name_fois + "_1"
print(fich_out1)
plot_snsHeat(df_RESU[col_MatConf_1], fich_out1, cmap='viridis', fig_size=(24,10), f_scale=2.0, rot=0, xf_size=20, yf_size=20)
fich_out2 = DATA_DIR8 + name_fois + "_2"
print(fich_out2)
plot_snsHeat(df_RESU[col_MatConf_2], fich_out2, cmap='viridis', fig_size=(24,12), f_scale=2.0, rot=0, xf_size=20, yf_size=22)


# In[168]:


_ = sns.relplot(x="drop",y="TP",hue="multi",size='drop',data=df_RESU)   # size="pas"


xcol = sorted(list(df_RESU["itera"].unique()))
plot = df_RESU.hvplot(x="drop", y=["TN", "FN", "FP", "TP"], kind='line')
# colnames = ["TN", "FN", "FP", "TP"]
# df_RESU.plot(x=df_RESU["itera"], y=colnames[:], kind = 'line', legend=False, 
#                 subplots = True, sharex = True, figsize = (6,4), ls="none", marker="o")
#
# df_RESU.plot(x='itera',y=['TN','TP'],figsize=(10,5),kind="scatter",grid=True)
hvplot.save(plot, DATA_DIR8 +"results-dropout.png")
#_ = plt.savefig(DATA_DIR8 +"results-dropout.png")
plt.show()


# ### plots resultats

# In[61]:





# In[168]:


#fois = ["1", "3", "4", "3-4" "1-3-4"]

# tout les résultatsfois = ["1-3-4"]
for multi in fois:
    name_fois = "results-" + str(multi)
    nom_fichier = DATA_DIR9 + name_fois + "fois.txt"
    fich_res1 = DATA_DIR9 + name_fois + "TOUT.txt"
    fich_res2 = DATA_DIR9 + name_fois + "TOUT2.txt"
    print(multi, nom_fichier, fich_res)
    # Compilation des résultats enfichier
    df_RESULTATS = compil_resul(nom_fichier, list_col1, list_col2, fich_res1)
    #
    # df_TP  = df_RESULTATS.sort_values(['TP'], ascending=False)
    # df_RESULTATS.sort_values(['FP'], ascending=True)
    # df_TN  = df_RESULTATS.sort_values(['TN'], ascending=False)
    # df_FPR = df_RESULTATS.sort_values(['FPR'], ascending=True)
    # frames = [df_TP, df_TN, df_TPR, df_FPR, df_ACC, df_PUR]
    # df_TOUT = pd.concat(frames)
    # df_TOUT.to_csv(fich_res2, sep="\t", index=True, header=True) 
    df_RESU  = df_RESULTATS.sort_values(['mod','mult','iteration'], ascending=False)
    df_RESU.to_csv(DATA_DIR9 +"results-3-4sort.txt", sep="\t", index=True, header=True)
    df_RESU
    # plot resultats
    fich_R1 = DATA_DIR8 + name_fois + "_1"
    plot_snsHeat(df_TEST1,fich_R1,cmap='viridis',fig_size=(12,7),f_scale=1.3,rot=0,xf_size=20,yf_size=13)
    #
    fich_R2 = DATA_DIR8 + name_fois + "_2"
    plot_snsHeat(df_TEST2,fich_R2,cmap='viridis',fig_size=(15,7),f_scale=1.3,rot=0,xf_size=22,yf_size=14)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[402]:


vmin = min(df_TEST1.values.min(), df_TEST2.values.min())
vmax = max(df_TEST1.values.max(), df_TEST2.values.max())

fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[2,2,0.2]))

sns.heatmap(df_TEST1, annot=False, cmap='viridis', cbar=False, ax=axs[0], vmin=vmin)
sns.heatmap(df_TEST1, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax)

fig.colorbar(axs[1].collections[0], cax=axs[2])


# In[ ]:





# ### anciens tests

# In[171]:


listeRES_0 = [[0   ,   1, 118,472,530,994,1240,858,1279,1294],
              [1059,1411,1645,940,882,418,172, 554, 133, 118],
              [   0,   7, 472,1908, 2118,4110,5000,3538,5137,5210],
              [4218,5617,6560,3717,3506,1514, 624,2086, 487, 414]]
colRES_0 = ["1_3_10_2","1_4_10_2","1_5_10_2","1_4_100_2","1_4_20_10","1_4_400_2",
            "1_4_80_10","1_4_10_80","1_4_40_20","1_4_20_40"]
df_RES_0 = pd.DataFrame(listeRES_0, index=['TPa','FPa','TP0a','FP0a'], columns=colRES_0)

df_RES_0T = df_RES_0.T

df_RES_0T['True-Test']  = df_RES_0T['TPa']  + df_RES_0T['FPa']
df_RES_0T['True-Train'] = df_RES_0T['TP0a'] + df_RES_0T['FP0a']
df_RES_0T['TPR']  = round(df_RES_0T['TPa']  / (df_RES_0T['TPa']  + df_RES_0T['FPa']),3)
df_RES_0T['TPR0'] = round(df_RES_0T['TP0a'] / (df_RES_0T['TP0a'] + df_RES_0T['FP0a']),3)

df_RES_0T['TP']  = round(df_RES_0T['TPa'] / df_RES_0T['True-Test'],3)
df_RES_0T['FP']  = round(df_RES_0T['FPa'] / df_RES_0T['True-Test'],3)
df_RES_0T['TP0'] = round(df_RES_0T['TP0a'] / df_RES_0T['True-Train'],3)
df_RES_0T['FP0'] = round(df_RES_0T['FP0a'] / df_RES_0T['True-Train'],3)

df_RES_0T['index'] = df_RES_0T.index
df_RES_0T[['model','mult','epoques','step']] = df_RES_0T['index'].str.split("_",expand=True,)
df_RES_0T[['epoques','step']] = df_RES_0T[['epoques','step']].astype(int)

#df_RES_0T['epoques'] = df_RES_0T['iter'] * df_RES_0T['step']

df_RES_0T['iter'] = df_RES_0T['epoques'] * df_RES_0T['step']


#FPR = round(FP / (TN + FP),3)   # FP / N
#ACC = round((TP + TN0) / (TN + FP + FN + TP),3)  # TP + TN / (P + N)
#PUR = round(TP / (TP + FP),3) # PUR=PRECIS = TP / (TP + FP)

df_RES_0T = df_RES_0T.set_index(['epoques','step','iter'])
df_RES_0T
resultats_0 = df_RES_0T.to_csv(DATA_DIR9 +"results-départ0.txt", sep=",", index=True, header=True)

#annot_kws={'fontsize':10,'fontstyle':'italic','color':"k",'alpha':0.6,'rotation':"horizontal",
#    'verticalalignment':'center','backgroundcolor':'w'}
#seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=True,
#    fmt='.2g', annot_kws= annot_kws, linewidths=0, linecolor='white', cbar=True, cbar_kws=None,
#    cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
_ = plt.figure(figsize=(12,4), facecolor='w', edgecolor='k',)
yticks = df_RES_0T.index
xticks = ['TP','FP']
sns.set(font_scale=1.3)
ax = sns.heatmap(df_RES_0T[['TP','FP']],cmap='viridis',cbar=True, linewidths = 50, square= False, linecolor="k",
                linewidth=0,xticklabels=xticks, yticklabels=yticks)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 22)
_ = ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 13)  
# df_TP.to_latex(index=False)
df_RES_0T.to_excel(DATA_DIR9 +"results-départ0.xlsx", sheet_name='sheet1')
plt.savefig(DATA_DIR8 +"Results0.png")


# In[ ]:





# # KERAS hyperparametres
# 
# Nombre de couches cachées
# 
# Nombre de neurones pour chaques couches cachées
# 
# Fonction d'activation
# 
# Taux apprentissage (alpha ?) ... et sa décroissance
# 
# Momentum – β
# 
# Adam’s hyperparameter – β1, β2, ε
# 
# Taille Mini-batch 
# 

# In[116]:


print(train_data)
print(type(train_labels))


# In[110]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

filt      = 3    # 5
filters   = (filt, filt)
strides   = (1,1)
drop  = 0.1         # 0.5


def create_model(neurons=1, optimizer='adam', dropout_rate=0.0, learn_rate=0.01, momentum=0, 
                 init_mode='uniform', activation='relu', weight_constraint=0):
    # create model
    model = Sequential()
    model.add(Conv2D(6 , filters, padding="same", strides=(1,1), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, filters, padding="same", strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dropout(drop))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(drop))
    model.add(Dense(2, activation="sigmoid"))
    #model = Sequential()
    #model.add(Dense(neurons, kernel_initializer=init_mode, input_dim=8, activation=activation)'))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    #optimizer = SGD(lr=learn_rate, momentum=momentum)
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
            
# fix random seed for reproducibility
np.random.seed(42)

# load dataset
train_data, test_data     = (data_x0, data_x1)
train_labels, test_labels = (y0, y1)
# split into input (X) and output (Y) variables
X = train_data
Y = train_labels


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, dropout_rate=0.2, verbose=0)

# define the grid search parameters
#  Beta1 = 0.9
#  Beta2 = 0.999
#  Epsilon = 10e-8
learn_rate = [0.001, 0.01, 0.1]   # [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
batch_size = [10, 20, 60]   #  [10, 20, 40, 60, 80, 100]
epochs = [10, 20, 40]    # [10, 20, 40, 60, 100]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [i for i in range (1, 6, 1)]
dropout_rate = [i/10 for i in range (0, 5, 1)]
neurons = [1, 5, 10, 15, 20, 25, 30]

param_grid = dict(learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

#param_grid = dict(neurons=neurons, learn_rate=learn_rate, momentum=momentum, optimizer=optimizer, 
#                init_mode=init_mode, batch_size=batch_size, epochs=epochs, activation=activation,
#                dropout_rate=dropout_rate, weight_constraint=weight_constraint)
#  score = ???
###################################################
################## Autre manière  #################
#params = {'batch_size':[16,32,64,128], 'epochs':[2,3], 'optimizer':['adam','rmsprop']}
#grid_search = GridSearchCV(estimator=classifier,param_grid=params,scoring="accuracy",cv=2)
###################################################
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)   # n_jobs=-1  parallisation
grid_result = grid.fit(X, Y)

best_param = grid_result.best_params_
best_accuracy = grid_result.best_score_
# summarize results ==>  best_score et best_params_
print("Best: %f using %s" % (best_accuracy, best_param))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # TESTS

# In[14]:


from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset
dataset = np.loadtxt(DATA_DIR10 + "data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
modelX = Sequential()
modelX.add(Dense(12, input_dim=8, activation='relu'))
modelX.add(Dense(8, activation='relu'))
modelX.add(Dense(1, activation='sigmoid'))
# Compile model
modelX.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
modelX.fit(X, Y, epochs=3, batch_size=10)
# evaluate the model
scores = modelX.evaluate(X, Y)
print("\n%s: %.2f%%" % (modelX.metrics_names[1], scores[1]*100))


# In[134]:


ann_viz(modelX, view=True, title="My first neural network")


# In[130]:


cv2d0 = 6
cv2d = 16
modelY = Sequential()
modelY.add(Conv2D(cv2d0 , filters, padding="same", strides=strides, input_shape=input_shape))
modelY.add(Conv2D(2*cv2d, filters, padding="same", strides=strides))
modelY.add(MaxPooling2D(pool_size=pool_size))
modelY.add(Flatten())
modelY.add(Dense(dens1, activation='relu'))
modelY.add(Dropout(drop))
modelY.add(Dense(dens2, activation='relu'))
modelY.add(Dropout(drop))
modelY.add(Dense(2, activation=activ))   


# In[54]:


ann_viz(modelX, view=True, title="My first neural network")


# In[60]:


def build_cnn_model():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3),padding="same",input_shape=(10, 10, 3),activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3),padding="same",input_shape=(32, 32, 3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3),padding="same",input_shape=(32, 32, 3),activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3),padding="same",input_shape=(32, 32, 3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    return model

model = build_cnn_model()
ann_viz(model, filename="RN_1.gv", title="TEST", orient="LR")   #  BT or LR
plot_model(model, to_file='vgg.png')


# In[ ]:





# In[57]:


#importing required modules
from keras.applications import VGG16
#loading the saved model we are using the complete architecture thus include_top=True
modelvgg16 = VGG16(weights='imagenet',include_top=True)
#show the summary of model
#modelvgg16.summary()

# plot_model(modelvgg16, to_file='vgg.png')

