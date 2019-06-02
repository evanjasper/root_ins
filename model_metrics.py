# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:10:04 2019

@author: Tebe
"""

#%%
import sys, os, glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import time

import seaborn as sns
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from scipy.sparse import hstack, csr_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
#from yellowbrick.classifier import ClassificationReport
from sklearn.utils.multiclass import unique_labels
import matplotlib.patheffects as PathEffects

sys.path.append(os.getcwd())

SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)

#plt.rcParams['figure.dpi'] = 240
#%%
def plot_roc_curve(model, y_true, y_predicted, ax):
	'''
	Function to plot the receiver operating characteristic curve for a model.
	Care should be used when applying to an unbalanced dataset.
	'''
	fpr, tpr, threshold = roc_curve(y_true, y_predicted)
	ax.set_title('ROC Curve')
	ax.plot(fpr, tpr, '-o', label=type(model).__name__)
	ax.plot([0, 1], [0, 1], '--', label='min')
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	return

def plot_precision_recall(model, X_test, y_test, ax):
	'''
	This function plots the Precision-Recall curve for a given model.
	For imbalanced datasets, should be apply to the unbalanced dataset.
	'''
	y_score = model.decision_function(X_test)
	avg_precision = average_precision_score(y_test, y_score)
	precision, recall, _ = precision_recall_curve(y_test, y_score)
	print('Average Precision-Recall score: {0:0.2f}'.format(avg_precision))
	ax.set_title('Precision-Recall curve: \n average Precision-Recall Score = {0:0.2f}'.format(avg_precision))
	ax.step(recall, precision, color='purple', alpha=0.3, where='post')
	ax.fill_between(recall, precision, step='post', alpha=0.2, color='dodgerblue')
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_ylim(0.0,1.05)
	ax.set_xlim(0.0,1.0)
	return


def plot_confusion_matrix(y_true, y_pred, ax, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
	Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white", fontsize=24)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
    #fig.tight_layout()
    return

#FIX THIS

def model_report_card(model, X_train, y_train, X_test, y_test, normalize=False):
	'''
	Function to plot a "report card" for a given model based on some metrics from scikit-learn.
	Plot classification report, confusion matrix, ROC curve, and Precision-Recall curve
	'''
	model_name = type(model).__name__
	
	fig, ax = plt.subplots(2,2)
	
	y_train_pred = model.predict(X_train)
	plt.suptitle(model_name+' report')
	plot_confusion_matrix(y_train, y_train_pred, ax=ax[0,1], normalize=normalize)
	plot_precision_recall(model, X_test, y_test, ax=ax[1,0])
	plot_roc_curve(model, y_train, y_train_pred, ax=ax[1,1])
	c = classification_report(y_train, y_train_pred, output_dict=False)
	ax[0,0].axis('off')
	ax[0,0].text(0,0.2,c,fontsize=14)
#	
#	fpr, tpr, threshold = roc_curve(y_train, y_train_pre)
#	c = classification_report(y_train, y_train_pre, output_dict=False)
##	undersample_y_score = model.decision_function(original_Xtest)
##	precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)
#	undersample_y_score = model.decision_function(X_test)
#	undersample_average_precision = average_precision_score(y_test, undersample_y_score)
#	precision, recall, _ = precision_recall_curve(y_test, undersample_y_score)
#	
#	plt.suptitle(model_name+' report')
#	ax[0,0].axis('off')
#	ax[0,0].text(0,0.2,c,fontsize=14)
#	plot_confusion_matrix(y_true=y_train, y_pred=y_train_pre, ax= ax[0,1])
#	plot_roc_curve(fpr,tpr, ax= ax[1,1])
#	plot_precision_recall(precision, recall, undersample_average_precision, ax= ax[1,0])
#	#fig.tight_layout()
	return






