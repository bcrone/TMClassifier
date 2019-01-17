import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature
from matplotlib.font_manager import FontProperties

dataDirectory = '/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS/merge'
	
'''TISSUES = ['Aorta','Bladder','Brain_Myeloid','Brain_Non-Myeloid','Diaphragm','Fat','Heart',
		   	   'Kidney','Large_Intestine','Limb_Muscle','Liver','Lung','Mammary_Gland','Marrow',
		       'Pancreas','Skin','Spleen','Thymus','Tongue','Trachea']'''
TISSUES = ['Bladder']
def main():

	for tissue in TISSUES:
		runRandomForest(tissue)

def runRandomForest(tissue):
	tissueData = importTissue(tissue, dataDirectory)
	cellIDDict = dict(enumerate(tissueData['cell'].cat.categories))
	cellCatDict = dict(enumerate(tissueData['cell_ontology_class'].cat.categories))
	
	category_columns = list(tissueData.select_dtypes(include='category').columns)
	for col in category_columns:
		tissueData[col] = tissueData[col].cat.codes
	nclasses = len(tissueData['cell_ontology_class'].unique())
	X = tissueData.drop(columns=['cell_ontology_class'])
	y = label_binarize(tissueData['cell_ontology_class'], classes=list(range(nclasses)))
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
	clf=RandomForestClassifier(n_estimators=100)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	y_pred_map = y_pred.argmax(axis=1)
	y_test_map = y_test.argmax(axis=1)
	for i,x in np.ndenumerate(y_pred_map):
		index = i[0]
		cell = cellIDDict[X_test.iloc[index]['cell']]
		pred = cellCatDict[y_pred_map[i]]
		truth = cellCatDict[y_test_map[i]]
		print(cell,pred,truth)

		#print(y_pred_map[i], X_test.iloc[index]['cell'])
	del tissueData

def importTissue(tissue, dataDirectory):
	print("Importing %s tissue data" % tissue)
	fileName = "%s-counts-transpose-merge.csv" % tissue
	path = os.path.join(dataDirectory, fileName)
	tissueData = pd.read_csv(path)

	object_columns = list(tissueData.select_dtypes(include='object').columns)
	for col in object_columns:
		tissueData[col] = tissueData[col].astype('category')

	category_columns = list(tissueData.select_dtypes(include='category').columns)
	for col in category_columns:
		if tissueData[col].isna().any():
			print("Category column %s contains NA" % col)
			tissueData[col] = tissueData[col].cat.add_categories(['NA'])
			tissueData[col] = tissueData[col].fillna('NA')

	not_category_columns = list(tissueData.select_dtypes(exclude='category').columns)
	for col in not_category_columns:
		if tissueData[col].isna().any():
			print("Non-category column %s contains NA" % col)
			tissueData[col] = tissueData[col].fillna(0)

	return tissueData

if __name__ == "__main__":
	main()