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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature
from matplotlib.font_manager import FontProperties

dataDirectory = '/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS/merge/'
outDirectory = '/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS/results'

'''TISSUES = ['Aorta','Bladder','Brain_Non-Myeloid','Diaphragm','Fat','Heart',
		   	   'Kidney','Large_Intestine','Limb_Muscle','Liver','Lung','Mammary_Gland','Marrow',
		       'Pancreas','Skin','Spleen','Trachea']'''
TISSUES = ['Aorta']

def main():
	for tissue in TISSUES:
		runRFParameterization(tissue,dataDirectory)

def runRFParameterization(tissue,dataDirectory):
	tissueData = importTissue(tissue, dataDirectory)
	cellIDDict = dict(enumerate(tissueData['cell'].cat.categories))
	cellCatDict = dict(enumerate(tissueData['cell_ontology_class'].cat.categories))
	category_columns = list(tissueData.select_dtypes(include='category').columns)
	for col in category_columns:
		tissueData[col] = tissueData[col].cat.codes
	nclasses = len(tissueData['cell_ontology_class'].unique())
	X = tissueData.drop(columns=['tissue','cell_ontology_class','cell_ontology_term_iri','cell_ontology_id'])
	y = label_binarize(tissueData['cell_ontology_class'], classes=list(range(nclasses)))
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]
	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}
	clf=RandomForestClassifier()
	rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
	rf_random.fit(X_train, y_train)

	print(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
	predictions = model.predict(test_features)
	errors = abs(predictions - test_labels)
	mape = 100 * np.mean(errors / test_labels)
	accuracy = 100 - mape
	print('Model Performance')
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))
    
	return accuracy

def importTissue(tissue, dataDirectory):
	print("Importing %s tissue data" % tissue)
	fileName = "%s-counts-transpose-norm-merge.csv" % tissue
	path = os.path.join(dataDirectory, fileName)
	tissueData = pd.read_csv(path)
	tissueData.dropna(how='any', subset=['tissue','cell_ontology_class'],inplace=True)
	tissueData.dropna(how='all', axis='columns', inplace=True)
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
	print(tissueData)
	return tissueData

if __name__ == "__main__":
	main()