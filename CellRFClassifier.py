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
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature
from matplotlib.font_manager import FontProperties

dataDirectory = '/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS/merge'
outDirectory = '/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS/results'
'''TISSUES = ['Aorta','Bladder','Brain_Myeloid','Brain_Non-Myeloid','Diaphragm','Fat','Heart',
		   	   'Kidney','Large_Intestine','Limb_Muscle','Liver','Lung','Mammary_Gland','Marrow',
		       'Pancreas','Skin','Spleen','Thymus','Tongue','Trachea']'''
# For single tissues, ignoring binary class tissues (Brain_Myeloid, Thymus, Tongue)
'''TISSUES = ['Aorta','Bladder','Brain_Non-Myeloid','Diaphragm','Fat','Heart',
		   	   'Kidney','Large_Intestine','Limb_Muscle','Liver','Lung','Mammary_Gland','Marrow',
		       'Pancreas','Skin','Spleen','Trachea']
'''
TISSUES = ['Heart']
def main():

	for tissue in TISSUES:
		runRandomForest(tissue,dataDirectory)

def runRandomForest(tissue, dataDirectory):

	tissueData = importTissue(tissue, dataDirectory)

	classCounts = tissueData['cell_ontology_class'].value_counts()
	threshold = classCounts[classCounts >= 15].index.tolist()
	droppedClasses = classCounts[classCounts < 15].index.tolist()
	print("Dropped classes: ")
	for i in droppedClasses:
		print(i)
	tissueData = tissueData[tissueData['cell_ontology_class'].isin(threshold)]
	tissueData['cell_ontology_class'] = tissueData['cell_ontology_class'].astype('category')
	tissueData['cell_ontology_class'] = tissueData['cell_ontology_class'].cat.remove_unused_categories()
	cellIDDict = dict(enumerate(tissueData['cell'].cat.categories))
	cellCatDict = dict(enumerate(tissueData['cell_ontology_class'].cat.categories))
	cellTissueDict = dict()
	for cell in list(tissueData['cell']):
		cellTissueDict[cell] = tissueData.loc[tissueData['cell'] == cell, 'tissue'].values[0]

	category_columns = list(tissueData.select_dtypes(include='category').columns)
	for col in category_columns:
		tissueData[col] = tissueData[col].cat.codes
	nclasses = len(tissueData['cell_ontology_class'].unique())
	
	X = tissueData.drop(columns=['tissue','cell_ontology_class','cell_ontology_term_iri','cell_ontology_id'])
	y = label_binarize(tissueData['cell_ontology_class'], classes=list(range(nclasses)))
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
	clf=RandomForestClassifier(n_estimators=1000)
	clf.fit(X_train,y_train)

	y_pred = np.array(clf.predict_proba(X_test))
	y_pred_map = {}
	y_test_map = {}
	# Collapse y_pred
	y_pred_collapse = np.zeros(y_pred.shape[:2], dtype=int).T
	for i in range(y_pred.shape[0]):
		for j in range(y_pred.shape[1]):
			y_pred_collapse[j,i] = y_pred[i,j].argmax(axis=0)
	for i in range(y_pred_collapse.shape[0]):
		y_pred_map[i] = y_pred_collapse[i].argmax(axis=0)
		y_test_map[i] = y_test[i].argmax(axis=0)

	# Calculate F1 Scores	
	f = open(os.path.join(outDirectory,"%s.f1.csv" % tissue), 'w')
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["AVERAGE","SCORE"])
	for avg in ['macro','micro','weighted',None]:
		if avg == None:
			writer.writerow(['None',f1_score(y_test, y_pred_collapse, average=avg)])
		else:
			writer.writerow([avg,f1_score(y_test, y_pred_collapse, average=avg)])
	f.close()

	# Output raw results
	out_match = pd.DataFrame(columns=['cell','tissue','prediction','truth'])
	out_mismatch = pd.DataFrame(columns=['cell','tissue','prediction','truth'])
	for i,x in y_pred_map.items():
		cell = cellIDDict[X_test.iloc[i]['cell']]
		tis = cellTissueDict[cell]
		pred = cellCatDict[y_pred_map[i]]
		truth = cellCatDict[y_test_map[i]]
		if pred == truth:
			out_match = out_match.append({'cell':cell, 'tissue':tis, 'prediction':pred, 'truth':truth}, ignore_index=True)
		else:
			out_mismatch = out_mismatch.append({'cell':cell, 'tissue':tis, 'prediction':pred, 'truth':truth}, ignore_index=True)
	out_match.to_csv(os.path.join(outDirectory,"%s.match.csv" % tissue), index=False)
	out_mismatch.to_csv(os.path.join(outDirectory,"%s.mismatch.csv" % tissue), index=False)
	# Output important features
	feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)[:50]
	f = open(os.path.join(outDirectory,"%s.features.csv" % tissue), 'w')
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["FEATURE","IMPORTANCE"])
	for key,value in feature_imp.items():
		writer.writerow([key,value])
	f.close()
	# ROC curves
	fpr = dict()
	tpr = dict()
	thresholds = dict()
	roc_auc = dict()
	average_precision = dict()
	precision = dict()
	recall = dict()
	for i in range(nclasses):
		fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_test[:,i], y_pred[i][:, 1])
		roc_auc[i] = auc(fpr[i], tpr[i])
		average_precision[i] = average_precision_score(y_test[:,i], y_pred[i][:, 1])
		precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], y_pred[i][:, 1])
	# Plot AUROC
	lw = 2
	fig = plt.figure()
	colors = ['darkorange', 'blue', 'brown', 'green', 'red']
	label = cellCatDict
	f = open(os.path.join(outDirectory,"%s.auc.csv" % tissue), 'w')
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["CLASS","ROC_AUC","FPR","TPR"])
	for i in range(len(fpr)):
		writer.writerow([label[i], roc_auc[i], fpr[i], tpr[i]])
		plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=lw, label='ROC curve (%s) (area = %0.4f)' % (label[i], roc_auc[i]))
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
	f.close()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC - %s' % tissue)
	plt.legend(loc="lower right")
	fig.savefig(os.path.join(outDirectory,"%s.AUROC.png" % tissue))
	plt.close()
	# Plot AUPRC
	fontP = FontProperties()
	fontP.set_size('small')
	step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
	fig = plt.figure()
	f = open(os.path.join(outDirectory,"%s.precision-recall.csv" % tissue), 'w')
	writer = csv.writer(f, delimiter=',')
	writer.writerow(["CLASS","AVERAGE PRECISION","PRECISION","RECALL"])
	for i in range(len(precision)):
		writer.writerow([label[i],average_precision[i],precision[i],recall[i]])
		plt.step(recall[i], precision[i], color=colors[i % len(colors)], alpha=0.8, lw=lw, where='post', label="%s (AP=%0.4f)" % (label[i], average_precision[i]))
	f.close()
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	art = []
	lgd = plt.legend(prop=fontP, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
	art.append(lgd)
	plt.title('Precision-Recall Curve (%s)' % tissue)
	fig.savefig(os.path.join(outDirectory,"%s.AUPRC.png" % tissue), additional_artists=art, bbox_inches="tight")
	plt.close()
	del tissueData

def importTissue(tissue, dataDirectory):
	print("Importing %s tissue data" % tissue)
	fileName = "%s-counts-transpose-norm-merge.csv" % tissue
	path = os.path.join(dataDirectory, fileName)
	tissueData = pd.read_csv(path)
	tissueData.dropna(how='any',inplace=True)
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