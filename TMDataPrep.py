import pandas as pd 
import scanpy as sc
import sys
import os

def main():
	dataDirectory = "/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS"
	annotationsFile = "/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/annotations_FACS.csv"

	log = open("/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Code/logs/TMDataPrep.log", 'w')
	dataList = generateDataList(dataDirectory)
	transposeList = []
	normList = []
	mergeList = []
	annotations = pd.read_csv(annotationsFile)

	for file in dataList:
		transposeFile = transposeTissue(file, dataDirectory, log)
		transposeList.append(transposeFile)

	for file in transposeList:
		normFile = normalizeTissue(file, dataDirectory, log)
		normList.append(normFile)

	for file in normList:
		annotateTissue(file, dataDirectory, annotations, log)

	log.close()

def transposeTissue(file, dataDirectory, log):
	path = "%s/%s" % (dataDirectory,file)
	log.write(path+"\n")
	tissue = pd.read_csv(path)
	tissue_transpose = tissue.T
	tissue_transpose.columns = tissue_transpose.iloc[0]
	tissue_transpose = tissue_transpose.reindex(tissue_transpose.index.drop('Unnamed: 0'))
	transposeFile = addPostfix(file,'transpose')
	transposePath = "%s/transpose/%s" % (dataDirectory, transposeFile)
	tissue_transpose.to_csv(transposePath)
	return transposeFile

def normalizeTissue(file, dataDirectory, log):
	path = "%s/transpose/%s" % (dataDirectory, file)
	log.write(path+"\n")
	tissue_transpose = sc.read_csv(path, first_column_names=True)
	log.write("Gene count (pre-filter): %s\n" % len(tissue_transpose.var_names))
	sc.pp.log1p(tissue_transpose)
	sc.pp.highly_variable_genes(tissue_transpose, flavor='seurat')
	highly_variable= tissue_transpose.var['highly_variable']
	filter_result = highly_variable[highly_variable==True].keys()
	tissue_transpose = tissue_transpose[:, filter_result]
	log.write("Gene count (post-filter): %s\n" % len(tissue_transpose.var_names))
	sc.pp.normalize_per_cell(tissue_transpose,counts_per_cell_after=1)
	sc.pp.scale(tissue_transpose)	
	tissue_norm = pd.DataFrame(data=tissue_transpose.X,index=tissue_transpose.obs_names,columns=tissue_transpose.var_names)
	tissue_norm.index.name = 'cell'
	normFile = addPostfix(file,'norm')
	normPath = "%s/norm/%s" % (dataDirectory, normFile)
	tissue_norm.to_csv(normPath, index=True)
	return normFile

def annotateTissue(file, dataDirectory, annotations, log):
	path = "%s/norm/%s" % (dataDirectory, file)
	log.write(path + "\n")
	tissue_norm = pd.read_csv(path)
	tissue_norm_merge = tissue_norm.merge(annotations, how="left", on="cell")
	mergeFile = addPostfix(file, 'merge')
	mergePath = "%s/merge/%s" % (dataDirectory, mergeFile)
	tissue_norm_merge.to_csv(mergePath, index=False)

def generateDataList(dataDirectory):
	dataList = []
	for file in os.listdir(dataDirectory):
		if file.endswith('counts.csv'):
			dataList.append(file)
	return dataList

def addPostfix(filename,postfix):
	return "%s-%s.csv" % (filename.split(".")[0],postfix)

if __name__ == "__main__":
	main()