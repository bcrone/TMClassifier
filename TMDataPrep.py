import pandas as pd 
import sys
import os

def main():
	dataDirectory = "/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/FACS"
	annotationsFile = "/Users/bcrone/Documents/RESEARCH/Rotations/WelchLab/Data/tabula-muris/00_facs_raw_data/annotations_FACS.csv"

	dataList = generateDataList(dataDirectory)
	transposeList = []
	normList = []
	mergeList = []
	annotations = pd.read_csv(annotationsFile)

	for file in dataList:
		path = "%s/%s" % (dataDirectory,file)
		print(path)
		tissue = pd.read_csv(path)
		tissue_transpose = tissue.T
		tissue_transpose.columns = tissue_transpose.iloc[0]
		tissue_transpose = tissue_transpose.reindex(tissue_transpose.index.drop('Unnamed: 0'))
		transposeFile = addPostfix(file,'transpose')
		transposeList.append(transposeFile)
		transposePath = "%s/transpose/%s" % (dataDirectory, transposeFile)
		tissue_transpose.to_csv(transposePath)

	for file in transposeList:
		transposePath = "%s/transpose/%s" % (dataDirectory, file)
		print(transposePath)
		tissue_transpose = pd.read_csv(transposePath)
		tissue_transpose_counts = tissue_transpose.select_dtypes(exclude=[object])
		tissue_transpose_counts = tissue_transpose_counts.div(tissue_transpose_counts.sum(axis=1), axis=0)
		tissue_transpose[tissue_transpose_counts.columns] = tissue_transpose_counts
		normFile = addPostfix(file,'norm')
		normList.append(normFile)
		normPath = "%s/norm/%s" % (dataDirectory, mergeFile)
		tissue_transpose.to_csv(normPath, index=False)

	for file in normList:
		normPath = "%s/norm/%s" % (dataDirectory, file)
		print(normPath)
		tissue_norm = pd.read_csv(normPath)
		tissue_norm.rename(columns={'Unnamed: 0':'cell'},inplace=True)
		tissue_norm_merge = tissue_transpose.merge(annotations, how="left", on="cell")
		mergeFile = addPostfix(file,'merge')
		mergeList.append(mergeFile)
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