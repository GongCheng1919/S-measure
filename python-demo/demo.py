import os
import cv2
from StructureMeasure import *
# set the ground truth path and the foreground map path
gtPath = '../demo/GT/'
fgPath = '../demo/FG/'
# set the result path
resPath = './python-result/'
if not os.path.exists(resPath):
	os.mkdir(resPath)
# set the foreground map methods
MethodNames = ['MDF','mc','DISC','rfcn','DCL','dhsnet']
# load the gtFiles
gtFiles = ListFile(gtPath,'png')
#print(gtFiles)
for item in gtFiles:
	print('Processing %s...\n'%(item))
	GT=cv2.imread(item[0]+item[1],0).astype(np.float32)/255.0
	#print(GT.shape,GT.max(),GT.min())
	for method in MethodNames:
		predname=fgPath+item[1][:-4]+"_"+method+".png"
		#print(predname)
		prediction=cv2.imread(predname,0).astype(np.float32)
		prediction=(prediction-prediction.min())/(prediction.max()-prediction.min())
		#print(prediction.shape,prediction.max(),prediction.min())
		score=StructureMeasure(prediction,GT)
		resName=resPath+item[1][:-4]+"_%0.4f_"%(score)+method+".png"
		cv2.imwrite(resName,(prediction*255).astype(np.uint8))
print('The results are saved in %s\n'%resPath);