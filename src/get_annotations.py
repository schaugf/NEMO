# retrieve ROIs from image sets and save as binary mask

import os
import csv
import getpass
import numpy as np
from PIL import Image
from omero.gateway import BlitzGateway

saveDir = '.'
DatasetID = 3702

# save parameters
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

host = raw_input('host location')
un = raw_input('enter your OMERO username for')
pw = getpass.getpass('enter your OMERO password')

conn = BlitzGateway(un, pw, host=host, port=4064)
conn.connect()

roi_service = conn.getRoiService()

# get list of imageIds in dataset here.
dataset = conn.getObject('Dataset', DatasetID)

fileNames = []
for i in dataset.listChildren():
    img = conn.getObject('Image', i.getId())
    name = img.getName()
    print('loading image', name)
    fileNames.append(str.split(name, ' ')[0])
    
    saveFile = os.path.splitext(name)[0] + '_mask.npy'
    result = roi_service.findByImage(img.getId(), None)
    
    # generate empty mask
    mask = np.zeros((img.getSizeX(), img.getSizeY()))    
    for roi in result.rois:
        for s in roi.copyShapes():
            xdim = [int(s.getX().getValue()), 
                    int(s.getX().getValue()) + int(s.getWidth().getValue())]
            ydim = [int(s.getY().getValue()), 
                    int(s.getY().getValue()) + int(s.getHeight().getValue())]
            mask[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 1       
    # transpose mask and save
    mask = mask.T    
    np.save(os.path.join(saveDir, saveFile), mask) 

with open('/Users/schau/projects/HEtumor/data/filenames.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(fileNames)
writeFile.close()

print('done')
