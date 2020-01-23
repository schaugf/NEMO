import os
import sys
import argparse
import getpass
import tarfile
import shutil
import numpy as np
import pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
from omero.gateway import BlitzGateway
import omero.model
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from normalizeStaining import normalizeStaining

Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser()
parser.add_argument('--slideFile', type=str, 
                    default='data/raw/0000003154_1009835.svs')
parser.add_argument('--tileDir', type=str,
                    default='data/tiles')
parser.add_argument('--assemblyDir', type=str,
                    default='data/assemblies')
parser.add_argument('--tmpDir', type=str,
                    default='/mnt/scratch')
parser.add_argument('--annotationFile', type=str,
                    default='data/annotation.csv')
parser.add_argument('--splitFile', type=str,
                    default='data/splitTable.csv')
parser.add_argument('--doNormalize', type=int, default=1)
parser.add_argument('--host', type=str,
                    default='graylab.ohsu.edu')
parser.add_argument('--datasetID', type=int,
                    default=3702)
parser.add_argument('--rescaleFactor', type=int,
                    default=100)
parser.add_argument('--tileSize', type=int,
                    default=299)
parser.add_argument('--cutoff', type=int,
                    default=230)
parser.add_argument('--un', type=str, default=None)
parser.add_argument('--pw', type=str, default=None)
args = parser.parse_args()

if args.un is None:
    args.un = raw_input('enter your OMERO username for ' + args.host + ': ') 
if args.pw is None:
    args.pw = getpass.getpass('enter your OMERO password for ' + args.host + ': ')

if not os.path.isdir(args.tileDir):
    os.makedirs(args.tileDir)
    
if not os.path.isdir(args.assemblyDir):
    os.makedirs(args.assemblyDir)    
    
# open omero connection
print 'establishing BlitzGateway with OMERO'
conn = BlitzGateway(args.un, args.pw, host=args.host, port=4064)
conn.connect()
roi_service = conn.getRoiService()

splitDF = pd.read_csv(args.splitFile)

# load slide
print 'loading slide', args.slideFile
slideFile = os.path.basename(args.slideFile)
slideName = str.split(slideFile, '.')[0]
slide = openslide.OpenSlide(args.slideFile)

# get imageID from splitDF
slideId = splitDF['Slide.ID'][splitDF['slideFile'] == slideFile]
omeroId = int(splitDF['omeroId'][splitDF['slideFile'] == slideFile])
split = list(splitDF['split'][splitDF['slideFile'] == slideFile])[0]
group = list(splitDF['Group'][splitDF['slideFile'] == slideFile])[0]

print 'omeroID:', omeroId, 'split to', split, 'in group', group

img = conn.getObject('Image', omeroId)
name = img.getName()
print 'loading image ROIS', name
result = roi_service.findByImage(img.getId(), None)

# generate empty mask
mask = np.zeros((img.getSizeX() / args.rescaleFactor, 
                 img.getSizeY() / args.rescaleFactor))

# if has no annotations and is not normal liver: sys.exit()
if (len(result.rois) == 0) & (group != 'normal_liver'):
    sys.exit('no ROI annotation for '+ slideFile)


for roi in result.rois:
    for s in roi.copyShapes():
        # check if rectangle v. polygon
        if type(s) == omero.model.RectangleI:
            xdim = [int(s.getX().getValue()), 
                    int(s.getX().getValue()) + int(s.getWidth().getValue())]
            xdim = [i / args.rescaleFactor for i in xdim]
            ydim = [int(s.getY().getValue()), 
                    int(s.getY().getValue()) + int(s.getHeight().getValue())]
            ydim = [i / args.rescaleFactor for i in ydim]
            
            mask[xdim[0]:xdim[1], ydim[0]:ydim[1]] = 1
               
print 'rescaling mask'
#Image.fromarray(255*mask).show()
fullMask = (resize(mask, (img.getSizeX(),img.getSizeY()), mode='reflect')>0.5)*1
#Image.fromarray(255*fullMask.astype(np.uint8)).show()

# crop tile and save
nc = (slide.level_dimensions[0][0] // args.tileSize) - 1
nr = (slide.level_dimensions[0][1] // args.tileSize) - 1

# stack tiles and maskp
tileStack = []
maskStack = []
tileCoords = []
for c in tqdm(range(nc), desc='tiling'): 
    for r in range(nr):
        ir = args.tileSize * r * int(slide.level_downsamples[0])
        ic = args.tileSize * c * int(slide.level_downsamples[0])
        tile = np.array(slide.read_region((ic, ir), 0, (args.tileSize, args.tileSize)))[...,0:3]
        tileMask = fullMask[ic:(ic+args.tileSize), ir:(ir+args.tileSize)]
        if ((tile.mean() < args.cutoff) & (tile.mean() > 10)):  # remove black tiles
            tileStack.append(tile)
            maskStack.append(tileMask)
            tileCoords.append((ir, ic))

nTiles = len(tileStack)

if args.doNormalize:
    print 'normalizing tile stack'
    normTiles, H, E = normalizeStaining(np.vstack(tileStack))
    #normTiles = normTiles.reshape(len(tileStack), args.tileSize, args.tileSize, 3)
    normTiles.tolist()
    #Image.fromarray(normTiles)
    tileStack = normTiles

os.makedirs(os.path.join(args.tmpDir, slideName, 'tiles'))
os.makedirs(os.path.join(args.tmpDir, slideName, 'masks'))

for i in tqdm(range(len(tileStack)), desc='saving to tmpDir'):
    
    #tile = normTiles[i*args.tileSize:i*args.tileSize+args.tileSize,...]
    tile = tileStack[i]
    
    tileMask = maskStack[i]
    ir, ic = tileCoords[i]
    
    # compare
    Image.fromarray(tile)
    Image.fromarray(tileStack[i])
    Image.fromarray((tileMask*255).astype(np.uint8))
    
    if tileMask.any():
        saveFile = slideName+'-'+str(ir)+'_'+str(ic)+'-'+'Tumor.png'
    else:
        saveFile = slideName+'-'+str(ir)+'_'+str(ic)+'-'+'NonTumor.png'
    
    tileSaveFile = os.path.join(args.tmpDir, slideName, 'tiles', saveFile)    
    maskSaveFile = os.path.join(args.tmpDir, slideName, 'masks', saveFile)

    Image.fromarray(tile).save(tileSaveFile)
    Image.fromarray((tileMask*255).astype(np.uint8)).save(maskSaveFile)
    

print 'compressing to tileDir'
with tarfile.open(os.path.join(args.tileDir, slideName+'.tar.gz'), 'w:gz') as tar:
    tar.add(os.path.join(args.tmpDir, slideName), arcname=slideName)

img = slide.get_thumbnail(mask.shape)

tMask = (mask*255).T.astype(np.uint8)
cTmask = np.zeros((tMask.shape[0], tMask.shape[1], 3))
cTmask[:,:,0] = tMask
cTmImg = Image.fromarray(cTmask.astype(np.uint8))

cTmImg = cTmImg.resize(img.size)

print 'image shape:', np.array(img).shape
print 'mask shape:', np.array(cTmImg).shape

maskOverlayImage = Image.blend(img, cTmImg, alpha=0.5)
maskOverlayImage.save(os.path.join(args.assemblyDir, slideName+'_ROIoverlay.png'))


# assembly should be only as big as it needs to be.
assembly = np.zeros((slide.dimensions[1], slide.dimensions[0], 3), dtype=np.uint8)
tileClass = np.zeros((slide.dimensions[1], slide.dimensions[0], 3), dtype=np.uint8)

tileFiles = os.listdir(os.path.join(args.tmpDir, slideName, 'tiles'))

# paste each tile onto image
for t in tqdm(tileFiles, desc='generating tile overlays'):
    tile = Image.open(os.path.join(args.tmpDir, slideName, 'tiles', t))
    npc = np.array(tile) 
    ii = t.split('.')[0]
    st = ii.split('-')[-1]
    ir = int(ii.split('-')[1].split('_')[0])
    ic = int(ii.split('-')[1].split('_')[1])
    assembly[ir:ir+args.tileSize, ic:ic+args.tileSize, ...] = npc[...,0:3]
    if st == 'Tumor':
        tileClass[ir:ir+args.tileSize, ic:ic+args.tileSize, 0] = 255
    elif st == 'NonTumor':
        tileClass[ir:ir+args.tileSize, ic:ic+args.tileSize, 2] = 255


# save image
smallTileClass = tileClass[::10,::10,:]
smallAssembly = assembly[::10,::10,:]

smallTileOverlay = Image.blend(Image.fromarray(smallTileClass),
                               Image.fromarray(smallAssembly), alpha=0.5)
smallTileOverlay.save(os.path.join(args.assemblyDir, slideName+'_assembly.jpg'))







