import os
import argparse
import numpy as np
import pandas as pd
import tarfile
import openslide
from PIL import Image

from keras.models import load_model
from keras.preprocessing import image

from normalizeStaining import normalizeStaining

Image.MAX_IMAGE_PIXELS = None


def Thermal(val):
    ''' convert single value [0,1] to 3-channel thermal color map
    '''
    val = val * 255 * 3
    colc = np.zeros((3))
    colc[0] = max(0, min(val, 255))         # red
    colc[1] = max(0, min(val-255, 255))     # yellow
    colc[2] = max(0, min(val-255-255, 255)) # white
    return colc


def CropSlide(args):
    ''' preprocess slide into cropped images
    '''
    # load slide
    slide = openslide.OpenSlide(args.slideFile)
    slideName = args.slideFile.split('/')[-1].split('.')[0]
    
    # create save directories
    os.makedirs(args.tmpDir, exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, slideName), exist_ok=True)
    os.makedirs(args.tileDir, exist_ok=True)   
    os.makedirs(args.tumorDir, exist_ok=True)
    os.makedirs(args.hetDir, exist_ok=True)
    os.makedirs('data/assemblies', exist_ok=True)
    
    # crop tile, save, and symlink
    nc = (slide.level_dimensions[args.level][0] // args.tileSize) - 1
    nr = (slide.level_dimensions[args.level][1] // args.tileSize) - 1
    
    tileStack = []
    tileCoords = []
    for c in range(nc):
        for r in range(nr):
            ir = args.tileSize * r * int(slide.level_downsamples[args.level])
            ic = args.tileSize * c * int(slide.level_downsamples[args.level])
            tile = np.array(slide.read_region((ic, ir), args.level, (args.tileSize, args.tileSize)))[...,0:3]
            if tile.mean() < args.cutoff:
                tileStack.append(tile)
                tileCoords.append((ir, ic))
    
    if args.doNormalize:
        print('normalizing tile stack')
        normTiles, H, E = normalizeStaining(np.vstack(tileStack[1:4]))
        normTiles.tolist()
        tileStack = normTiles
    
    for i in range(len(tileStack)):
        tile = tileStack[i]
        ir, ic = tileCoords[i]
        saveFile = slideName+'-'+str(ir)+'_'+str(ic)+'.png'
        fullSaveFile = os.path.join(args.tmpDir, slideName, saveFile)    
        Image.fromarray(tile).save(fullSaveFile)
    
    print('compressing to tileDir')
    with tarfile.open(os.path.join(args.tileDir, slideName+'.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(args.tmpDir, slideName), arcname=slideName)
    
    
    # identify only the tumor tiles
    model = load_model(args.modelFile)
    
    # load tiles
    tileDir = os.path.join(args.tmpDir, slideName)
    print('loading images from', tileDir)
    x_test = []
    files = os.listdir(tileDir)
    files.sort()
    for i in files:
        x_test.append(np.array(image.load_img(os.path.join(tileDir, i))))
    x_test = np.array(x_test)
    x_test = x_test / 255
    print('x_test found with shape', x_test.shape)
    
    print('generating predictions')
    predictions = model.predict(x_test)
    
    # save predictions
    filenames = os.listdir(tileDir)
    filenames.sort()

    modelPredictions = pd.DataFrame()
    modelPredictions['filename'] = filenames
    modelPredictions['tumor'] = predictions
    modelPredictions.to_csv(os.path.join(args.hetDir, slideName+'_modelPredictions.csv'), index=False)
    
    modelPredictions['SlideID'] = [int(i.split('_')[0].lstrip('0')) for i in modelPredictions.filename]
    modelPredictions['RowIdx'] = [int(i.split('-')[1].split('_')[0]) for i in modelPredictions.filename]
    modelPredictions['ColIdx'] = [int(i.split('-')[1].split('_')[1].split('.')[0]) for i in modelPredictions.filename]
    
    # make composite assembly
    rowMax = modelPredictions['RowIdx'].max()
    rowMin = modelPredictions['RowIdx'].min()
    colMax = modelPredictions['ColIdx'].max()
    colMin = modelPredictions['ColIdx'].min()
    
    modelPredictions['RowIdx'] = modelPredictions['RowIdx'] - rowMin
    modelPredictions['ColIdx'] = modelPredictions['ColIdx'] - colMin

    # make assembly
    assembly = np.zeros((rowMax - rowMin + args.tileSize, 
                         colMax - colMin + args.tileSize, 3), dtype=np.uint8)
    probOver = np.zeros((rowMax - rowMin + args.tileSize, 
                         colMax - colMin + args.tileSize, 3), dtype=np.uint8)
    
    
    tileFiles = os.listdir(os.path.join(args.tmpDir, slideName))
    
    # paste each tile onto image
    for t in tileFiles:
        tile = Image.open(os.path.join(args.tmpDir, slideName, t))
        npc = np.array(tile)
        try:
            rowIdx = int(modelPredictions['RowIdx'][modelPredictions['filename'] == t])
            colIdx = int(modelPredictions['ColIdx'][modelPredictions['filename'] == t])
            assembly[rowIdx:rowIdx + args.tileSize, 
                     colIdx:colIdx +args.tileSize, ...] = npc[...,0:3]
            
            prob = float(modelPredictions['tumor'][modelPredictions['filename'] == t])
            probOver[rowIdx:rowIdx + args.tileSize, 
                     colIdx:colIdx +args.tileSize, :] = Thermal(prob)
        except:
            print('passing', t) 
            
            
    probOver = probOver[::10, ::10,:]
    assembly = assembly[::10, ::10,:]
    
    blendedImg = Image.blend(Image.fromarray(probOver),
                             Image.fromarray(assembly), alpha=0.5)
    
    blendedImg.save(os.path.join(args.assemblyDir, slideName+'_tumor_assembly.jpg'))
        
    # select tiles to save
    toDrop = predictions < args.tumorCutoff
    dropFiles = [files[i] for i in range(len(toDrop)) if toDrop[i]]
    
    for f in dropFiles:
        os.remove(os.path.join(args.tmpDir, slideName, f))
    
    with tarfile.open(os.path.join(args.tumorDir, slideName+'.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(args.tmpDir, slideName), arcname=slideName)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--slideFile', type=str, default=None)
    parser.add_argument('--tileDir', type=str, default='data/tiles')
    parser.add_argument('--tumorDir', type=str, default='data/tumor')
    parser.add_argument('--hetDir', type=str, default='data/HEtumor')
    parser.add_argument('--tmpDir', type=str, default='/mnt/scratch/tmpNEMO')
    parser.add_argument('--modelFile', type=str, default='src/hetumor.h5')
    parser.add_argument('--tumorCutoff', type=float, default=0.5)
    parser.add_argument('--assemblyDir', type=str, default='data/assemblies') 
    
    parser.add_argument('--tileSize', type=int, default=299)
    parser.add_argument('--cutoff', type=int, default=230, help='background level')
    parser.add_argument('--level', type=int, default=0, help='svs level param')
    parser.add_argument('--verbose', type=int, default=1, help='yell at console')
    parser.add_argument('--doNormalize', type=int, default=1, help='normalize tile stack')
    args = parser.parse_args()
    
   # args.slideFile='data/raw/LiverMets/0000002366_1026325.svs'
   # args.tmpDir='tmpDir'
    CropSlide(args)


