import os
import argparse
import tarfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


def Thermal(val):
    ''' convert prob value to color map '''
    val = val * 255 * 3
    colc = np.zeros((3))
    colc[0] = max(0, min(val, 255))
    colc[1] = max(0, min(val-255, 255))
    colc[2] = max(0, min(val-255-255, 255))
    return colc


def MakeOverlay(args):
    ''' overlay probability map onto reassembly figure '''
    
    print('generating overlay')
    os.makedirs(os.path.join(args.saveDir, 'overlays'), exist_ok=True)
    slideID = args.tileFile.split('/')[-1].split('.tar.gz')[0]    
    modelPredictions = pd.read_csv(os.path.join(args.saveDir, 
                                                'modelPredictions', 
                                                slideID+'_modelPredictions.csv'))
        
    # preprocessing
    modelPredictions['SlideID'] = [int(i.split('_')[0].lstrip('0')) for i in modelPredictions.filename]
    modelPredictions['RowIdx'] = [int(i.split('-')[1].split('_')[0]) for i in modelPredictions.filename]
    modelPredictions['ColIdx'] = [int(i.split('-')[1].split('_')[1].split('.')[0]) for i in modelPredictions.filename]
    
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
    confOver = np.zeros((rowMax - rowMin + args.tileSize, 
                         colMax - colMin + args.tileSize, 3), dtype=np.uint8)
    
    tileFiles = os.listdir(os.path.join(args.scratchDir, slideID, 'tiles'))


    # paste each tile onto image
    for t in tqdm(tileFiles, desc='generating tile overlays'):
        tile = Image.open(os.path.join(args.scratchDir, slideID, 'tiles', t))
        
        npc = np.array(tile)
        
        try:
            rowIdx = int(modelPredictions['RowIdx'][modelPredictions['filename'] == t])
            colIdx = int(modelPredictions['ColIdx'][modelPredictions['filename'] == t])
            assembly[rowIdx:rowIdx + args.tileSize, 
                     colIdx:colIdx +args.tileSize, ...] = npc[...,0:3]
            
            prob = float(modelPredictions['tumor'][modelPredictions['filename'] == t])
            probOver[rowIdx:rowIdx + args.tileSize, 
                     colIdx:colIdx +args.tileSize, :] = Thermal(prob)
            
            # set confusion matrix (white = true, blue = FP, RED=FN)
            truth = str.split(str.split(t, '-')[-1], '.')[0]
            pred = ['Tumor' if prob > 0.5 else 'NonTumor'][0]
            if truth == pred:
                confOver[rowIdx:rowIdx + args.tileSize, 
                         colIdx:colIdx +args.tileSize, :] = [255,255,255]
            elif (truth == 'Tumor') & (pred == 'NonTumor'):
                # false positive
                confOver[rowIdx:rowIdx + args.tileSize, 
                         colIdx:colIdx +args.tileSize, :] = [0,0,255]
            elif (truth == 'NonTumor') & (pred == 'Tumor'):
                # false negative
                confOver[rowIdx:rowIdx + args.tileSize, 
                         colIdx:colIdx +args.tileSize, :] = [255,0,0]
        except:
            print('passing', t)
    
    
    probOver = probOver[::10, ::10]
    confOver = confOver[::10, ::10]
    assembly = assembly[::10, ::10,:]
    
    blendedImg = Image.blend(Image.fromarray(probOver),
                             Image.fromarray(assembly), alpha=0.5)
    
    blendedImg.save(os.path.join(args.saveDir, 'overlays', slideID+'_assembly.jpg'))
    
    confusionImg = Image.blend(Image.fromarray(confOver),
                               Image.fromarray(assembly), alpha=0.5)
    
    confusionImg.save(os.path.join(args.saveDir, 'overlay', slideID+'_confusionAssembly.jpg'))
    

def EvaluateModel(args):
    ''' load and evaluate keras model on new input svs file '''
    # reload model 
    model = load_model(args.modelFile)

    # point to crop directory
    slideID = args.tileFile.split('/')[-1].split('.tar.gz')[0]
    fullTileDir = os.path.join(args.scratchDir, slideID, 'tiles')

    # load images
    print('loading images from', fullTileDir)    
    x_test = []
    files = os.listdir(fullTileDir)
    files.sort()
    for i in files:
        x_test.append(np.array(image.load_img(os.path.join(fullTileDir, i))))
    x_test = np.array(x_test)
    x_test = x_test / 255
    print('x_test found with shape', x_test.shape)
    
    # generate predictions
    print('generating predictions')
    predictions = model.predict(x_test)
    
    # save predictions
    results = pd.DataFrame()
    filenames = os.listdir(fullTileDir)
    filenames.sort()

    results['filename'] = filenames
    results['tumor'] = predictions
    
    os.makedirs(os.path.join(args.saveDir, 'modelPredictions'), exist_ok=True)
    results.to_csv(os.path.join(args.saveDir, 'modelPredictions', slideID+'_modelPredictions.csv'), index=False)
     

def ExtractTiles(args):
    ''' extract tar arxiv of tiles to scratchDir '''
    os.makedirs(args.scratchDir, exist_ok=True)
    print('extracting', args.tileFile, 'to', args.scratchDir)
    tar = tarfile.open(args.tileFile)
    tar.extractall(path=args.scratchDir)
    tar.close()
    print('extraction complete')


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='evaluate NEMO model')
    parser.add_argument('--tileFile',type=str, default='data/tiles/0000003154_1009835.tar.gz', help='tile.tar.gz file')
    parser.add_argument('--saveDir',type=str, default='results/', help='save dir')
    parser.add_argument('--modelFile',  type=str, default='results', help='where to save encodings')
    parser.add_argument('--scratchDir', type=str, default='/mnt/scratch/tmp_nemo', help='scratch dir')
    parser.add_argument('--tileSize',   type=int, default=299,  help='size of image')
    parser.add_argument('--nChannel',   type=int, default=3,    help='number of channels')
    parser.add_argument('--nClasses',   type=int, default=2,    help='num class predictions')
    parser.add_argument('--sigma',      type=float, default=0.0, help='smoothing sigma')
    args = parser.parse_args()
    
    
    args.tileFile='data/tiles/0000003154_1009835.tar.gz'
    args.modelFile='results/190402/model.h5'
    args.saveDir='results/190402'
    args.scratchDir='tmpDir'
    
    ExtractTiles(args)
    EvaluateModel(args)
    MakeOverlay(args)
    
