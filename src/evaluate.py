import os
import argparse
import tarfile
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


def Thermal(val):
    ''' convert single value [0,1] to 3-channel thermal color map
    '''
    val = val * 255 * 3
    colc = np.zeros((3))
    colc[0] = max(0, min(val, 255))         # red
    colc[1] = max(0, min(val-255, 255))     # yellow
    colc[2] = max(0, min(val-255-255, 255)) # white
    return colc
        

def MakeOverlay(args):
    ''' overlay probability map onto reassembly figure
    * loads assembly file
    * takes model_results, creates colormap
    '''
    if args.verbose:
        print('generating overlay')
    slideID = args.slideFile.split('/')[-1].split('.tar.gz')[0]
    
    modelPredictions = pd.read_csv(os.path.join(args.modelDir, 'modelPredictions', slideID+'_modelPredictions.csv'))
    
    predictionNames = list(modelPredictions)[1:]  # this could be wrong (or how I assign names originally)
    
    # preprocessing
    modelPredictions['SlideID'] = [int(i.split('_')[0].lstrip('0')) for i in modelPredictions.filename]
    modelPredictions['RowIdx'] = [int(i.split('-')[1].split('_')[0]) for i in modelPredictions.filename]
    modelPredictions['ColIdx'] = [int(i.split('-')[1].split('_')[1].split('.')[0]) for i in modelPredictions.filename]
    
    # create blank assembly
    rowMax = modelPredictions['RowIdx'].max()
    rowMin = modelPredictions['RowIdx'].min()
    colMax = modelPredictions['ColIdx'].max()
    colMin = modelPredictions['ColIdx'].min()
    
    modelPredictions['RowIdx'] = modelPredictions['RowIdx'] - rowMin
    modelPredictions['ColIdx'] = modelPredictions['ColIdx'] - colMin

    assembly = np.zeros((rowMax - rowMin + args.tileSize, 
                         colMax - colMin + args.tileSize, 3), dtype=np.uint8)
    
    tileFiles = os.listdir(os.path.join(args.scratchDir, slideID))

    # fill new assembly for each pass through the loop
    for t in tileFiles:
        tile = Image.open(os.path.join(args.scratchDir, slideID, t))
        npc = np.array(tile) 
        # parse row, col indexes from filename
        rowIdx = int(modelPredictions['RowIdx'][modelPredictions['filename'] == t])
        colIdx = int(modelPredictions['ColIdx'][modelPredictions['filename'] == t])
        assembly[rowIdx:rowIdx + args.tileSize, 
                 colIdx:colIdx +args.tileSize, ...] = npc[...,0:3]
        
    assembly = assembly[::10, ::10,:]
    assemblyImage = Image.fromarray(assembly)
    
    
    # generate heat map for each label
    for p in predictionNames:
        overlay = np.zeros((rowMax - rowMin + args.tileSize, 
                            colMax - colMin + args.tileSize, 3), dtype=np.uint8)        
        
        for i in range(modelPredictions.shape[0]):
            valcol = Thermal(modelPredictions[p][i])
            
            ridx = modelPredictions.RowIdx[i]
            cidx = modelPredictions.ColIdx[i]
            # save blank boarders
            overlay[ridx+10:ridx+args.tileSize-10, cidx+10:cidx+args.tileSize-10, [0,1,2]] = valcol
            
        # blend and save
        overlay = overlay[::10, ::10, :]
        colmat = Image.fromarray(overlay)
        overlayImage = Image.blend(assemblyImage, colmat, alpha=0.5)
        
        os.makedirs(os.path.join(args.modelDir, 'overlays'), exist_ok=True)
        os.makedirs(os.path.join(args.modelDir, 'overlays', str(slideID)), exist_ok=True)
        overlayImage.save(os.path.join(args.modelDir, 'overlays', str(slideID), str(slideID)+'_'+p+'_prob_overlay_test.jpg'))


def EvaluateModel(args):
    ''' load and evaluate keras model on new input svs file
    '''
    # reload model 
    model = load_model(os.path.join(args.modelDir, 'model.h5'))

    # point to crop directory
    slideID = args.slideFile.split('/')[-1].split('.tar.gz')[0]
    fullTileDir = os.path.join(args.scratchDir, slideID)

    # load images
    if args.verbose:
        print('loading images from', fullTileDir)    
    x_test = []
    for i in os.listdir(fullTileDir):
        x_test.append(np.array(image.load_img(os.path.join(fullTileDir, i))))
    x_test = np.array(x_test)
    x_test = x_test / 255
        # load classnames
    classNames = list(pd.read_csv(os.path.join(args.modelDir, 'classIDs.csv'), header=None)[0])

    # generate predictions
    if args.verbose:
        print('x_test found with shape', x_test.shape)
        print('generating predictions')
    
    predictions = model.predict(x_test)

    # save predictions
    results = pd.DataFrame()
    results['filename'] = os.listdir(fullTileDir)
    print('classes found:', len(classNames))
    for i in range(len(classNames)):
        results[classNames[i]] = predictions[:,i]
    os.makedirs(os.path.join(args.modelDir, 'modelPredictions'), exist_ok=True)
    results.to_csv(os.path.join(args.modelDir, 'modelPredictions', slideID+'_modelPredictions.csv'), index=False)
         
    
def ExtractTiles(args):
    ''' extract tar arxiv of tiles to scratchDir
    '''
    os.makedirs(args.scratchDir, exist_ok=True)
    tar = tarfile.open(args.slideFile)
    if args.verbose:
        print('extracting', args.slideFile, 'to', args.scratchDir)
    tar.extractall(path=args.scratchDir)
    tar.close()


if __name__=='__main__':
        
    parser = argparse.ArgumentParser(description='evaluate NEMO model')
    parser.add_argument('--slideFile',  type=str, default=None,  help='input svs.tar.gz file')
    parser.add_argument('--assemblyDir',type=str, default='data/assemblies', help='assembly dir')
    parser.add_argument('--scratchDir', type=str, default='/mnt/scratch/tmp_nemo', help='scratch dir')
    parser.add_argument('--modelDir',    type=str, default='results', help='where to save encodings')
    parser.add_argument('--tileSize',   type=int, default=299,  help='size of image')
    parser.add_argument('--nChannel',   type=int, default=3,    help='number of channels')
    parser.add_argument('--verbose', type=int, default=1, help='talky talky')
    args = parser.parse_args()
    
    #args.slideFile = 'data/tiles/0000002366_1026325.tar.gz'
    #args.scratchDir = 'tmpDir'
    #args.modelDir='results/190418/'
    
    ExtractTiles(args)
    #EvaluateModel(args)
    MakeOverlay(args)
    
