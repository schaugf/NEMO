import os
import tarfile
import argparse
import shutil
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--saveDir', type=str,
                    default='data')
parser.add_argument('--tileDir', type=str, 
                    default='data/tiles')
parser.add_argument('--tmpDir', type=str,
                     default='tmpSort')
parser.add_argument('--splitTable', type=str,
                    default='data/splitTable.csv')
args = parser.parse_args()


if not os.path.isdir(args.tmpDir):
    os.makedirs(args.tmpDir)
    os.makedirs(os.path.join(args.tmpDir, 'train'))
    os.makedirs(os.path.join(args.tmpDir, 'val'))
    os.makedirs(os.path.join(args.tmpDir, 'train', 'tumor'))
    os.makedirs(os.path.join(args.tmpDir, 'train', 'nontumor'))
    os.makedirs(os.path.join(args.tmpDir, 'val', 'tumor'))
    os.makedirs(os.path.join(args.tmpDir, 'val', 'nontumor'))
    os.makedirs(os.path.join(args.tmpDir, 'hold'))
    
    
splitDF = pd.read_csv(args.splitTable)

for fname in tqdm(os.listdir(args.tileDir), 'extracting'):
    slideName = str.split(fname, '.')[0]    
    split = list(splitDF['split'][splitDF['slideFile'] == slideName+'.svs'])[0]
    if split != 'test':
        tar = tarfile.open(os.path.join(args.tileDir, fname))
        tar.extractall(os.path.join(args.tmpDir, 'hold'))
        slideTileDir = os.path.join(args.tmpDir, 'hold', slideName, 'tiles')
        
        for tile in os.listdir(slideTileDir):
            if 'NonTumor' in tile:
                outputDir = os.path.join(args.tmpDir, split, 'nontumor')
            else:
                outputDir = os.path.join(args.tmpDir, split, 'tumor')
            shutil.move(os.path.join(slideTileDir, tile), outputDir)
        
        shutil.rmtree(os.path.join(args.tmpDir, 'hold', slideName))

# remove hold dir
shutil.rmtree(os.path.join(args.tmpDir, 'hold'))

with tarfile.open(os.path.join(args.saveDir, 'HEtumorTrainingData.tar.gz'), 'w:gz') as tar:
    tar.add(args.tmpDir, arcname='HEtumor')

print('removing temporary directory')
shutil.rmtree(args.tmpDir)

print('done')
