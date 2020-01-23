import os
import tarfile
import argparse
import shutil
import pandas as pd
from tqdm import tqdm

def SortSlides(args):
    ''' sort .tar.gz files into directory structure by annotated group into tempDir
    '''
    
    os.makedirs(args.tmpDir, exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, 'NEMO'), exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, 'NEMO', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, 'NEMO', 'val'), exist_ok=True)

    os.makedirs(os.path.join(args.tmpDir, 'NEMO3c'), exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, 'NEMO3c', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.tmpDir, 'NEMO3c', 'val'), exist_ok=True)

    annotations = pd.read_csv(args.annotFile)
    classGroups = ['colonic_adenocarcinoma',
                   'neuroendocrine_carcinoma',
                   'gastrointestinal_stromal']
    
    for fname in tqdm(os.listdir(args.tileDir), desc='compiling'):
        slideName = fname.split('/')[-1].split('.')[0]
        slideID = int(slideName.split('_')[0].lstrip('0'))
        group = annotations['Group'][annotations['Slide.ID'] == slideID].to_string().split('    ')[1]
        split = annotations['split'][annotations['Slide.ID'] == slideID].to_string().split('    ')[1]
        
        if ((split != 'test') & (group != 'unused')):
            outputDir = os.path.join(args.tmpDir, 'NEMO', split, group)
            os.makedirs(os.path.join(outputDir), exist_ok=True)
            # copy contents of tar file
            tar = tarfile.open(os.path.join(args.tileDir, fname))
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member,outputDir)
            tar.close()

        # for three classes
        if ((split != 'test') & (group in classGroups)):
            outputDir = os.path.join(args.tmpDir, 'NEMO3c', split, group)
            os.makedirs(outputDir, exist_ok=True)
            tar = tarfile.open(os.path.join(args.tileDir, fname))
            for member in tar.getmembers():
                if member.isreg():
                    member.name=os.path.basename(member.name)
                    tar.extract(member, outputDir)
            tar.close()
        
    print('compressing full dataset')
    with tarfile.open(os.path.join(args.saveDir, 
                                   'TrainingDataTumorOnly.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(args.tmpDir, 'NEMO'), arcname='NEMO')

    print('compressing 3 class dataset')
    with tarfile.open(os.path.join(args.saveDir,
                                   'TrainingDataTumorOnly3Class.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(args.tmpDir, 'NEMO3c'), arcname='NEMO')

    print('removing temporary directory')
    shutil.rmtree(args.tmpDir)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileDir', type=str, default=None)
    parser.add_argument('--tmpDir', type=str, default=None)
    parser.add_argument('--saveDir', type=str, default=None)
    parser.add_argument('--annotFile', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    
    SortSlides(args)

