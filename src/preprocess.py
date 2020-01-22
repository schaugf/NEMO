import os
import argparse
import openslide
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from normalize_staining import NormalizeStaining
Image.MAX_IMAGE_PIXELS = None


def PreprocessSlide(slide_file, save_dir, tile_dim = 256, do_normalize = False,
                    image_level = 1, background_cutoff = 240):
    '''Preprocess whole slide image file
    
    Arguments:
        slide_file (string): file to open (svs or scn)
        save_dir (str): location to save tiles and annotations
        tile_dim (int): dimension of tile
        do_normalize (bool): execute normalization routine
        image_level (int): openslide zoom level 
        background_cutoff (int): mean tile 
          
    Returns:
        A tuple containing an npy array of size [n, d, d, c] of n tiles with
        height at width w and c color channels and a pandas dataframe
        containing coordinate annotations for each tile
    '''
    slide = openslide.OpenSlide(slide_file)
    slide_name = slide_file.split('/')[-1].split('.')[0]
    save_dir = os.path.join(save_dir, slide_name)
    os.makedirs(save_dir, exist_ok=True)   
    
    nc = (slide.level_dimensions[image_level][0] // tile_dim)
    nr = (slide.level_dimensions[image_level][1] // tile_dim)
    
    tile_stack, tile_coords = [], []
    for c in tqdm(range(nc)):
        for r in range(nr):
            ir = tile_dim * r * round(slide.level_downsamples[image_level])
            ic = tile_dim * c * round(slide.level_downsamples[image_level])
            tile = slide.read_region((ic, ir), image_level, 
                                     (tile_dim, tile_dim))
            tile = np.array(tile)[...,0:3]
            
            if (tile[...,1].mean() < background_cutoff) & (tile.mean() > 40):
                tile_stack.append(tile)
                tile_coords.append((ir, ic))
    
    if do_normalize:
        try:
            normTiles, H, E = NormalizeStaining(np.vstack(tile_stack))
            tiles = []
            for i in range(len(tile_stack)):
                tile = normTiles[(i*tile_dim):((i*tile_dim) + 
                                  tile_dim),...]
                tiles.append(tile)
            tile_stack = tiles
        except:
            print('failure to normalize')
    
    # generate and save assembly figure
    tile_coords = pd.DataFrame(tile_coords)
    scale_dim = round(slide.level_downsamples[image_level])
    tile_coords[0] = (tile_coords[0] - tile_coords[0].min()) // scale_dim
    tile_coords[1] = (tile_coords[1] - tile_coords[1].min()) // scale_dim
    
    save_image_dim = (tile_coords[0].max() + tile_dim,
                      tile_coords[1].max() + tile_dim)
    grid_image = np.zeros(save_image_dim + (3,), dtype=np.uint8) 
    for i in range(len(tile_stack)):
        x, y = tile_coords.iloc[i]
        grid_image[x:(x+tile_dim),y:(y+tile_dim), :] = tile_stack[i]
    grid_image = Image.fromarray(grid_image[::10,::10,:])
    
    # save tile assemblies
    grid_image.save(os.path.join(save_dir, slide_name + '.jpg'))
    
    # save coordinates
    tile_coords.to_csv(os.path.join(save_dir, slide_name+'.csv'), 
                       header=None, index=None)
    # save numpy array to disk
    tile_stack = np.stack(tile_stack)
    np.save(os.path.join(save_dir, slide_name + '.npy'), tile_stack)
    
    return tile_stack, tile_coords

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_file', type=str, 
                        default='../data/raw/primary/0000000044_2014-04-11_09_08_20.scn')
    parser.add_argument('--save_dir', type=str, default='../data/processed')
    parser.add_argument('--tile_dim', type=int, default=256)
    parser.add_argument('--background_cutoff', type=int, default=200)
    parser.add_argument('--image_level', type=int, default=1)
    parser.add_argument('--do_normalize', type=int, default=False)
    args = parser.parse_args()
    
    PreprocessSlide(slide_file = args.slide_file, 
                    save_dir = args.save_dir,
                    tile_dim = args.tile_dim,
                    background_cutoff = args.background_cutoff,
                    image_level = args.image_level, 
                    do_normalize = args.do_normalize)
