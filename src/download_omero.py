""" download images from OMERO server with list of image IDs
"""

import os
import argparse
import getpass
import numpy as np
import pandas as pd
#from scipy.misc import imresize
from omero.gateway import BlitzGateway

parser = argparse.ArgumentParser(description='Download from OMERO server') 
parser.add_argument('-s', type=str, default='images',
                    help='save directory')
parser.add_argument('-i', type=str, default='all_meta.csv',
                    help='csv datafile with imageID column')
parser.add_argument('-x', type=int, default=0,
                    help='save image width resolution')
parser.add_argument('-y', type=int, default=0,
                    help='save image height resolution')
parser.add_argument('-c', type=int, default=0,
                    help='save resolution (255 for 8 bit)')
parser.add_argument('-o', type=str, default='omero.ohsu.edu',
                    help='OMERO host')
args = parser.parse_args()

un = raw_input('enter your OMERO username for %s: ' % args.o)
pw = getpass.getpass('enter your OMERO password for %s: ' % args.o)

if os.path.exists(args.s) == False:
    os.makedirs(args.s)

df = pd.read_csv(args.i, usecols = ['ImageID'])

conn = BlitzGateway(un, pw, host=args.o, port=4064)
conn.connect()

for imageID in df.ImageID:
    print('downloading %s ' % imageID)
    img = conn.getObject("Image", imageID)
    pixels = img.getPrimaryPixels()
    channels = []
    for i in range(img.getSizeC()):
        ch = np.array(pixels.getPlane(0,i,0), dtype='f')
        i#f (args.x != 0) & (args.y != 0):
            #ch = imresize(ch, (args.x, args.y))
        if (args.c != 0):
            ch = (ch/np.amax(ch))*args.c
        channels.append(ch)
    plane = np.dstack(channels)
    np.save(os.path.join(args.s, str(imageID)), plane)

print('done!')
