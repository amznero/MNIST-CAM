import os
import imageio as io
from tqdm import tqdm

def visulization(index):
	images_prefix = 'cam_{}_{}.jpg'
	with io.get_writer('gifs/{}.gif'.format(index), mode='I', duration=0.5) as writer:
	    for file in tqdm(xrange(1, 11)):
	        image = io.imread('result/'+images_prefix.format(index, file))
	        writer.append_data(image)

if __name__ == '__main__':
	for idx in xrange(10):
		visulization(idx)
