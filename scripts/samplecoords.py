import numpy as np
import os

path = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed/val"

N = 100000

def get_mask(coord):
    mask = np.ones(len(coord), dtype=bool)
    if len(mask) <=N:
        return mask 
    mid_y = 0.5
    x_min = coord[:,0].min()
    x_max = coord[:,0].max()
    y_min = coord[:,1].min()
    y_max = coord[:,1].max()
    while mid_y >= 0.1:
        mid_x = 0.5
        while mid_x >= 0.1:
            for x_smaller in [0,1,2]:
                if x_smaller==0: # x <= x_mid
                    x_mask = coord[:,0] <= (x_min + (x_max - x_min) * mid_x) 
                elif x_smaller == 1: # x > x_mid
                    x_mask = coord[:,0] >  (x_max - (x_max - x_min) * mid_x)
                else:
                    x_mask = np.ones(len(coord), dtype=bool)
                for y_smaller in [0,1,2]:
                    if y_smaller==0: # y < y_mid
                        y_mask = coord[:,1] <= (y_min + (y_max - y_min) * mid_y) 
                    elif y_smaller==1: # y > y_mid
                        y_mask = coord[:,1] > (y_max - (y_max - y_min) * mid_y)
                    elif x_smaller==2: # can't ignore both
                        continue
                    else:
                        y_mask = np.ones(len(coord), dtype=bool)
                    mask = x_mask & y_mask
                    if N * 0.6 <= mask.sum() < N:
                        return mask
            mid_x-=0.1
        mid_y -=0.1
    print("Failed")
    return mask

for scene in os.listdir(path):
    coord_file = os.path.join(path,scene,"coord.npy")
    coord = np.load(coord_file)
    mask = get_mask(coord)
    #print(f"Original: {len(mask)}, New: {mask.sum()}, Reduction: {mask.sum()/len(mask)},Threshold: {mask.sum() <= 100000}")
