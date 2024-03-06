# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from dataPreprocess_CAMELS import dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing
from dataPreprocess_CAMELS import dataProcess_CAMELS_read_basin_grid
from dataPreprocess_CAMELS import setHomePath

# upland basin: fs/frac_snow > 0.15 (268 in all basins)




if __name__ == "__main__":
    root, home = setHomePath(root="E:")
    subdir = "basinClassification"
    
    # dpc_all = dataProcess_CAMELS_read_basin_grid(home, subdir)
    dpc = dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(home, subdir) # 552
    dpc(plot=False)
    print(dpc.basin_shp)
    
    
    
    
    