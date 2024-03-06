# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
import rasterio
from rasterio.plot import show
from tqdm import *
import numpy as np
from geo_func import search_grids, resample
from rasterio.enums import Resampling


def ExtractSrtmDEM(grid_shp, plot=True):
    # read dem data
    SrtmDEM_path = "E:/data/LULC/DEM/SRTM/US/Combine/srtm_11_03.tif"
    SrtmDEM = rasterio.open(SrtmDEM_path)

    # downscale
    downscale_factor = 1/8
    SrtmDEM_downscale = SrtmDEM.read(1,
                                     out_shape=(int(SrtmDEM.height * downscale_factor),
                                                int(SrtmDEM.width * downscale_factor)
                                                ),
                                     resampling=Resampling.average
                                     )

    transform = SrtmDEM.transform * SrtmDEM.transform.scale(
        (SrtmDEM.width / SrtmDEM_downscale.shape[-1]),
        (SrtmDEM.height / SrtmDEM_downscale.shape[-2])
    )

    # SrtmDEM grids
    ul = transform * (0, 0)
    lr = transform * (SrtmDEM_downscale.shape[1], SrtmDEM_downscale.shape[0])

    SrtmDEM_downscale_x = np.linspace(ul[0], lr[0], SrtmDEM_downscale.shape[1])
    SrtmDEM_downscale_y = np.linspace(ul[1], lr[1], SrtmDEM_downscale.shape[0])

    # grids in grid_shp
    grid_shp_x = grid_shp.point_geometry.x.values
    grid_shp_y = grid_shp.point_geometry.y.values

    # search SrtmDEM grids for each grid in grid_shp
    searched_grids_index = search_grids.search_grids_radius_rectangle(
        dst_lat=grid_shp_y, dst_lon=grid_shp_x, src_lat=SrtmDEM_downscale_y, src_lon=SrtmDEM_downscale_x,
        lat_radius=0.125, lon_radius=0.125)

    # resample for mean
    SrtmDEM_mean_Value = []
    for i in range(len(searched_grids_index)):
        searched_grid_index = searched_grids_index[i]
        searched_grid_lat = [SrtmDEM_downscale_y[searched_grid_index[0][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_lon = [SrtmDEM_downscale_x[searched_grid_index[1][j]] for j in range(len(searched_grid_index[0]))]
        searched_grid_data = [SrtmDEM_downscale[searched_grid_index[0][j], searched_grid_index[1][j]]
                              for j in range(len(searched_grid_index[0]))]  # index: (lat, lon), namely (row, col)
        dst_data = resample.resampleMethod_GeneralFunction(
            searched_grid_data, searched_grid_lat, searched_grid_lon, None, None,
            general_function=np.mean, missing_value=32767)
        SrtmDEM_mean_Value.append(dst_data)

    # set missing_value as none
    SrtmDEM_mean_Value = np.array(SrtmDEM_mean_Value)
    SrtmDEM_mean_Value[SrtmDEM_mean_Value == 32767] = None

    # save in grid_shp
    grid_shp["SrtmDEM_mean_Value"] = SrtmDEM_mean_Value

    # plot
    if plot:
        show(SrtmDEM_downscale, transform=transform)

    SrtmDEM.close()

    return grid_shp


def readSrtmDEMAsPCRasterMap():
    # read dem data
    SrtmDEM_path = "E:/data/LULC/DEM/SRTM/US/Combine/srtm_11_03.tif"
    SrtmDEM = rasterio.open(SrtmDEM_path)
