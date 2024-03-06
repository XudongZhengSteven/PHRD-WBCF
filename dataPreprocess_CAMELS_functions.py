# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from typing import Any
import numpy as np
import pandas as pd
from plot_func import create_gdf
import geopandas as gpd
from shapely import geometry
from matplotlib import pyplot as plt
from copy import copy
from WaterBudgetClosure.dataPreprocess_UMDLandCover import ExtractLCForEachBasinGrids
from WaterBudgetClosure.dataPreprocess_HWSD import ExtractHWSDSoilData, inquireHWSDSoilData
from WaterBudgetClosure.dataPreprocess_dem import ExtractSrtmDEM
from WaterBudgetClosure.dataPreprocess_GLEAM_E import ExtractGLEAM_E_daily
from WaterBudgetClosure.dataPreprocess_CAMELS_ForcingDaymet import ExtractForcingDaymet
from WaterBudgetClosure.dataPreprocess_TRMM_P import ExtractTRMM_P
from WaterBudgetClosure.dataPreprocess_ERA5_SM import ExtractERA5_SM
from WaterBudgetClosure.dataPreprocess_GlobalSnow_SWE import ExtractGlobalSnow_SWE, aggregate_func_SWE_axis1
from WaterBudgetClosure.dataPreprocess_GLDAS import ExtractGLDAS
import matplotlib.colors as mcolors
from matplotlib import cm
from datetime import datetime
import pickle
from tqdm import *
from functools import partial

import warnings
warnings.filterwarnings('ignore')
# import imp
# from importlib import reload
# reload(dataPreprocess_HWSD)


"""
independent function structure:
    general:
        setHomePath(home)
    
    read grids and HCDN shps function:
        readGridShp(home): read grid.shp and grid_label.shp file
            return grid_shp  # pd.Dataframe
        readHCDNShp(home): read HCDN.shp file
            return HCDN_shp  # pd.Dataframe
        createBoundaryShp(grid_shp): create boundary based on the grid_shp max/min lon/lat
            return boundary_shp, boundary_x_y # [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
    
    read data into grids:
        readSrtmDEMIntoGrids(grid_shp, plot): read SrtmDEM into grid_shp based on ExtractStrmDEM()
            return grid_shp
        readUMDLandCoverIntoGrids(grid_shp): read UMDLandCover into grid_shp based on ExtractLCForEachBasinGrids()
            return grid_shp
        readHWSDSoilDataIntoGirds(grid_shp, boundary_shp, plot): read HWSDSoilData into grid_shp based on ExtractHWSDSoilData()
            return grid_shp
        readGLEAMEDailyIntoGrids(grid_shp, period, var_name): read GLEAMDaily into grid_shp based on ExtractGLEAM_E_daily()
            return grid_shp
    
    read data into basins:
        readStreamflow(home): read CAMELS usgs_streamflow data
            return fns, fpaths, usgs_streamflow, streamflow_id  # list of pd.Dataframe
        checkStreamflowMissing(usgs_streamflow_, date_period): check usgs_streamflow to avoid missing data ('M in 1980-2010' or less length)
            return judgement, reason # bool, str
        removeStreamflowMissing(fns, fpaths, usgs_streamflow, date_period): remove stations containing missing data based on checkStreamflowMissing()
            return fns, fpaths, usgs_streamflow, streamflow_id, remove_files_Missing  # list of pd.Dataframe removed missing stations
        readStreamflowIntoBasins(basinShp, streamflow_id, usgs_streamflow): read usgs_streamflow into basinShp["streamflow"]
            return basinShp
        readForcingDaymet(home): read ForcingDaymet files
            return fns, fpaths, forcingDaymet, forcingDaymetGaugeAttributes  # list of str, str, pd.Dataframe, set("latitude", "elevation", "basinArea", "gauge_id")
        readForcingDaymetIntoBasins(forcingDaymet, forcingDaymetGaugeAttributes, basinShp, read_dates, read_keys): read ForcingDaymet data into basinShp
            return basinShp
        readBasinAttribute(home): read camels_attributes_v2.0 into a set
            return BasinAttribute  # set
    
    select basin function:
        removeBasinBasedOnStreamflowMissing(basinShp, streamflow_id_removeMissing): remove basin from basinShp based on streamflow_id_removeMissing
            return basinShp
        selectBasinBasedOnArea(basinShp, min_area, max_area): select basin from basinShp based on min_area <= area <= max_area
            return basinShp
        selectBasinBasedOnStreamflowWithZero(basinShp, usgs_streamflow, streamflow_id, zeros_min_num=100): select basin from basinShp based on numbers of zero value in usgs_streamflow > zero_min_num
            return basinShp
        selectBasinBasedOnAridity() # not yet realized
        selectBasinBasedOnElevSlope()  # not yet realized

    Intersects grids with HCDN:
        IntersectsGridsWithHCDN(grid_shp, basinShp): intersect grid_shp with basinShp to get intersects_grids (all grids intersected) and basinShp["intersects_grids"] (group for each basin)
            return basinShp, intersects_grids
        
    aggregate grid to basins function:
        aggregate_GLEAMEDaily(basin_shp): aggregate basinShp["intersects_grids"]["GLEAM_v3.8a_daily_E"] into basinShp["aggregated_GLEAM_v3.8a_daily_E"]
            return basin_shp
        aggregate_GLEAMEpDaily(basin_shp): aggregate basinShp["intersects_grids"]["GLEAM_v3.8a_daily_Ep"] into basinShp["aggregated_GLEAM_v3.8a_daily_Ep"]
            return basin_shp
    
    plot function:
        plotShp(basinShp_original, basinShp, grid_shp, intersects_grids,
                boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                fig=None, ax=None): plot basinShp, grid_shp and intersects_grids, it can receive existing fig and ax
            return fig, ax
        plotLandCover(basinShp_original, basinShp, grid_shp, intersects_grids,
                  boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                  fig=None, ax=None): plot LandCover
            return fig, ax
        plotHWSDSoilData(basinShp_original, basinShp, grid_shp, intersects_grids,
                     boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                     fig=None, ax=None, fig_T=None, ax_T=None, fig_S=None, ax_S=None): plot HWSDSoilData
            return fig, ax, fig_S, ax_S, fig_T, ax_T
        plotStrmDEM(basinShp_original, basinShp, grid_shp, intersects_grids,
                boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                fig=None, ax=None): plot StrmDEM
            return fig, ax
            
    other function:
        checkGaugeBasin(basinShp, usgs_streamflow, BasinAttribute, forcingDaymetGaugeAttributes): print information about gauge id in different variables
        exportToCsv(basin_shp, fpath_dir): save data into Csv format
    
"""


def setHomePath(root="E:"):
    home = f"{root}/data/hydrometeorology/CAMELS"
    return root, home

# ------------------------ read grids and HCDN shps function ------------------------


def readGridShp(home):
    grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
    grid_shp_label = gpd.read_file(grid_shp_label_path)
    print(grid_shp_label)

    grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
    grid_shp = gpd.read_file(grid_shp_path)
    print(grid_shp)

    # combine grid_shp_lable into grid_shp
    grid_shp["point_geometry"] = grid_shp_label.geometry

    return grid_shp


def readHCDNShp(home):
    # read data: HCDN
    HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    HCDN_shp = gpd.read_file(HCDN_shp_path)
    HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
    print(HCDN_shp)
    return HCDN_shp


def createBoundaryShp(grid_shp):
    cgdf = create_gdf.CreateGDF()
    boundary_x_min = min(grid_shp["point_geometry"].x)
    boundary_x_max = max(grid_shp["point_geometry"].x)
    boundary_y_min = min(grid_shp["point_geometry"].y)
    boundary_y_max = max(grid_shp["point_geometry"].y)
    boundary_shp = cgdf.createGDF_polygons(lon=[[boundary_x_min, boundary_x_max, boundary_x_max, boundary_x_min]],
                                           lat=[[boundary_y_max, boundary_y_max, boundary_y_min, boundary_y_min]],
                                           crs=grid_shp.crs)
    boundary_x_y = [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]

    return boundary_shp, boundary_x_y


# ------------------------ read data into grids ------------------------
def readSrtmDEMIntoGrids(grid_shp, plot=False):
    grid_shp = ExtractSrtmDEM(grid_shp, plot=plot)
    return grid_shp


def readUMDLandCoverIntoGrids(grid_shp):
    grid_shp = ExtractLCForEachBasinGrids(grid_shp)
    return grid_shp


def readHWSDSoilDataIntoGirds(grid_shp, boundary_shp, plot=False):
    grid_shp = ExtractHWSDSoilData(grid_shp, boundary_shp, plot=plot)
    MU_GLOBALS = grid_shp["HWSD_BIL_Value"].values
    T_USDA_TEX_CLASS, S_USDA_TEX_CLASS = inquireHWSDSoilData(MU_GLOBALS)
    grid_shp["T_USDA_TEX_CLASS"] = T_USDA_TEX_CLASS
    grid_shp["S_USDA_TEX_CLASS"] = S_USDA_TEX_CLASS

    # set None
    grid_shp.loc[grid_shp["HWSD_BIL_Value"] == 0, "HWSD_BIL_Value"] = None
    return grid_shp


def readGLEAMEDailyIntoGrids(grid_shp, period, var_name):
    grid_shp = ExtractGLEAM_E_daily(grid_shp, period, var_name)
    return grid_shp


def readTRMMPIntoGrids(grid_shp, period, var_name):
    grid_shp = ExtractTRMM_P(grid_shp, period, var_name)
    return grid_shp


def readERA5_SMIntoGrids(grid_shp, period, var_name):
    grid_shp = ExtractERA5_SM(grid_shp, period, var_name)
    return grid_shp

def readGlobalSnow_SWEIntoGrids(grid_shp, period, var_name):
    grid_shp = ExtractGlobalSnow_SWE(grid_shp, period, var_name)
    return grid_shp
    
def readGLDAS_CanopIntIntoGrids(grid_shp, period, var_name):
    grid_shp = ExtractGLDAS(grid_shp, period, var_name)
    return grid_shp


# ------------------------ read data into basins ------------------------


def readStreamflow(home):
    # general set
    usgs_streamflow_dir = os.path.join(
        home, "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow")
    fns = []
    fpaths = []
    usgs_streamflow = []

    for dir in os.listdir(usgs_streamflow_dir):
        fns.extend([fn for fn in os.listdir(os.path.join(usgs_streamflow_dir, dir)) if fn.endswith(".txt")])
        fpaths.extend([os.path.join(usgs_streamflow_dir, dir, fn)
                      for fn in os.listdir(os.path.join(usgs_streamflow_dir, dir)) if fn.endswith(".txt")])

    for i in range(len(fns)):
        fpath = fpaths[i]
        usgs_streamflow_ = pd.read_csv(fpath, sep="\s+", header=None)
        usgs_streamflow.append(usgs_streamflow_)

    # fns -> id
    streamflow_id = [int(fns[:fns.find("_")]) for fns in fns]

    return fns, fpaths, usgs_streamflow, streamflow_id


def checkStreamflowMissing(usgs_streamflow_, date_period=["19980101", "20101231"]):
    reason = ''
    date_period_range = pd.date_range(start=date_period[0], end=date_period[1], freq="D")
    usgs_streamflow_date = list(map(lambda i: datetime(*i), zip(usgs_streamflow_.loc[:, 1], usgs_streamflow_.loc[:, 2], usgs_streamflow_.loc[:, 3])))
    usgs_streamflow_date = np.array(usgs_streamflow_date)
    
    try:
        startIndex = np.where(usgs_streamflow_date == date_period_range[0])[0][0]
        endIndex = np.where(usgs_streamflow_date == date_period_range[-1])[0][0]
        if 'M' not in usgs_streamflow_.iloc[startIndex:endIndex + 1, -1].values:
            judgement = True
        else:
            judgement = False
            reason += f" M in {date_period[0]}-{date_period[1]} "
        if len(usgs_streamflow_.iloc[startIndex:endIndex + 1, :]) < len(date_period_range):
            judgement = False
            reason += f" len < {len(date_period_range)} "

    except:
        judgement = False
        reason += f" cannot find {date_period[0]} or {date_period[1]} in file "

    return judgement, reason


def removeStreamflowMissing(fns, fpaths, usgs_streamflow, date_period):
    """_summary_

    Returns:
        list of dicts: remove_files_Missing
            # unpack remove_files_Missing
            remove_reason_streamflow_Missing= [f["reason"] for f in remove_files_Missing]
            remove_fn_streamflow_Missing = [f["fn"] for f in remove_files_Missing]
            remove_fpath_streamflow_Missing = [f["fpath"] for f in remove_files_Missing]
            remove_usgs_streamflow_Missing = [f["usgs_streamflow"] for f in remove_files_Missing]
    """
    # copy
    fns = copy(fns)
    fpaths = copy(fpaths)
    usgs_streamflow = copy(usgs_streamflow)

    # general set
    remove_files_Missing = []

    # remove Streamflow with 'M' or less len
    i = 0
    while i < len(fns):
        fn = fns[i]
        fpath = fpaths[i]
        usgs_streamflow_ = usgs_streamflow[i]
        judgement, reason = checkStreamflowMissing(usgs_streamflow_, date_period)
        if judgement:
            i += 1
        else:
            # remove file from fns and fpaths
            print(f"remove {fn}")
            remove_files_Missing.append(
                {"fn": fn, "fpath": fpath, "usgs_streamflow": usgs_streamflow_, "reason": reason})
            fns.pop(i)
            fpaths.pop(i)
            usgs_streamflow.pop(i)

    # fns -> id
    streamflow_id = [int(fns[:fns.find("_")]) for fns in fns]

    print(f"count: remove {len(remove_files_Missing)} files, remaining {len(usgs_streamflow)} files")

    return fns, fpaths, usgs_streamflow, streamflow_id, remove_files_Missing


def readStreamflowIntoBasins(basinShp, streamflow_id, usgs_streamflow, read_dates=None):
    extract_lists = []
    for i in tqdm(basinShp.index, desc="loop for reading streamflow into basins", colour="green"):
        # extract hru_id
        basinShp_i = basinShp.loc[i, :]
        hru_id = basinShp_i.hru_id
        extract_index = streamflow_id.index(hru_id)
        usgs_streamflow_ = usgs_streamflow[extract_index]
        
        if read_dates:
            # extract date
            date_period_range = pd.date_range(start=read_dates[0], end=read_dates[1], freq="D")
            usgs_streamflow_date = list(map(lambda i: datetime(*i), zip(usgs_streamflow_.loc[:, 1], usgs_streamflow_.loc[:, 2], usgs_streamflow_.loc[:, 3])))
            usgs_streamflow_date = np.array(usgs_streamflow_date)
            usgs_streamflow_date_str = np.array([d.strftime("%Y%m%d") for d in usgs_streamflow_date])
            
            try:
                startIndex = np.where(usgs_streamflow_date <= date_period_range[0])[0][-1]
            except:
                startIndex = 0
            try:
                endIndex = np.where(usgs_streamflow_date >= date_period_range[-1])[0][0]
            except:
                endIndex = len(usgs_streamflow_)
            
            usgs_streamflow_ = usgs_streamflow_.iloc[startIndex: endIndex + 1, :]
            usgs_streamflow_.loc[:, "date"] = usgs_streamflow_date_str[startIndex: endIndex + 1]
            
        extract_lists.append(usgs_streamflow_)  
        
    basinShp["streamflow"] = extract_lists

    return basinShp


def readForcingDaymet(home):
    forcingDaymet_dir = os.path.join(
        home, "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet")
    fns = []
    fpaths = []
    forcingDaymet = []
    forcingDaymetGaugeAttributes = []

    for dir in os.listdir(forcingDaymet_dir):
        fns.extend([fn for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])
        fpaths.extend([os.path.join(forcingDaymet_dir, dir, fn)
                      for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])

    for i in range(len(fns)):
        fn = fns[i]
        fpath = fpaths[i]
        forcingDaymet.append(pd.read_csv(fpath, sep="\s+", skiprows=3))
        GaugeAttributes_ = pd.read_csv(fpath, header=None, nrows=3).values
        forcingDaymetGaugeAttributes.append({"latitude": GaugeAttributes_[0][0],
                                             "elevation": GaugeAttributes_[1][0],
                                             "basinArea": GaugeAttributes_[2][0],
                                             "gauge_id": int(fn[:8])})

    return fns, fpaths, forcingDaymet, forcingDaymetGaugeAttributes


def readForcingDaymetIntoBasins(forcingDaymet, forcingDaymetGaugeAttributes, basinShp, read_dates, read_keys):
    """
    params:
        read_dates: pd.date_range("19800101", "20141231", freq="D"), should be set to avoid missing value
        read_keys: ["prcp(mm/day)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
    """
    extract_lists = [[] for i in range(len(read_keys))]
    for i in tqdm(basinShp.index, desc="loop for reading forcing Daymet into basins", colour="green"):
        basinShp_i = basinShp.loc[i, :]
        hru_id = basinShp_i.hru_id
        for j in range(len(read_keys)):
            key = read_keys[j]
            extract_list = extract_lists[j]
            forcingDaymet_basin_set, _ = ExtractForcingDaymet(
                forcingDaymet, forcingDaymetGaugeAttributes, hru_id, read_dates)
            extract_list.append(forcingDaymet_basin_set[key])

    for j in range(len(read_keys)):
        key = read_keys[j]
        extract_list = extract_lists[j]
        basinShp[key] = extract_list

    return basinShp


def readBasinAttribute(home):
    # general set
    dir_camels_attribute = "camels_attributes_v2.0"
    camels_clim = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_clim.txt"), sep=";")
    camels_geol = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_geol.txt"), sep=";")
    camels_hydro = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_hydro.txt"), sep=";")
    camels_soil = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_soil.txt"), sep=";")
    camels_topo = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_topo.txt"), sep=";")
    camels_vege = pd.read_csv(os.path.join(home, dir_camels_attribute, "camels_vege.txt"), sep=";")

    BasinAttribute = {"camels_clim": camels_clim,
                      "camels_geol": camels_geol,
                      "camels_hydro": camels_hydro,
                      "camels_soil": camels_soil,
                      "camels_topo": camels_topo,
                      "camels_vege": camels_vege}

    return BasinAttribute


def readBasinAttributeIntoBasins(basinAttribute, basinShp, prefix=None):
    # basinAttribute
    id_basinAttribute = "gauge_id"
    all_columns = list(basinAttribute.columns)

    if prefix:
        all_columns = [prefix + c for c in all_columns]
        basinAttribute.columns = all_columns
        id_basinAttribute = prefix + id_basinAttribute

    # basinShp
    id_basinShp = "hru_id"

    # merge
    basinShp = basinShp.merge(basinAttribute, left_on=id_basinShp, right_on=id_basinAttribute, how="left")

    return basinShp

# ------------------------ select basin function ------------------------


def removeBasinBasedOnStreamflowMissing(basinShp, streamflow_id_removeMissing):
    print(f"remove Basin based on StreamflowMissing: remove {len(basinShp) - len(streamflow_id_removeMissing)} files")
    not_remove_index = [id in streamflow_id_removeMissing for id in basinShp.hru_id.values]
    basinShp = basinShp.iloc[not_remove_index, :]

    print(f"remain {len(basinShp)}")
    return basinShp


def selectBasinBasedOnArea(basinShp, min_area, max_area):
    print(f"select Basin based on Area: {min_area} - {max_area}")
    basinShp = basinShp.loc[(basinShp.loc[:, "AREA_km2"] >= min_area) & (basinShp.loc[:, "AREA_km2"] <= max_area), :]

    print(f"remain {len(basinShp)}")
    return basinShp


def selectBasinBasedOnStreamflowWithZero(basinShp, usgs_streamflow, streamflow_id, zeros_min_num=100):
    # loop for each basin
    selected_id = []
    print(f"select Basin based on StreamflowWithZero, zeros_min_num is {zeros_min_num}")
    for i in range(len(usgs_streamflow)):
        usgs_streamflow_ = usgs_streamflow[i]
        streamflow = usgs_streamflow_.iloc[:, 4].values
        if sum(streamflow == 0) > zeros_min_num:  # find basin with zero streamflow
            selected_id.append(streamflow_id[i])
            print(f"nums of zero value: {sum(streamflow == 0)}")
            # plt.plot(streamflow)
            # plt.ylim(bottom=0)
            # plt.show()

    selected_index = [id in selected_id for id in basinShp.hru_id.values]
    basinShp = basinShp.iloc[selected_index, :]

    print(f"remain {len(basinShp)}")
    return basinShp


def selectBasinBasedOnAridity(basinShp):
    pass


def selectBasinBasedOnElevSlope(basinShp):
    pass

# ------------------------ Intersects grids with HCDN ------------------------


def combineDataframeDropDuplicates(df1, df2, drop_based_columns=None):
    combine_df = df1.append(df2)
    combine_df.index = list(range(len(combine_df)))
    remain_index = combine_df.loc[:, drop_based_columns].drop_duplicates().index
    combine_df = combine_df.loc[remain_index, :]
    return combine_df


def IntersectsGridsWithHCDN(grid_shp, basinShp):
    intersects_grids_list = []
    intersects_grids = pd.DataFrame()
    for i in basinShp.index:
        basin = basinShp.loc[i, "geometry"]
        intersects_grids_ = grid_shp[grid_shp.intersects(basin)]
        intersects_grids = pd.concat([intersects_grids, intersects_grids_])
        intersects_grids_list.append(intersects_grids_)

    intersects_grids["grids_index"] = intersects_grids.index
    intersects_grids.index = list(range(len(intersects_grids)))
    droped_index = intersects_grids["grids_index"].drop_duplicates().index
    intersects_grids = intersects_grids.loc[droped_index, :]

    basinShp["intersects_grids"] = intersects_grids_list
    return basinShp, intersects_grids


# ------------------------ aggregate grid to basins function ------------------------

def aggregate_GLEAMEDaily(basin_shp):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "E"
    aggregate_GLEAMEDaily_list = []
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate gleam_e_daily", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_gleame_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_gleame_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["E"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["E"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["E"])
        aggregate_basin_date["E"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_GLEAMEDaily_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_GLEAMEDaily_list

    return basin_shp


def aggregate_GLEAMEpDaily(basin_shp):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "Ep"
    aggregate_GLEAMEpDaily_list = []
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate gleam_ep_daily", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_gleamep_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_gleamep_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["Ep"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["Ep"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["Ep"])
        aggregate_basin_date["Ep"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_GLEAMEpDaily_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_GLEAMEpDaily_list

    return basin_shp


def aggregate_TRMM_P(basin_shp):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_column = "precipitation"
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basins to aggregate TRMM_P", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df["precipitation"], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df["precipitation"])
        else:
            aggregate_basin_value = aggregate_func(concat_df["precipitation"])
        aggregate_basin_date["precipitation"] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp


def aggregate_ERA5_SM(basin_shp, aggregate_column="swvl1"):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate ERA5 SM", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp
    
    
def aggregate_GlobalSnow_SWE(basin_shp, aggregate_column="swe"):
    aggregate_func = aggregate_func_SWE_axis1
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate GlobalSnow_SWE", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp

def aggregate_GLDAS_CanopInt(basin_shp, aggregate_column="CanopInt_tavg"):
    aggregate_func = partial(np.nanmean, axis=1)
    aggregate_list = []
    
    for i in tqdm(basin_shp.index, desc="loop for basin to aggregate GLDAS_CanopInt", colour="green"):
        intersects_grids_basin = basin_shp.loc[i, "intersects_grids"]

        intersects_grids_basin_df_list = []
        for j in intersects_grids_basin.index:
            intersects_grid = intersects_grids_basin.loc[j, :]
            intersects_grid_daily = intersects_grid[aggregate_column]
            intersects_grids_basin_df_list.append(intersects_grid_daily)

        concat_df = pd.concat(intersects_grids_basin_df_list, axis=1)
        if isinstance(concat_df["date"], pd.Series):
            aggregate_basin_date = pd.DataFrame(concat_df["date"])
        else:
            aggregate_basin_date = pd.DataFrame(concat_df["date"].iloc[:, 0])
        if isinstance(concat_df[aggregate_column], pd.Series):
            aggregate_basin_value = pd.DataFrame(concat_df[aggregate_column])
        else:
            aggregate_basin_value = aggregate_func(concat_df[aggregate_column])
        aggregate_basin_date[aggregate_column] = aggregate_basin_value
        aggregate_basin = aggregate_basin_date

        aggregate_list.append(aggregate_basin)

    basin_shp[f"aggregated_{aggregate_column}"] = aggregate_list

    return basin_shp

# ------------------------ plot function ------------------------
# base function


def plotBackground(basinShp_original, grid_shp,
                   fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = {"facecolor": "none", "alpha": 0.7, "edgecolor": "k"}
    fig, ax = plotBasins(basinShp_original, None, fig, ax, plot_kwgs)
    fig, ax = plotGrids(grid_shp, None, fig, ax)

    return fig, ax


def plotGrids(grid_shp, column=None, fig=None, ax=None, plot_kwgs1=None, plot_kwgs2=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs1 = dict() if not plot_kwgs1 else plot_kwgs1
    plot_kwgs2 = dict() if not plot_kwgs2 else plot_kwgs2
    plot_kwgs1_ = {"facecolor": "none", "alpha": 0.2, "edgecolor": "gray"}
    plot_kwgs2_ = {"facecolor": "none", "alpha": 0.5, "edgecolor": "gray", "markersize": 0.5}

    plot_kwgs1_.update(plot_kwgs1)
    plot_kwgs2_.update(plot_kwgs2)

    grid_shp.plot(ax=ax, column=column, **plot_kwgs1_)
    grid_shp["point_geometry"].plot(ax=ax, **plot_kwgs2_)
    return fig, ax


def plotBasins(basinShp, column=None, fig=None, ax=None, plot_kwgs=None):
    if not ax:
        fig, ax = plt.subplots()
    plot_kwgs = dict() if not plot_kwgs else plot_kwgs
    plot_kwgs_ = {"legend": True}
    plot_kwgs_.update(plot_kwgs)
    basinShp.plot(ax=ax, column=column, **plot_kwgs_)

    return fig, ax


def setBoundary(ax, boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max):
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])
    return ax

# combine function


def plotShp(basinShp_original, basinShp, grid_shp, intersects_grids,
            boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
            fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp["geometry"].plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.2)
    grid_shp["point_geometry"].plot(ax=ax, markersize=0.5, edgecolor="gray", facecolor="gray", alpha=0.5)
    intersects_grids.plot(ax=ax, facecolor="r", edgecolor="gray", alpha=0.2)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax


def plotLandCover(basinShp_original, basinShp, grid_shp, intersects_grids,
                  boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                  fig=None, ax=None):
    colorlevel = [-0.5 + i for i in range(15)]
    colordict = cm.get_cmap("RdBu_r", 14)
    colordict = colordict(range(14))
    ticks = list(range(14))
    ticks_position = list(range(14))
    cmap = mcolors.ListedColormap(colordict)
    norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="major_umd_landcover_classification_grids", alpha=0.4,
                  legend=True, colormap=cmap, norm=norm,
                  legend_kwds={"label": "major_umd_landcover_classification_grids", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    ax_cb = fig.axes[1]
    ax_cb.set_yticks(ticks_position)
    ax_cb.set_yticklabels(ticks)

    return fig, ax


def plotHWSDSoilData(basinShp_original, basinShp, grid_shp, intersects_grids,
                     boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                     fig=None, ax=None, fig_T=None, ax_T=None, fig_S=None, ax_S=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="HWSD_BIL_Value", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "HWSD_BIL_Value", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    # T_USDA_TEX_CLASS
    if not ax_T:
        fig_T, ax_T = plt.subplots()
    basinShp_original.plot(ax=ax_T, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_T)
    grid_shp.plot(ax=ax_T, column="T_USDA_TEX_CLASS", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "T_USDA_TEX_CLASS", "shrink": 0.8})
    intersects_grids.plot(ax=ax_T, facecolor="none", edgecolor="k", alpha=0.7)
    ax_T.set_xlim([boundary_x_min, boundary_x_max])
    ax_T.set_ylim([boundary_y_min, boundary_y_max])

    # S_USDA_TEX_CLASS
    if not ax_S:
        fig_S, ax_S = plt.subplots()
    basinShp_original.plot(ax=ax_S, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax_S)
    grid_shp.plot(ax=ax_S, column="S_USDA_TEX_CLASS", alpha=0.4,
                  legend=True, colormap="Accent",
                  legend_kwds={"label": "S_USDA_TEX_CLASS", "shrink": 0.8})
    intersects_grids.plot(ax=ax_S, facecolor="none", edgecolor="k", alpha=0.7)
    ax_S.set_xlim([boundary_x_min, boundary_x_max])
    ax_S.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax, fig_S, ax_S, fig_T, ax_T


def plotStrmDEM(basinShp_original, basinShp, grid_shp, intersects_grids,
                boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max,
                fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    basinShp_original.plot(ax=ax, facecolor="none", alpha=0.7, edgecolor="k")
    basinShp.plot(ax=ax)
    grid_shp.plot(ax=ax, column="SrtmDEM_mean_Value", alpha=1,
                  legend=True, colormap="gray",
                  legend_kwds={"label": "SrtmDEM_mean_Value", "shrink": 0.8})
    intersects_grids.plot(ax=ax, facecolor="none", edgecolor="k", alpha=0.7)
    ax.set_xlim([boundary_x_min, boundary_x_max])
    ax.set_ylim([boundary_y_min, boundary_y_max])

    return fig, ax

# ------------------------ other function ------------------------


def checkGaugeBasin(basinShp, usgs_streamflow, BasinAttribute, forcingDaymetGaugeAttributes):
    id_basin_shp = set(basinShp.hru_id.values)
    id_usgs_streamflow = set([usgs_streamflow[i].iloc[0, 0] for i in range(len(usgs_streamflow))])
    id_BasinAttribute = set(BasinAttribute["camels_clim"].gauge_id.values)
    id_forcingDaymet = set([forcingDaymetGaugeAttributes[i]["gauge_id"]
                            for i in range(len(forcingDaymetGaugeAttributes))])
    print(f"id_HCDN_shp: {len(id_basin_shp)}, id_usgs_streamflow: {len(id_usgs_streamflow)}, id_BasinAttribute: {len(id_BasinAttribute)}, id_forcingDatmet: {len(id_forcingDaymet)}")
    print(f"id_usgs_streamflow - id_HCDN_shp: {id_usgs_streamflow - id_basin_shp}")
    print(f"id_BasinAttribute - id_HCDN_shp: {id_BasinAttribute - id_basin_shp}")
    print(f"id_forcingDatmet - id_HCDN_shp: {id_forcingDaymet - id_basin_shp}")
    # result
    # id_usgs_streamflow - id_HCDN_shp: {9535100, 6775500, 6846500}
    # id_forcingDatmet - id_HCDN_shp: {6846500, 6775500, 3448942, 1150900, 2081113, 9535100}


def exportToCsv(basin_shp, fpath_dir):
    columns = basin_shp.columns.to_list()
    streamflow = basin_shp.loc[:, columns.pop(11)]
    prep = basin_shp.loc[:, columns.pop(11)]
    gleam_e_daily = basin_shp.loc[:, columns.pop(-1)]

    columns.remove("geometry")
    # columns.remove("intersects_grids")

    csv_df = basin_shp.loc[:, columns]

    # save
    csv_df.to_csv(os.path.join(fpath_dir, "basin_shp.csv"))

    # save streamflow
    if not os.path.exists(os.path.join(fpath_dir, "streamflow")):
        os.mkdir(os.path.join(fpath_dir, "streamflow"))
    for i in tqdm(streamflow.index, desc="loop for basins to save streamflow", colour="green"):
        streamflow[i].to_csv(os.path.join(fpath_dir, "streamflow", f"{i}.csv"))

    # save prep
    if not os.path.exists(os.path.join(fpath_dir, "prep")):
        os.mkdir(os.path.join(fpath_dir, "prep"))
    for i in tqdm(prep.index, desc="loop for basins to save prep", colour="green"):
        prep[i].to_csv(os.path.join(fpath_dir, "prep", f"{i}.csv"))

    # save gleam_e_daily
    if not os.path.exists(os.path.join(fpath_dir, "gleam_e_daily")):
        os.mkdir(os.path.join(fpath_dir, "gleam_e_daily"))
    for i in tqdm(gleam_e_daily.index, desc="loop for basins to save gleam_e_daily", colour="green"):
        gleam_e_daily[i].to_csv(os.path.join(fpath_dir, "gleam_e_daily", f"{i}.csv"))
