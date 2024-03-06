# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import os
from typing import Any
import numpy as np
import pandas as pd
from WaterBudgetClosure.dataPreprocess_CAMELS_functions import Any
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
import matplotlib.colors as mcolors
from matplotlib import cm
import pickle
from tqdm import *
from WaterBudgetClosure.dataPreprocess_CAMELS_functions import *

import warnings
warnings.filterwarnings('ignore')
# import imp
# from importlib import reload
# reload(dataPreprocess_HWSD)

# ------------------------ dataProcess_CAMELS class ------------------------


class Basins(gpd.GeoDataFrame):

    def __add__(self, basins):
        pass

    def __sub__(self, basins):
        pass

    def __and__(self, basins):
        pass


class Grids(gpd.GeoDataFrame):

    def __add__(self, grids):
        pass

    def __sub__(self, grids):
        pass

    def __and__(self, grids):
        pass


def intersectGridsWithBasins(grids: Grids, basins: Basins):
    intersects_grids_list = []
    intersects_grids = Grids()
    for i in basins.index:
        basin = basins.loc[i, "geometry"]
        intersects_grids_ = grids[grids.intersects(basin)]
        intersects_grids = pd.concat([intersects_grids, intersects_grids_], axis=0)
        intersects_grids_list.append(intersects_grids_)

    intersects_grids["grids_index"] = intersects_grids.index
    intersects_grids.index = list(range(len(intersects_grids)))
    droped_index = intersects_grids["grids_index"].drop_duplicates().index
    intersects_grids = intersects_grids.loc[droped_index, :]

    basins["intersects_grids"] = intersects_grids_list
    return basins, intersects_grids


class HCDNBasins(Basins):
    def __init__(self, home, data=None, *args, geometry=None, crs=None, **kwargs):
        HCDN_shp_path = os.path.join(home, "basin_set_full_res", "HCDN_nhru_final_671.shp")
        HCDN_shp = gpd.read_file(HCDN_shp_path)
        HCDN_shp["AREA_km2"] = HCDN_shp.AREA / 1000000  # m2 -> km2
        super().__init__(HCDN_shp, *args, geometry=geometry, crs=crs, **kwargs)


class HCDNGrids(Grids):
    def __init__(self, home, data=None, *args, geometry=None, crs=None, **kwargs):
        grid_shp_label_path = os.path.join(home, "map", "grids_0_25_label.shp")
        grid_shp_label = gpd.read_file(grid_shp_label_path)
        grid_shp_path = os.path.join(home, "map", "grids_0_25.shp")
        grid_shp = gpd.read_file(grid_shp_path)
        grid_shp["point_geometry"] = grid_shp_label.geometry
        super().__init__(grid_shp, *args, geometry=geometry, crs=crs, **kwargs)

    def createBoundaryShp(self):
        boundary_shp, boundary_x_y = createBoundaryShp(self)
        return boundary_shp, boundary_x_y


class dataProcess_CAMELS:
    """ main/base class: dataProcess_CAMELS

    function structure:
        self.__init__(): set path
        self.bool_set(): a set of bool_params, a switch controlling enablement of the functions in the class
        self.__call__(): workflow of this class
            self.read_from_exist_file(): read variable from exist filem see save()
            self.read_basin_grid(): read basin shp and grid shp, create boundary_shp and boudary_x_y
            self.readDataIntoBasins(): read data into basins
            self.selectBasins(): select basin from original basin, all based on self.basin_shp
            self.intersectsGridsWithBasins(): intersect grids with basins to get intersects_grids
            self.readDataIntoGrids(): read data into grids, normally, read into intersects_grids
            self.combineIntersectsGridsWithBasins(): combine intersects_grids with basins to save the intersects_grids with data into basin_shp
            self.aggregate_grid_to_basins(): aggregate basin_shp["intersects_grids"][...] into basin_shp[...]
            self.save(): save data into files

    variable structure:
        private params: _params, params controlling initial settings
        main variable: 
            self.basin_shp: GeoDataframe, index is the basins id, columns contains "intersects_grids" "AREA" "AREA_km2" "lon_cen" "lat_cen"
            self.basin_shp_original: GeoDataframe, the backup of basin_shp (the basin_shp is modified in the process)
            self.grid_shp: GeoDataframe, rectangular grid, index is the grids id
            self.intersects_grids: GeoDataframe, same as grid_shp but intersected with basin_shp, thus is can be treated as grid_shp (e.g., you can read data into it)
            self.boundary_shp: GeoDataframe, rectangular shp corresponding the boundary, based on grid_shp
            self.boundary_x_y: list, [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
        other:
            self.fns_streamflow: list of str, file name corresponding to the usgs_streamflow
            self.streamflow_id: list of int, id corresponding to the usgs_streamflow
            self.fpaths_streamflow: list of str, file path corresponding to the usgs_streamflow 
            self.usgs_streamflow: list of pd.Dataframe, usgs streamflow data
            self.remove_files_Missing: list of set, attributes corresponding to the removed files
                set: {"fn": fn, "fpath": fpath, "usgs_streamflow": usgs_streamflow_, "reason": reason}

    save file structure:
        same as variable structure, all files are saved in .pkl format
            basin_shp.pkl
            basin_shp_original.pkl
            grid_shp.pkl
            intersects_grids.pkl
            boundary_shp.pkl
            boundary_x_y.pkl
            remove_files_Missing.pkl

    how to use:
        # general use
        subclass = class()  # specific your design, use subclass overriding any part you want
        instance = subclass()
        instance.bool_set()  # a switch controling enablement of the functions in the class, you can remove it and set instance function specifically
        instance()

        # append additional data into existing files
        (1) read_from_exist_file
        (2) use the instance.basin_shp/grid_shp and instance.function to read data again
        (3) self.save() again

    """

    def __init__(self, home, subdir, date_period) -> None:
        self._home = home
        self._subdir = subdir
        self._date_period = date_period

    def __call__(self, read_from_exist_file_bool=False, *args: Any, **kwds: Any) -> Any:
        if read_from_exist_file_bool:
            self.read_from_exist_file()
        else:
            self.read_basin_grid()
            self.readDataIntoBasins()
            self.selectBasins()
            self.intersectsGridsWithBasins()
            self.readDataIntoGrids()
            self.combineIntersectsGridsWithBasins()
            self.aggregate_grid_to_basins()
            self.save()

        if self._readBasinAttribute_bool:
            self.BasinAttribute = readBasinAttribute(self._home)  # 671

        if self._plot_bool:
            self.plot()

    def read_from_exist_file(self):
        # read data from file
        self.basin_shp = pd.read_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp.pkl"))
        self.basin_shp_original = pd.read_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp_original.pkl"))
        self.grid_shp = pd.read_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "grid_shp.pkl"))
        self.intersects_grids = pd.read_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "intersects_grids.pkl"))
        self.boundary_shp = pd.read_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_shp.pkl"))
        with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_x_y.pkl"), "rb") as f:
            self.boundary_x_y = pickle.load(f)
        if self._removeStreamflowMissing_bool:
            with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "remove_files_Missing.pkl"), "rb") as f:
                self.remove_files_Missing = pickle.load(f)

    def read_basin_grid(self):
        # read basin shp
        self.basin_shp = HCDNBasins(self._home)
        self.basin_shp_original = HCDNBasins(self._home)  # backup for HCDN_shp

        # read grids and createBoundaryShp
        self.grid_shp = HCDNGrids(self._home)
        # boundary_x_y = [boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max]
        self.boundary_shp, self.boundary_x_y = self.grid_shp.createBoundaryShp()

    def readDataIntoBasins(self):
        # read streamflow into basins
        self.readStreamflowIntoBasins()
        self.readForcingDaymetIntoBasins()

    def readStreamflowIntoBasins(self):
        self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
            self._home)  # 674

        # read streamflow into basins
        self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)

    def readForcingDaymetIntoBasins(self):
        fns_forcingDaymet, fpaths_forcingDaymet, forcingDaymet, forcingDaymetGaugeAttributes = readForcingDaymet(
            self._home)  # 677

        # read forcingDaymet (multi-variables) into basins
        read_dates = pd.date_range(self._date_period[0], self._date_period[1], freq="D")
        read_keys = ["prcp(mm/day)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
        self.basin_shp = readForcingDaymetIntoBasins(
            forcingDaymet, forcingDaymetGaugeAttributes, self.basin_shp, read_dates, read_keys)

    def selectBasins(self):
        self.removeStreamflowMissing()
        self.basin_shp = selectBasinBasedOnStreamflowWithZero(
            self.basin_shp, self.usgs_streamflow, self.streamflow_id, zeros_min_num=100)  # 552 -> 103

    def removeStreamflowMissing(self):
        # remove streamflow when Missing
        self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id, self.remove_files_Missing = removeStreamflowMissing(
            self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, date_period=self._date_period)  # 674 - 122 = 552

        # remove basins with streamflowMissing
        self.basin_shp = removeBasinBasedOnStreamflowMissing(self.basin_shp, self.streamflow_id)  # 671 - 122 = 552

    def intersectsGridsWithBasins(self):
        self.basin_shp, self.intersects_grids = intersectGridsWithBasins(self.grid_shp, self.basin_shp)

    def readDataIntoGrids(self):
        self.intersects_grids = readSrtmDEMIntoGrids(self.intersects_grids, self._plot_bool)
        self.intersects_grids = readUMDLandCoverIntoGrids(self.intersects_grids)
        self.intersects_grids = readHWSDSoilDataIntoGirds(self.intersects_grids, self.boundary_shp, self._plot_bool)
        self.intersects_grids = readGLEAMEDailyIntoGrids(
            self.intersects_grids, period=list(range(1980, 2011)), var_name="E")

    def combineIntersectsGridsWithBasins(self):
        self.basin_shp, self.intersects_grids = intersectGridsWithBasins(self.intersects_grids, self.basin_shp)

    def aggregate_grid_to_basins(self):
        self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)

    def save(self):
        self.basin_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp.pkl"))
        self.basin_shp_original.to_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "basin_shp_original.pkl"))
        self.grid_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "grid_shp.pkl"))
        self.intersects_grids.to_pickle(os.path.join(
            self._home, "dataPreprocess_CAMELS", self._subdir, "intersects_grids.pkl"))
        self.boundary_shp.to_pickle(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_shp.pkl"))

        with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "boundary_x_y.pkl"), "wb") as f:
            pickle.dump(self.boundary_x_y, f)

        if self._removeStreamflowMissing_bool:
            with open(os.path.join(self._home, "dataPreprocess_CAMELS", self._subdir, "remove_files_Missing.pkl"), "wb") as f:
                pickle.dump(self.remove_files_Missing, f)

    def plot(self):
        fig, ax = plotBackground(self.basin_shp_original, self.grid_shp, fig=None, ax=None)
        plot_kwgs1 = {"facecolor": "none", "alpha": 0.2, "edgecolor": "r"}
        plot_kwgs2 = {"facecolor": "none", "alpha": 0.5, "edgecolor": "r", "markersize": 0.5}
        fig, ax = plotGrids(self.intersects_grids, None, fig, ax, plot_kwgs1, plot_kwgs2)
        ax = setBoundary(ax, *self.boundary_x_y)

        if self._readUMDLandCoverIntoGrids_bool:
            plotLandCover(self.basin_shp_original, self.basin_shp, self.grid_shp,
                          self.intersects_grids, *self.boundary_x_y)
        if self._readHWSDSoilDataIntoGirds_bool:
            plotHWSDSoilData(self.basin_shp_original, self.basin_shp, self.grid_shp,
                             self.intersects_grids, *self.boundary_x_y)
        if self._readSrtmDEMIntoGrids_bool:
            plotStrmDEM(self.basin_shp_original, self.basin_shp, self.grid_shp,
                        self.intersects_grids, *self.boundary_x_y)
        plt.show()


class dataProcess_CAMELS_read_basin_grid(dataProcess_CAMELS):
    """_summary_: just read basin and grid of CAMELS for review

    Args:
        dataProcess_CAMELS (_type_): _description_
    """

    def __call__(self, plot=True) -> Any:
        self.read_basin_grid()
        if plot:
            self.plot()

    def plot(self):
        fig, ax = plotBackground(self.basin_shp_original, self.grid_shp, fig=None, ax=None)
        ax = setBoundary(ax, *self.boundary_x_y)
        plt.show()


class dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(dataProcess_CAMELS):
    """_summary_: read basin and grid of CAMELS, and remove streamflow missing
    baseline for dataProcess_CAMELS: 
        read basin grid: create basin and grid shp
        select basins: removeStreamflowMissing(), 671 -> 652
        read data into basins: readStreamflowIntoBasins(), readBasinAttributeIntoBasins()
        #* read data into grids: no read grids data, leave it for further customization

    Args:
        dataProcess_CAMELS (_type_): _description_
    """
    def __init__(self, home, subdir, date_period) -> None:
        self._date_period = date_period
        super().__init__(home, subdir)

    def __call__(self, plot=True) -> Any:
        self.read_basin_grid()
        self.readDataIntoBasins()
        self.selectBasins()
        self.intersectsGridsWithBasins()
        self.combineIntersectsGridsWithBasins()

        if plot:
            self.plot()
        
    def readDataIntoBasins(self):
        self.readStreamflowIntoBasins()
        self.readBasinAttributeIntoBasins()
        # self.readForcingDaymetIntoBasins()
    
    def readStreamflowIntoBasins(self):
        self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
            self._home)  # 674

        # read streamflow into basins
        self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)
    
    def readForcingDaymetIntoBasins(self):
        fns_forcingDaymet, fpaths_forcingDaymet, forcingDaymet, forcingDaymetGaugeAttributes = readForcingDaymet(
            self._home)  # 677
        
        # read forcingDaymet (multi-variables) into basins
        read_dates = pd.date_range(self._date_period[0], self._date_period[1], freq="D")
        read_keys = ["prcp(mm/day)", "swe(mm)", "tmax(C)", "tmin(C)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
        self.basin_shp = readForcingDaymetIntoBasins(
            forcingDaymet, forcingDaymetGaugeAttributes, self.basin_shp, read_dates, read_keys)
        
    def readBasinAttributeIntoBasins(self):
        BasinAttributes = readBasinAttribute(self._home)
        for key in BasinAttributes.keys():
            self.basin_shp = readBasinAttributeIntoBasins(BasinAttributes[key], self.basin_shp, prefix=key+":")
    
    def selectBasins(self):
        self.removeStreamflowMissing()  # 671 -> 652
    
    def removeStreamflowMissing(self):
        # remove streamflow when Missing
        self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id, self.remove_files_Missing = removeStreamflowMissing(
            self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, date_period=self._date_period)  # 671 -> 652

        # remove basins with streamflowMissing
        self.basin_shp = removeBasinBasedOnStreamflowMissing(self.basin_shp, self.streamflow_id)

    def plot(self):
        fig, ax = plotBackground(self.basin_shp_original, self.grid_shp, fig=None, ax=None)
        plot_kwgs1 = {"facecolor": "none", "alpha": 0.2, "edgecolor": "r"}
        plot_kwgs2 = {"facecolor": "none", "alpha": 0.5, "edgecolor": "r", "markersize": 0.5}
        fig, ax = plotGrids(self.intersects_grids, None, fig, ax, plot_kwgs1, plot_kwgs2)
        ax = setBoundary(ax, *self.boundary_x_y)
        plt.show()


class dataProcess_CAMELS_WaterBalanceAnalysis(dataProcess_CAMELS):
    """_summary_: WaterBalanceAnalysis detS = P - E -R, WaterClosureResidual = P - E - R - detS
    read grids data for water balance analysis: based on baseline data from dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing
    read grids data into basins and aggregate it
    
    Args:
        dataProcess_CAMELS (_type_): _description_
    """
    def __init__(self, home, subdir, date_period) -> None:
        self._date_period = date_period
        super().__init__(home, subdir)

    def __call__(self, dpc_base=None, iindex_basin_shp=0) -> Any:
        if dpc_base:
            self.basin_shp = dpc_base.basin_shp.iloc[iindex_basin_shp: iindex_basin_shp + 1, :]
            self.basin_shp_original = dpc_base.basin_shp_original
            self.grid_shp = dpc_base.grid_shp
            self.boundary_shp = dpc_base.boundary_shp
            self.boundary_x_y = dpc_base.boundary_x_y
            self.intersects_grids = dpc_base.intersects_grids
        else:
            self.read_basin_grid()
            
        self.intersectsGridsWithBasins()
        self.readDataIntoGrids()
        self.combineIntersectsGridsWithBasins()
        self.aggregate_grid_to_basins()

    def readDataIntoGrids(self):
        # read TRMM_P
        self.intersects_grids = readTRMMPIntoGrids(
            self.intersects_grids, period=self._date_period,
            var_name="precipitation")
        
        # read GLEAME_Daily
        self.intersects_grids = readGLEAMEDailyIntoGrids(
            self.intersects_grids, period=self._date_period, var_name="E")
        
        # read GLEAME_Daily_Ep
        self.intersects_grids = readGLEAMEDailyIntoGrids(
            self.intersects_grids, period=self._date_period, var_name="Ep")

        # read ERA5_SM
        self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="1")
        self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="2")
        self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="3")
        self.intersects_grids = readERA5_SMIntoGrids(self.intersects_grids, period=self._date_period, var_name="4")

        # read GlobalSnow_SWE
        self.intersects_grids = readGlobalSnow_SWEIntoGrids(self.intersects_grids, period=self._date_period, var_name="swe")
        
        # read GLDAS_CanopInt
        self.intersects_grids = readGLDAS_CanopIntIntoGrids(self.intersects_grids, period=self._date_period, var_name="CanopInt_tavg")

    def aggregate_grid_to_basins(self):
        # aggregate TRMM_P
        self.basin_shp = aggregate_TRMM_P(self.basin_shp)
        
        # aggregate GLEAME_Daily
        self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)
        self.basin_shp = aggregate_GLEAMEpDaily(self.basin_shp)
        
        # aggregate ERA5_SM
        self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl1")
        self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl2")
        self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl3")
        self.basin_shp = aggregate_ERA5_SM(self.basin_shp, aggregate_column="swvl4")
        
        # aggregate GlobalSnow_SWE
        self.basin_shp = aggregate_GlobalSnow_SWE(self.basin_shp, aggregate_column="swe")
        
        # aggregate GLDAS_CanopInt
        self.basin_shp = aggregate_GLDAS_CanopInt(self.basin_shp, aggregate_column="CanopInt_tavg")


class dataProcess_CAMELS_Malan_Basins_with_Zeros(dataProcess_CAMELS):
    
    def __call__(self, read_from_exist_file_bool=False, *args: Any, **kwds: Any) -> Any:
        self._removeStreamflowMissing_bool = True
        if read_from_exist_file_bool:
            self.read_from_exist_file()
        else:
            self.read_basin_grid()
            self.readDataIntoBasins()
            self.selectBasins()
            self.intersectsGridsWithBasins()
            self.readDataIntoGrids()
            self.combineIntersectsGridsWithBasins()
            self.aggregate_grid_to_basins()
            self.save()

    def readDataIntoBasins(self):
        self.readStreamflowIntoBasins()
        
    def readDataIntoGrids(self):
        # read GLEAME_Daily
        self.intersects_grids = readGLEAMEDailyIntoGrids(
            self.intersects_grids, period=self._date_period, var_name="E")
        
    def readStreamflowIntoBasins(self):
        self.fns_streamflow, self.fpaths_streamflow, self.usgs_streamflow, self.streamflow_id = readStreamflow(
            self._home)  # 674

        # read streamflow into basins
        self.basin_shp = readStreamflowIntoBasins(self.basin_shp, self.streamflow_id, self.usgs_streamflow, self._date_period)
    
    def selectBasins(self):
        self.removeStreamflowMissing()
        self.basin_shp = selectBasinBasedOnStreamflowWithZero(
            self.basin_shp, self.usgs_streamflow, self.streamflow_id, zeros_min_num=100)  # 552 -> 103
    
    def aggregate_grid_to_basins(self):
        # aggregate GLEAME_Daily
        self.basin_shp = aggregate_GLEAMEDaily(self.basin_shp)
        

# ------------------------ patch ------------------------
class dataProcess_CAMELS_WaterBalanceAnalysis_patch_Ep(dataProcess_CAMELS):
    """_summary_: WaterBalanceAnalysis detS = P - E -R, WaterClosureResidual = P - E - R - detS

    Args:
        dataProcess_CAMELS (_type_): _description_
    """
    def __init__(self, home, subdir, date_period) -> None:
        self._date_period = date_period
        super().__init__(home, subdir)

    def __call__(self, dpc_base=None, iindex_basin_shp=0) -> Any:
        if dpc_base:
            self.basin_shp = dpc_base.basin_shp.iloc[iindex_basin_shp: iindex_basin_shp + 1, :]
            self.basin_shp_original = dpc_base.basin_shp_original
            self.grid_shp = dpc_base.grid_shp
            self.boundary_shp = dpc_base.boundary_shp
            self.boundary_x_y = dpc_base.boundary_x_y
            self.intersects_grids = dpc_base.intersects_grids
        else:
            self.read_basin_grid()
            
        self.intersectsGridsWithBasins()
        self.readDataIntoGrids()
        self.combineIntersectsGridsWithBasins()
        self.aggregate_grid_to_basins()

    def readDataIntoGrids(self):
        # read GLEAME_Daily_Ep
        self.intersects_grids = readGLEAMEDailyIntoGrids(
            self.intersects_grids, period=self._date_period, var_name="Ep")

    def aggregate_grid_to_basins(self):
        # aggregate GLEAME_Daily
        self.basin_shp = aggregate_GLEAMEpDaily(self.basin_shp)

# ------------------------ demo ------------------------


def reviewCAMELSBasinData(home):
    subdir = "review_CAMELS"
    dpc = dataProcess_CAMELS_read_basin_grid(home, subdir)
    dpc()
    return dpc
    

def demoMalan_Basins_with_Zeros(home):
    subdir = "Malan_Basins_with_Zeros"
    date_period = ["19800101", "20101231"]
    dpc = dataProcess_CAMELS_Malan_Basins_with_Zeros(home, subdir, date_period)
    dpc(read_from_exist_file_bool=False)

    # export to csv
    exportToCsv(dpc.basin_shp, fpath_dir="F:/work/malan/20230913CAMELSdata")


if __name__ == "__main__":
    # general set
    root, home = setHomePath(root="E:")

    # review
    # dpc_review = reviewCAMELSBasinData(home)

    # demo
    demoMalan_Basins_with_Zeros(home)

    # select basin
    # HCDN_shp.AREA_km2.hist()
    # HCDN_shp = selectBasinBasedOnArea(HCDN_shp, min_area=2000, max_area=5000)  # 552 -> 114(1000-10000) -> 34(2000-5000)
    # HCDN_shp.AREA_km2.hist()
