# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import sys
from WaterBudgetClosure import dataPreprocess_CAMELS
from WaterBudgetClosure.WaterBalanceAnalysis import WaterBalanceAnalysis, gapFillingSWE
wba = WaterBalanceAnalysis()
from WaterBudgetClosure.dataPreprocess_CAMELS_functions import setHomePath
from WaterBudgetClosure.dataPreprocess_CAMELS_ForcingDaymet import ExtractForcingDaymet
from tqdm import *
import pandas as pd
import numpy as np
__location__ =  os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__location__))
import pickle


def unitFormat(forcing):
    """
    note:
        qt/forcing_streamflow: m3/s
        streamflow: feet3/s
        (streamflow * 0.0283168 = qt)
        
        qt for model to simulation
        streamflow for WaterBudgetAnalysis to analyze
    """
    # read
    streamflow = forcing.loc[:, "streamflow"]
    sm1 = forcing.loc[:, "swvl1"]
    sm2 = forcing.loc[:, "swvl2"]
    sm3 = forcing.loc[:, "swvl3"]
    sm4 = forcing.loc[:, "swvl4"]
    swe_ = forcing.loc[:, ["date", "swe"]]
    
    # gap filling: swe
    swe_, _ = gapFillingSWE(swe_, plot=False)
    swe = swe_.loc[:, "swe_filling"]
    
    # unit conversion: streamflow, sm
    # streamflow: feet3/s -> m3/s
    factor_feet2meter = 0.0283168
    qt = streamflow * factor_feet2meter
    
    # sm: v2h, note layer is  the height of m
    layer1 = 0.07
    layer2 = 0.21
    layer3 = 0.72
    layer4 = 1.89
    sm1 = sm1 * layer1
    sm2 = sm2 * layer2
    sm3 = sm3 * layer3
    sm4 = sm4 * layer4
    
    # sm: m2mm
    sm1 = sm1 * 1000
    sm2 = sm2 * 1000
    sm3 = sm3 * 1000
    sm4 = sm4 * 1000
    
    # combine sm
    sm = sm1 + sm2 + sm3 + sm4
    
    # combine into forcing
    # keep streamflow as original unit, set new column forcing_streamflow/qt and converse it as m3/s
    forcing.loc[:, "forcing_streamflow"] = qt  # m3/s
    forcing.loc[:, "swvl1"] = sm1  # mm/day
    forcing.loc[:, "swvl2"] = sm2  # mm/day
    forcing.loc[:, "swvl3"] = sm3  # mm/day
    forcing.loc[:, "swvl4"] = sm4  # mm/day
    forcing.loc[:, "swv"] = sm  # mm/day
    forcing.loc[:, "swe"] = swe  # mm/day
    
    return forcing
    

def forcing_Preparation(dpc_base, daymet_varnames, analysis_period):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    dst_home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # loop for basins to analysis
    _, _, forcingDaymet, forcingDaymetGaugeAttributes = dataPreprocess_CAMELS.readForcingDaymet(home="E:/data/hydrometeorology/CAMELS")
    for i in tqdm(range(len(fnames)), desc="loop for basins to prepare forcing", colour="grey"):
        
        # read
        fpath = fpaths[i]
        fname = fnames[i]
        forcing = pd.read_csv(fpath, index_col=0)
        
        # unit format
        forcing = unitFormat(forcing)
        
        # extract basin
        basin_index = int(fname[: fname.find("_")])
        basin_shp = dpc_base.basin_shp.loc[basin_index, :]
        
        # info
        basinArea = basin_shp["AREA_km2"]
        basinLatCen = basin_shp["lat_cen"]
        hru_id = basin_shp["hru_id"]
        
        # get forcingDaymet
        forcingDaymet_gauge_set, _ = ExtractForcingDaymet(forcingDaymet, forcingDaymetGaugeAttributes, hru_id, analysis_period, plot=False)
        
        # combine
        for vn in daymet_varnames:
            forcingDaymet_ = forcingDaymet_gauge_set[vn]
            forcingDaymet_ = pd.DataFrame(forcingDaymet_)
            forcingDaymet_.loc[:, "date"] = [int(s) for s in list(forcingDaymet_.index.strftime("%Y%m%d"))]
            forcingDaymet_.index = list(range(len(forcingDaymet_)))
            forcing = forcing.merge(forcingDaymet_, on="date", how="left")

        # set DateTimeIndex
        forcing.index = pd.to_datetime(forcing.loc[:, "date"], format="%Y%m%d")

        # rename
        forcing.rename(columns={"tmax(C)": "tmax", "tmin(C)": "tmin", "precipitation": "prec", "forcing_streamflow": "qt", "Ep": "pet"}, inplace=True)
        
        # cal "tmean"
        forcing.loc[:, "tmean"] = forcing.loc[:, ["tmin", "tmax"]].mean(axis=1)
        
        # add info
        forcing.loc[forcing.index[0], "basinArea"] = basinArea
        forcing.loc[forcing.index[0], "basinLatCen"] = basinLatCen
        
        # save
        dst_fname = fname.replace("dpc", "forcing")
        dst_path = os.path.join(dst_home, dst_fname)
        forcing.to_csv(dst_path)
        
        
if __name__ == "__main__":
    # 1998-2010
    analysis_period = ["19980101", "20101231"]
    root, home = setHomePath(root="E:")
    subdir = "WaterBalanceAnalysis"
    
    # instance dpc_base
    with open(os.path.join(home, "dataPreprocess_CAMELS", subdir, "dpc_base.pkl"), "rb") as f:
        dpc_base = pickle.load(f)

    # daymet_varnames
    daymet_varnames = ["tmax(C)", "tmin(C)"]
    
    # forcing_Preparation
    forcing_Preparation(dpc_base, daymet_varnames, analysis_period)