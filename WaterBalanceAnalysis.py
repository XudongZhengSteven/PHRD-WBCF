# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import pandas as pd
import numpy as np
from WaterBudgetClosure.dataPreprocess_CAMELS import dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing
from WaterBudgetClosure.dataPreprocess_CAMELS import dataProcess_CAMELS_WaterBalanceAnalysis
from WaterBudgetClosure.dataPreprocess_CAMELS import dataProcess_CAMELS_WaterBalanceAnalysis_patch_Ep
from WaterBudgetClosure.dataPreprocess_CAMELS import dataProcess_CAMELS_read_basin_grid
from WaterBudgetClosure.dataPreprocess_CAMELS import setHomePath
import pickle
from tqdm import *
import matplotlib.pyplot as plt
from copy import copy
from scipy.interpolate import interp1d


def instance_dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(home, subdir, analysis_period):
    dpc_base = dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(home, subdir, analysis_period)  # 653
    dpc_base(plot=True)
    print(dpc_base.basin_shp)
    
    with open(os.path.join(home, "dataPreprocess_CAMELS", subdir, "dpc_base.pkl"), "wb") as f:
        pickle.dump(dpc_base, f)


def dataProcess_CAMELS_iterateBasin(dpc_base, home, subdir, analysis_period):
    for iindex_basin_shp in tqdm(range(len(dpc_base.basin_shp)), colour="green", desc="loop for iteratre Basins"):
        index_basin_shp = dpc_base.basin_shp.index[iindex_basin_shp]
        dpc_basin = dataProcess_CAMELS_WaterBalanceAnalysis(home, subdir, analysis_period)
        dpc_basin(dpc_base, iindex_basin_shp)
        
        # save
        streamflow = dpc_basin.basin_shp.loc[index_basin_shp, "streamflow"]
        streamflow.rename(columns = {4: "streamflow"}, inplace=True)
        streamflow = streamflow.loc[:, ["date", "streamflow"]]
        TRMM_P_precipitation = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_precipitation"]
        GLEAM_E = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_E"]
        swvl1 = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_swvl1"]
        swvl2 = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_swvl2"]
        swvl3 = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_swvl3"]
        swvl4 = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_swvl4"]
        swe = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_swe"]
        CanopInt_tavg = dpc_basin.basin_shp.loc[index_basin_shp, "aggregated_CanopInt_tavg"]
        
        df_concat = streamflow.merge(TRMM_P_precipitation, on="date", how="left")
        df_concat = df_concat.merge(GLEAM_E, on="date", how="left")
        df_concat = df_concat.merge(swvl1, on="date", how="left")
        df_concat = df_concat.merge(swvl2, on="date", how="left")
        df_concat = df_concat.merge(swvl3, on="date", how="left")
        df_concat = df_concat.merge(swvl4, on="date", how="left")
        df_concat = df_concat.merge(swe, on="date", how="left")
        df_concat = df_concat.merge(CanopInt_tavg, on="date", how="left")
        
        df_concat.to_csv(os.path.join(home, "dataPreprocess_CAMELS", subdir, "basins", f"{index_basin_shp}_dpc_basin_{dpc_basin.basin_shp.hru_id.values[0]}.csv"))


def patch_addswe_dataProcess_CAMELS_iterateBasin(dpc_base):
    # review
    for i in dpc_base.basin_shp.index:
        swe_sum = sum(dpc_base.basin_shp.loc[i, "swe(mm)"])
        print(swe_sum)
    
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # read
    for i in tqdm(range(len(fnames)), desc="loop for patch, adding swe", colour="green"):
        fpath = fpaths[i]
        fname = fnames[i]
        df_basin = pd.read_csv(fpath, index_col=0)
        basin_index = int(fname[: fname.find("_")])
        
        # get swe
        swe = dpc_base.basin_shp.loc[basin_index, "swe(mm)"]
        if sum(swe) > 0:
            print(f"basin{basin_index}", sum(swe))
        date = pd.DataFrame([int(d) for d in swe.index.strftime(date_format="%Y%m%d")], columns=["date"], index=swe.index)
        swe = pd.concat([date, swe], axis=1)
        swe.rename(columns = {"swe(mm)": "swe"}, inplace=True)
        
        # combine
        df_basin = df_basin.merge(swe, on="date")
        
        # save
        df_basin.to_csv(fpath)


def path_remove_columns(column="swe_x"):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # read
    for i in tqdm(range(len(fnames)), desc=f"loop for patch, removing column {column}", colour="green"):
        fpath = fpaths[i]
        fname = fnames[i]
        df_basin = pd.read_csv(fpath, index_col=0)
        
        # remove column
        df_basin.drop(column, axis=1, inplace=True)
        
        # save
        df_basin.to_csv(fpath)
        
        
def patch_addGLEAM_Ep_dataProcess_CAMELS_iterateBasin(dpc_base, home, subdir, analysis_period):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # read
    for i in tqdm(range(len(fnames)), desc="loop for patch, adding GLEAM_Ep", colour="green"):
        fpath = fpaths[i]
        fname = fnames[i]
        df_basin = pd.read_csv(fpath, index_col=0)
        basin_index = int(fname[: fname.find("_")])
        iindex_basin_shp = np.where(dpc_base.basin_shp.index == basin_index)[0][0]
        
        # get GLEAM_Ep
        dpc_basin_patch = dataProcess_CAMELS_WaterBalanceAnalysis_patch_Ep(home, subdir, analysis_period)
        dpc_basin_patch(dpc_base, iindex_basin_shp)
        GLEAM_Ep = dpc_basin_patch.basin_shp.loc[basin_index, "aggregated_Ep"]
        
        # combine
        df_basin["date"] = df_basin["date"].astype(str)
        df_basin = df_basin.merge(GLEAM_Ep, on="date")
        
        # save
        df_basin.to_csv(fpath)


def path_remove_columns(column=["Ep", "Ep_x", "Ep_y"]):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # read
    for i in tqdm(range(len(fnames)), desc=f"loop for patch, removing column {column}", colour="green"):
        fpath = fpaths[i]
        fname = fnames[i]
        df_basin = pd.read_csv(fpath, index_col=0)
        
        # remove column
        for c in column:
            try:
                df_basin.drop(c, axis=1, inplace=True)
            except:
                continue
        # save
        df_basin.to_csv(fpath)

class WaterBalanceAnalysis:

    def __init__(self):
        pass

    def __call__(self, streamflow, pre, et, sm1, sm2, sm3, sm4, swe, basinArea, layer1, layer2, layer3, layer4, window, date_residual=None):
        # unit conversion
        streamflow, pre, et, sm1, sm2, sm3, sm4, swe = self.unitConversion(streamflow, pre, et, sm1, sm2, sm3, sm4, swe, basinArea, layer1, layer2, layer3, layer4)
        
        # combine sm
        sm = sm1 + sm2 + sm3 + sm4
        
        # filtering
        streamflow_filter = self.filter_period(streamflow, window)
        pre_filter = self.filter_period(pre, window)
        et_filter = self.filter_period(et, window)
        det_sm = self.filter_differential(sm, window)
        det_swe = self.filter_differential(swe, window)

        # water balance
        residual = self.waterBalanceEquation(pre_filter, et_filter, streamflow_filter, det_sm, det_swe)
        if date_residual is not None:
            residual = pd.DataFrame(residual, index=date_residual)
        else:
            residual = pd.DataFrame(residual, index=pd.date_range(start="19980101", end=None, periods=len(residual), freq='D'))

        # water
        res_all, res_quantile_all, res_std_all, res_abs_mean_all = self.waterClosureIndex(residual)
        
        return res_all, res_quantile_all, res_std_all, res_abs_mean_all
        
    def unitConversion(self, streamflow, pre, et, sm1, sm2, sm3, sm4, swe,
                       basinArea, layer1, layer2, layer3, layer4):
        # streamflow: feet2meter
        factor_feet2meter = 0.0283168

        # streamflow: vloume2height
        factor_vloume2height = 1 / basinArea
        
        # streamflow: s2day
        factor_s2day = 24*3600
        
        # # streamflow: m2mm
        factor_m2mm = 1000
        
        # streamflow: conversion
        streamflow = streamflow * factor_feet2meter * factor_vloume2height * factor_s2day * factor_m2mm
        
        # sm: v2h, note layer is  the height of m
        sm1 = sm1 * layer1
        sm2 = sm2 * layer2
        sm3 = sm3 * layer3
        sm4 = sm4 * layer4
        
        # sm: m2mm
        sm1 = sm1 * 1000
        sm2 = sm2 * 1000
        sm3 = sm3 * 1000
        sm4 = sm4 * 1000
        
        return streamflow, pre, et, sm1, sm2, sm3, sm4, swe

        
    @staticmethod
    def filter_period(vector, window=2):
        """ filter for period within a window, such as Pre, ET, streamflow """
        if vector is not None:
            vector = np.array(vector)
            
            # weight
            unit_weight = 1 / (window - 1) / 2
            weight_margin = unit_weight
            weight_main = unit_weight * 2
            weight = [weight_margin]
            weight.extend([weight_main for i in range(window - 2)])
            weight.append(weight_margin)
            weight = np.array(weight)
            
            filter_func = lambda x: sum(x * weight)
            
            vector_filtering = [filter_func(vector[i: i + window]) for i in range(len(vector) - window + 1)]
            
            vector_filtering = np.array(vector_filtering)
        else:
            vector_filtering = None
        
        return vector_filtering
    
    @staticmethod
    def filter_differential(vector, window=2):
        """ filter for differential within a window, such as detSM, detSWE in detS """
        if vector is not None:
            vector = np.array(vector)
            
            filter_func = lambda x: (x[-1] - x[0]) / (window - 1)
            
            vector_filtering = [filter_func(vector[i: i + window]) for i in range(len(vector) - window + 1)]  
            
            vector_filtering = np.array(vector_filtering)
        else:
            vector_filtering = None
        
        return vector_filtering

    @staticmethod
    def waterBalanceEquation(pre, et, streamflow, det_sm, det_swe):
        IO_system = np.array(pre) - np.array(et) - np.array(streamflow)
        det_TWS = np.array(det_sm) + np.array(det_swe)
        res = IO_system - det_TWS
        return res
    
    @staticmethod
    def waterClosureIndex(res):
        res_monthly = res.resample("M").sum()
        res_yearly = res.resample("Y").sum()
        scales = ["daily", "monthly", "yearly"]
        res_all = dict()
        res_quantile_all = dict()
        res_std_all = dict()
        res_abs_mean_all = dict()
        for res, scale in zip([res, res_monthly, res_yearly], scales):
            res_quantile = np.percentile(res, (0, 25, 50, 75, 100), interpolation='midpoint')  # min, 25, mean, 75, max
            res_std = np.std(res.values, ddof=1)
            res_abs_mean = np.mean(abs(res))
            
            res_all[scale] = res
            res_quantile_all[scale] = res_quantile
            res_std_all[scale] = res_std
            res_abs_mean_all[scale] = res_abs_mean
        
        return res_all, res_quantile_all, res_std_all, res_abs_mean_all
    
    @staticmethod
    def formatNPArray(array_like):
        return np.array(array_like)

    def plot(self, res, streamflow_filter, pre_filter, et_filter, det_sm):
        plt.plot(res)
        # plt.plot(res / pre_filter)
        plt.plot(pre_filter)
        plt.plot(streamflow_filter)
        plt.plot(et_filter)
        plt.plot(det_sm)


class WaterBalanceAnalysis_correct(WaterBalanceAnalysis):
    
    def __call__(self, streamflow, pre, et, sm, swe, basinArea, window, date_residual=None, factor_feet2meter=0.0283168):
        # unit conversion
        streamflow = self.unitConversion(streamflow, basinArea, factor_feet2meter)
        
        # filtering
        streamflow_filter = self.filter_period(streamflow, window)
        pre_filter = self.filter_period(pre, window)
        et_filter = self.filter_period(et, window)
        det_sm = self.filter_differential(sm, window)
        det_swe = self.filter_differential(swe, window)

        # water balance
        residual = self.waterBalanceEquation(pre_filter, et_filter, streamflow_filter, det_sm, det_swe)
        
        if date_residual is not None:
            residual = pd.DataFrame(residual, index=date_residual)
        else:
            residual = pd.DataFrame(residual, index=pd.date_range(start="19980101", end=None, periods=len(residual), freq='D'))

        # water
        res_all, res_quantile_all, res_std_all, res_abs_mean_all = self.waterClosureIndex(residual)
        
        return res_all, res_quantile_all, res_std_all, res_abs_mean_all
    
    def unitConversion(self, streamflow, basinArea, factor_feet2meter=0.0283168):
        # sm, swe have been handled in forcing_Preparation.unitFormat, only unit conversion for streamflow (not qt, qt is treated as m3/s in forcing_Preparation.unitFormat)
        # streamflow: feet2meter: factor_feet2meter=0.0283168(feet2meter) or 1 (meter2meter)

        # streamflow: vloume2height
        factor_vloume2height = 1 / (basinArea * 1000000) # m3 -> mm (area: km2 -> m2)
        
        # streamflow: s2day
        factor_s2day = 24*3600
        
        # # streamflow: m2mm
        factor_m2mm = 1000
        
        # streamflow: conversion
        streamflow = streamflow * factor_feet2meter * factor_vloume2height * factor_s2day * factor_m2mm
        
        return streamflow
    

def calResidual(forcing, factor_feet2meter=0.0283168):
    # waterbalance analysis
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    
    wbac = WaterBalanceAnalysis_correct()
    res_all, res_quantile_all, res_std_all, res_abs_mean_all = wbac(forcing.streamflow, forcing.prec, forcing.E,
                                                                forcing.all_sm, forcing.swe, basinArea,
                                                                window=2, date_residual=pd.to_datetime(forcing.index[1:]),
                                                                factor_feet2meter=factor_feet2meter)
    
    return res_all, res_quantile_all, res_std_all, res_abs_mean_all


def analysisGapInSWE():
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # read swe
    swe_list = [pd.read_csv(fp, index_col=0).loc[:, "swe"].values for fp in fpaths]
    r = pd.read_csv(fpaths[1], index_col=0).loc[:, ["date", "swe"]]
    r.swe.dropna(axis=0)
    r.index = pd.to_datetime(r.date, format="%Y%m%d")
    r.groupby(r.index.month)["swe"].mean()
    
    
    # loop for analysis missing case
    before_window_all = []
    after_window_all = []
    for i in tqdm(range(len(swe_list)), desc="loop for basins to analysis Gap", colour="green"):
        swe = swe_list[i]
        swe_nan_bools = np.isnan(swe)
        swe_nan_index = np.where(swe_nan_bools)
        
        for j in range(len(swe_nan_index[0])):
            try:
                before_index = np.where(~np.isnan(swe[:swe_nan_index[0][j]]))[0][-1]
            except:
                before_index = np.nan
            try:
                after_index = np.where(~np.isnan(swe[swe_nan_index[0][j] + 1:]))[0][0]
            except:
                after_index = np.nan
                
            before_window = swe_nan_index[0][j] - before_index  # 1 [29, nan, 30]
            after_window = after_index + 1 - 0  # 1 [29, nan, 30]

            before_window_all.append(before_window)
            after_window_all.append(after_window)
    
    before_window_all = np.array(before_window_all)
    after_window_all = np.array(after_window_all)
    before_window_all = before_window_all[~np.isnan(before_window_all)]
    after_window_all = after_window_all[~np.isnan(after_window_all)]
    
    set_before_window = set(before_window_all)
    set_after_window = set(after_window_all)
    
    # before_window_all = before_window_all[before_window_all <= 20]
    # after_window_all = after_window_all[after_window_all <= 20]
    
    # plt.hist2d(before_window_all, after_window_all)
    # plt.hist(before_window_all)
    # plt.hist(after_window_all)
    # plt.boxplot([before_window_all, after_window_all], labels=["before_window", "after_window"])
    

def gapFillingSWE(swe, gap_filling_window=2, plot=False):
    swe_filling = copy(swe)
    swe_filling.index = pd.to_datetime(swe_filling.date, format="%Y%m%d")
    climatology_swe = swe_filling.groupby([swe_filling.index.month, swe_filling.index.day])["swe"].mean()
    
    swe_array = swe_filling.loc[:, "swe"].values
    bool_nan = np.isnan(swe_array)
    index_nan = np.where(bool_nan)[0]
    
    for i in range(len(index_nan)):
        try:
            # gapFilling1: linear interpolate
            # get array in the gap_filling_window
            index = index_nan[i]
            gap_filling_window_array = swe_array[index - gap_filling_window: index + gap_filling_window + 1]
            
            # interp1d function
            x = np.array(list(range(len(gap_filling_window_array))))
            y = gap_filling_window_array
            
            # remove nan
            nan_index_y = np.isnan(y)
            x_ = x[~nan_index_y]
            y_ = y[~nan_index_y]
            

            interp_function = interp1d(x_, y_, kind="linear")
            
            # interp
            swe_array[index] = interp_function(gap_filling_window)
            
        except:
            # gapFilling2: climatology
            climatology_swe_ = climatology_swe[(swe_filling.index[index].month, swe_filling.index[index].day)]
            swe_array[index] = climatology_swe_
            
            if np.isnan(swe_array[index]):
                # gapFilling3: 0
                swe_array[index] = 0
    
    # update
    swe.loc[:, "swe_filling"] = swe_filling.swe.values
    
    # plot
    if plot:
        font = {'family': "Arial", 'weight': 'normal', 'size': 5}
        fig, ax = plt.subplots(figsize=(3,2), dpi=300)
        date = pd.to_datetime(swe.date, format="%Y%m%d")
        ax.scatter(date, swe.swe,
                   c="dodgerblue", # royalblue
                   label="Original data",
                   s=0.6, marker="o",
                   alpha=0.3, zorder=10)
        # ax.plot(date, swe.swe, "b-", label="Original data", linewidth=0.5, zorder=10)
        ax.scatter(date[index_nan], swe.swe_filling[index_nan],
                   c="r",
                   label="Gap Filling data",
                   s=0.4, marker="o",
                   zorder=10)
        ax.plot(date, swe.swe_filling, "gray",
                label="all data", linewidth=0.3, zorder=2)
        ax.set_xlim([date.iloc[0], date.iloc[-1]])
        ax.tick_params(labelsize=font["size"], direction='in', width=0.5, length=2)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_font(font) for label in labels]
        ax.spines[:].set_linewidth(0.5)
        plt.show()
    else:
        fig = None
    
    return swe, fig

def waterBalanceAanalysis(dpc_base):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    waterClosureIndex_list = []
    
    # loop for basins to analysis
    for i in tqdm(range(len(fnames)), desc="loop for basins to analysis", colour="green"):
        
        # read
        fpath = fpaths[i]
        fname = fnames[i]
        df_basin = pd.read_csv(fpath, index_col=0)
        
        # extract basinArea
        basin_iindex = int(fname[: fname.find("_")])
        basinArea = dpc_base.basin_shp.loc[basin_iindex, "AREA"]
        
        # set layer
        layer1 = 0.07
        layer2 = 0.21
        layer3 = 0.72
        layer4 = 1.89
        
        # extract variable
        streamflow = df_basin.loc[:, "streamflow"]
        pre = df_basin.loc[:, "precipitation"]
        et = df_basin.loc[:, "E"]
        sm1 = df_basin.loc[:, "swvl1"]
        sm2 = df_basin.loc[:, "swvl2"]
        sm3 = df_basin.loc[:, "swvl3"]
        sm4 = df_basin.loc[:, "swvl4"]
        swe_ = df_basin.loc[:, ["date", "swe"]]
        
        # gap filling swe
        swe_, _ = gapFillingSWE(swe_, plot=True)
        swe = swe_.loc[:, "swe_filling"]
        # swe = None
        
        # water balance analysis
        wba = WaterBalanceAnalysis()
        res_all, res_quantile_all, res_std_all, res_abs_mean_all = wba(streamflow, pre, et, sm1, sm2, sm3, sm4, swe, basinArea, layer1, layer2, layer3, layer4, window=2)
        
        # append
        # waterClosureIndex_list.append(waterClosureIndex)
    
    # dpc_base.basin_shp["waterClosureIndex"] = waterClosureIndex_list
    
    # plot

def SWE_gapFilling_Plot(dpc_base):
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis/basins"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    dst_home = "F:/research/WaterBudgetClosure/SWE_Gap_Filling"
    
    # random index
    random_index = [0, 10, 50, 100, 200, 300, 400, 500, 600] # np.random.randint(0, len(fnames), 9)
    fnames = np.array(fnames)[random_index]
    fpaths = np.array(fpaths)[random_index]
    
    # loop for basins to analysis
    for i in tqdm(range(len(fnames)), desc="loop for basins to analysis", colour="green"):
        
        # read
        fpath = fpaths[i]
        fname = fnames[i]
        dst_name = fname[:fname.find('_')] + fname[fname.rfind('_'): fname.find('.')] + "_swe_gapfilling" + ".svg"
        
        df_basin = pd.read_csv(fpath, index_col=0)
        
        # extract basinArea
        basin_index = int(fname[: fname.find("_")])
        basinArea = dpc_base.basin_shp.loc[basin_index, "AREA"]
        
        # extract variable
        swe_ = df_basin.loc[:, ["date", "swe"]]
        
        # gap filling swe
        swe_, fig = gapFillingSWE(swe_, plot=True)
        
        # fig.savefig(os.path.join(dst_home, dst_name))
        

if __name__ == "__main__":
    # 1998-2010
    analysis_period = ["19980101", "20101231"]
    root, home = setHomePath(root="E:")
    subdir = "WaterBalanceAnalysis"
    
    # review all basins
    # dpc_review_all_basins = dataProcess_CAMELS_read_basin_grid(home, subdir, analysis_period)
    # dpc_review_all_basins(plot=False)
    
    # instance dpc_base
    read_dpc_base_from_file = True
    if read_dpc_base_from_file:
        with open(os.path.join(home, "dataPreprocess_CAMELS", subdir, "dpc_base.pkl"), "rb") as f:
            dpc_base = pickle.load(f)
    else:
        instance_dataProcess_CAMELS_read_basin_grid_removeStreamflowMissing(home, subdir, analysis_period)
    
    # remove
    # remove_id = [int(rfm["fn"][:rfm["fn"].find("_")]) for rfm in dpc_base.remove_files_Missing]  # remove 19 but 18 is real, 9536100 not in bains

    # iteratre Basin
    # dataProcess_CAMELS_iterateBasin(dpc_base, home, subdir, analysis_period)
    
    # patch
    # patch_addswe_dataProcess_CAMELS_iterateBasin(dpc_base)
    # path_remove_columns("swe")
    # patch_addGLEAM_Ep_dataProcess_CAMELS_iterateBasin(dpc_base, home, subdir, analysis_period)
    # path_remove_columns(column=["Ep", "Ep_x", "Ep_y"])
    
    # analysisGapInSWE
    # analysisGapInSWE()
    
    # waterBalance
    # waterBalanceAanalysis(dpc_base)
    
    # SWE_gapFilling_Plot
    SWE_gapFilling_Plot(dpc_base)
    

 