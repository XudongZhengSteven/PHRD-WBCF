# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WaterBudgetClosure import dataPreprocess_CAMELS
from WaterBudgetClosure.dataPreprocess_CAMELS_functions import setHomePath
from WaterBudgetClosure.WaterBalanceAnalysis import WaterBalanceAnalysis
import pickle
from tqdm import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from WaterBudgetClosure.lumod_hbv.BuildModel import Lumod_HBV
from WaterBudgetClosure.lumod_hbv import CalibrateModel
from copy import deepcopy
from lumod.tools import metrics
from deap import creator, base, tools, algorithms
from WaterBudgetClosure.decomposition_residuals import read_decomposition_func
import matplotlib
import operator

def read_correct_result(basin_name="0_forcing_basin_1013500"):
    result_home = "F:/research/WaterBudgetClosure/Correct/Basins"
    correct_dict = {}
    keys = ["iter_correct",
            "forcing_correct_list",
            "best_params_dict_correct_list",
            "best_fitness_correct_list",
            "weighted_best_fitness_correct_list",
            "sim_correct_list",
            "front_correct_list",
            "gridPop_correct_list",
            "res_correct_list",
            "res_abs_mean_correct_list",
            "leak_error_abs_mean_all_correct_list"
            ]
    for key in keys:
        with open(os.path.join(result_home, f"{basin_name}_{key}.pkl"), "rb") as f:
            correct_dict[key] = pickle.load(f)
    
    return correct_dict
    
    
def plot_correct_dict(correct_dict, save_prefix):
    # correct_dict = {"forcing_correct_list": forcing_correct_list,
    #         "best_params_dict_correct_list": best_params_dict_correct_list,
    #         "best_fitness_correct_list": best_fitness_correct_list,
    #         "sim_correct_list": sim_correct_list,
    #         "front_correct_list": front_correct_list,
    #         "gridPop_correct_list": gridPop_correct_list,
    #         "res_correct_list": res_correct_list,
    #         "res_abs_mean_correct_list": res_abs_mean_correct_list,
    #         "leak_error_abs_mean_all_correct_list": leak_error_abs_mean_all_correct_list,
    #             }
    iter_correct = correct_dict["iter_correct"]
    forcing_correct_list = correct_dict["forcing_correct_list"]
    best_params_dict_correct_list = correct_dict["best_params_dict_correct_list"]
    best_fitness_correct_list = correct_dict["best_fitness_correct_list"]
    weighted_best_fitness_correct_list = correct_dict["weighted_best_fitness_correct_list"]
    sim_correct_list = correct_dict["sim_correct_list"]
    front_correct_list = correct_dict["front_correct_list"]
    gridPop_correct_list = correct_dict["gridPop_correct_list"]
    res_correct_list = correct_dict["res_correct_list"]
    res_abs_mean_correct_list = correct_dict["res_abs_mean_correct_list"]
    leak_error_abs_mean_all_correct_list = correct_dict["leak_error_abs_mean_all_correct_list"]
    
    fig = plt.figure()
    plt.plot(weighted_best_fitness_correct_list)
    plt.xlabel("correct iterations")
    plt.ylabel("weighted KGE")
    fig.savefig(f"{save_prefix}_weighted_best_fitness_correct_list.png")
    
    fig, ax = plt.subplots(3, 1, sharex="col")
    ax[0].plot(range(len(res_abs_mean_correct_list)), [r["daily"] for r in res_abs_mean_correct_list])
    ax[1].plot(range(len(res_abs_mean_correct_list)), [r["monthly"] for r in res_abs_mean_correct_list])
    ax[2].plot(range(len(res_abs_mean_correct_list)), [r["yearly"] for r in res_abs_mean_correct_list])
    ax[0].set_ylabel("mean daily residual")
    ax[1].set_ylabel("mean monthly residual")
    ax[2].set_ylabel("mean yearly residual")
    ax[2].set_xlabel("correct iterations")
    fig.savefig(f"{save_prefix}_res_abs_mean_correct_list.png")
    
    fig, ax = plt.subplots(3, 1, sharex="col")
    ax[0].plot(range(len(leak_error_abs_mean_all_correct_list[1:])), [r["daily"] for r in leak_error_abs_mean_all_correct_list[1:]])
    ax[1].plot(range(len(leak_error_abs_mean_all_correct_list[1:])), [r["monthly"] for r in leak_error_abs_mean_all_correct_list[1:]])
    ax[2].plot(range(len(leak_error_abs_mean_all_correct_list[1:])), [r["yearly"] for r in leak_error_abs_mean_all_correct_list[1:]])
    ax[0].set_ylabel("mean daily leak")
    ax[1].set_ylabel("mean monthly leak")
    ax[2].set_ylabel("mean yearly leak")
    ax[2].set_xlabel("correct iterations")
    fig.savefig(f"{save_prefix}_leak_error_abs_mean_all_correct_list.png")
    
    from Calibration.Calibration_Deap import Calibration_NSGAII_multi_obj
    ca = Calibration_NSGAII_multi_obj()
    for i in range(1, len(gridPop_correct_list)):
        gridPop_correct = gridPop_correct_list[i]
        front_correct = front_correct_list[i]
        save = f"{save_prefix}_front_correct_iter{i}"
        ca.plot_result_3d(gridPop_correct, front_correct, save, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    
    fig, ax = plt.subplots(3, 1, sharex="col")
    for i in range(1, len(forcing_correct_list) - 1):
        ax[0].plot(forcing_correct_list[i].prec, "gray", alpha=0.5)
        ax[1].plot(forcing_correct_list[i].sm, "gray", alpha=0.5)
        ax[2].plot(forcing_correct_list[i].swe, "gray", alpha=0.5)
    
    ax[0].plot(forcing_correct_list[0].prec, "b", label="start")
    ax[1].plot(forcing_correct_list[0].sm, "b", label="start")
    ax[2].plot(forcing_correct_list[0].swe, "b", label="start")

    ax[0].plot(forcing_correct_list[-1].prec, "r", label="end")
    ax[1].plot(forcing_correct_list[-1].sm, "r", label="end")
    ax[2].plot(forcing_correct_list[-1].swe, "r", label="end")
    
    ax[0].set_ylabel("pre")
    ax[1].set_ylabel("sm")
    ax[2].set_ylabel("swe")
    ax[2].set_xlabel("date")
    
    plt.legend()
    fig.savefig(f"{save_prefix}_forcing_correct_list.png")
    
    fig, ax = plt.subplots(3, 1, sharex="col")
    for i in range(1, len(res_correct_list) - 1):
        ax[0].plot(res_correct_list[i]["daily"], "gray", alpha=0.5)
        ax[1].plot(res_correct_list[i]["monthly"], "gray", alpha=0.5)
        ax[2].plot(res_correct_list[i]["yearly"], "gray", alpha=0.5)
    
    ax[0].plot(res_correct_list[0]["daily"], "b", label="start")
    ax[1].plot(res_correct_list[0]["monthly"], "b", label="start")
    ax[2].plot(res_correct_list[0]["yearly"], "b", label="start")

    ax[0].plot(res_correct_list[-1]["daily"], "r", label="end")
    ax[1].plot(res_correct_list[-1]["monthly"], "r", label="end")
    ax[2].plot(res_correct_list[-1]["yearly"], "r", label="end")
    
    ax[0].hlines(y=0, xmin=res_correct_list[0]["daily"].index[0],
                 xmax=res_correct_list[0]["daily"].index[-1],
                 linestyle="--", color="k")
    ax[1].hlines(y=0, xmin=res_correct_list[0]["daily"].index[0],
                xmax=res_correct_list[0]["daily"].index[-1],
                linestyle="--", color="k")
    ax[2].hlines(y=0, xmin=res_correct_list[0]["daily"].index[0],
            xmax=res_correct_list[0]["daily"].index[-1],
            linestyle="--", color="k")
    
    # start mean
    ax[0].hlines(y=abs(res_correct_list[0]["daily"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                 xmax=res_correct_list[0]["daily"].index[-1],
                 linestyle="--", color="b", label="start abs mean")
    ax[1].hlines(y=abs(res_correct_list[0]["monthly"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                xmax=res_correct_list[0]["daily"].index[-1],
                linestyle="--", color="b", label="start abs mean")
    ax[2].hlines(y=abs(res_correct_list[0]["yearly"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                xmax=res_correct_list[0]["daily"].index[-1],
                linestyle="--", color="b", label="start abs mean")
    
    # end mean
    ax[0].hlines(y=abs(res_correct_list[-1]["daily"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                 xmax=res_correct_list[0]["daily"].index[-1],
                 linestyle="--", color="r", label="end abs mean")
    ax[1].hlines(y=abs(res_correct_list[-1]["monthly"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                xmax=res_correct_list[0]["daily"].index[-1],
                linestyle="--", color="r", label="end abs mean")
    ax[2].hlines(y=abs(res_correct_list[-1]["yearly"]).mean(), xmin=res_correct_list[0]["daily"].index[0],
                xmax=res_correct_list[0]["daily"].index[-1],
                linestyle="--", color="r", label="end abs mean")

    ax[0].set_ylabel("daily")
    ax[1].set_ylabel("monthly")
    ax[2].set_ylabel("yearly")
    ax[2].set_xlabel("date")
    
    plt.legend()
    fig.savefig(f"{save_prefix}_res_correct_list.png")
        
    
def calibration_basin(model, forcing, save=None):
    # copy model
    model = deepcopy(model)
    
    # calibration
    front, gridPop, fig_result, param_names, param_types  = CalibrateModel.calibration_deap_NSGAII(model.model, forcing,
                                                                                                   start="20000101", end="20101231",
                                                                                                   maxGen=500, save=save)
    best_params, weighted_best_fitness = CalibrateModel.select_front_weighted_fitness(front, threshold_perc=50)
    best_fitness = best_params.fitness
    best_params = [param_types[i](best_params[i]) for i in range(len(best_params))]
    best_params_dict = dict(zip(param_names, best_params))
    
    # sim based on calibrated params
    s0 = forcing.loc[forcing.index[0], "sm"] / best_params_dict["fc"]
    s0 = s0 if s0 < 1 else 1
    s0 = s0 if s0 > 0 else 0
    sim = model.run(forcings=forcing,
                    s0=s0,
                    **best_params_dict)
    
    return best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop


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


def correct_model_input(forcing, sim, fc, correct_step=0.1):
    # res of sim: leak error
    # res of obs: leak error + inconsistency error
    # use pre to reduce leak error, namely, reduce res of sim
    # note: the signal of leak error is not must be same as inconsistency error
    # copy
    forcing = deepcopy(forcing)
    forcing_corrected = deepcopy(forcing)
    
    # init wbac
    wbac = WaterBalanceAnalysis_correct()
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    
    # sim res (leak error)
    forcing_sim = deepcopy(forcing)
    forcing_sim.all_sm = sim.all_sm
    forcing_sim.swe = sim.snow
    forcing_sim.streamflow = sim.qt
    forcing_sim.E = sim.et
    leak_all_sim, leak_quantile_all_sim, leak_std_all_sim, leak_abs_mean_all_sim = calResidual(forcing_sim, factor_feet2meter=1)

    # obs res (leak error + inconsistency error)
    # res_all_obs, res_quantile_all_obs, res_std_all_obs, res_abs_mean_all_obs = calResidual(forcing)
    
    # correct pre
    prec_period = wbac.filter_period(forcing_sim.prec, window=2)
    prec_period = pd.DataFrame(prec_period, index=pd.to_datetime(forcing.index[1:]))
    
    # monthly correct: daily will lead to many negative P values
    prec_period_monthly = prec_period.resample("M").sum()
    factor = -leak_all_sim["monthly"] / prec_period_monthly  # res = IO_system - det_TWS, if res > 0, IO should substract values to balance it
    factor *= correct_step
    factor += 1
    
    for index_ in forcing_sim.index:
        year = index_.year
        month = index_.month
        forcing_sim.loc[index_, "prec"] *= factor[(factor.index.year==year) & (factor.index.month==month)].values[0][0]
    
    # cal the leak_error after corrected pre (others keep same with sim)
    forcing_sim.prec[forcing_corrected.prec < 0] = 0
    
    leak_error_all_correct, leak_error_quantile_all_correct, leak_error_std_all_correct, leak_error_abs_mean_all_correct = calResidual(forcing_sim, 1)
    
    # save the corrected values
    forcing_corrected.prec = forcing_sim.prec
    
    return forcing_corrected, leak_error_all_correct, leak_error_abs_mean_all_correct


def correct_model_output(forcing, sim, fc, correct_step=0.1):
    # *do not correct qt/streamflow (see it as truth)
    # copy
    forcing = deepcopy(forcing)
    forcing_corrected = deepcopy(forcing)
    
    # init wbac
    wbac = WaterBalanceAnalysis_correct()
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    
    # sim res (leakage error)
    forcing_sim = deepcopy(forcing)
    forcing_sim.all_sm = sim.all_sm
    forcing_sim.swe = sim.snow
    forcing_sim.streamflow = sim.qt
    forcing_sim.E = sim.et
    res_all_sim, res_quantile_all_sim, res_std_all_sim, res_abs_mean_all_sim = calResidual(forcing_sim, factor_feet2meter=1)
    
    # obs res (leakage error + inconsistency error)
    res_all_obs, res_quantile_all_obs, res_std_all_obs, res_abs_mean_all_obs = calResidual(forcing)
    
    # inconsistency error = residual - leakage error
    
    # correct for output: et, swe
    forcing_corrected.E = forcing_corrected.E + (sim.et - forcing.E) * correct_step
    forcing_corrected.swe = forcing_corrected.swe + (sim.snow - forcing_corrected.swe) * correct_step
    
    # correct for output: sm and groundwater_reservoir  # TODO  使用相关性类型的校准, copula?
    sim_diff_sm = sim.sm - sim.sm.mean()
    forcing_corrected_diff_sm = forcing_corrected.sm - forcing_corrected.sm.mean()
    forcing_corrected.sm = forcing_corrected.sm + (sim_diff_sm - forcing_corrected_diff_sm) * correct_step
    
    sim_diff_groundwater_reservoir = sim.groundwater_reservoir - sim.groundwater_reservoir.mean()
    forcing_corrected_diff_groundwater_reservoir = forcing_corrected.groundwater_reservoir - forcing_corrected.groundwater_reservoir.mean()
    forcing_corrected.groundwater_reservoir = forcing_corrected.groundwater_reservoir + (sim_diff_groundwater_reservoir - forcing_corrected_diff_groundwater_reservoir) * correct_step
    
    # cal all_sm = sm + groundwater_reservoir
    forcing_corrected.all_sm = forcing_corrected.sm + forcing_corrected.groundwater_reservoir
    
    # cal the leak_error, #* note the identification of leak error (model sim leak error)
    leak_error_abs_mean_all_correct = res_abs_mean_all_sim
    leak_error_all_correct = res_all_sim

    return forcing_corrected, leak_error_all_correct, leak_error_abs_mean_all_correct


def correct_process(forcing, model, correct_function, correct_step):
    # calibration
    best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = calibration_basin(model, forcing, save=None)
    fc = best_params_dict["fc"]
    sim.loc[:, "sm"] = sim.ws * fc
    sim.loc[:, "groundwater_reservoir"] = sim.ws1 + sim.ws2
    sim.loc[:, "all_sm"] = sim.sm + sim.groundwater_reservoir
    
    # correct
    forcing_correct, leak_error_all_correct, leak_error_abs_mean_all_correct = correct_function(forcing, sim, fc, correct_step)
    
    return forcing_correct, leak_error_all_correct, leak_error_abs_mean_all_correct, best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop


def correct_citeria_input(iter_correct, leak_error_abs_mean_all_correct_list, weighted_best_fitness_correct_list, citeria_window, tolerance1=1.05, tolerance2=0.1):
    # correct_citeria_input 1
    # if iter_correct < citeria_window + 1:
    #     correct_bool = True
    # else:
    #     leak_error_abs_mean_last_window = [r["monthly"] for r in leak_error_abs_mean_all_correct_list[-citeria_window - 1: -1]]
    #     best_fitness_last_window = weighted_best_fitness_correct_list[-citeria_window - 1: -1]
        
    #     leak_error_abs_mean_current = leak_error_abs_mean_all_correct_list[-1]["monthly"]
    #     best_fitness_current = weighted_best_fitness_correct_list[-1]
        
    #     leak_last_window_mean = np.nanmean(leak_error_abs_mean_last_window)
    #     best_fitness_last_window_mean = np.nanmean(best_fitness_last_window)
        
    #     correct_bool_leak_o = leak_error_abs_mean_current <= leak_last_window_mean
    #     correct_bool_fitness_o = best_fitness_current >= best_fitness_last_window_mean
        
    #     correct_bool_leak = leak_error_abs_mean_current <= leak_last_window_mean * tolerance1 if correct_bool_fitness_o else correct_bool_leak_o
        
    #     # *note the fitness can be negative
    #     correct_bool_fitness = (best_fitness_current - best_fitness_last_window_mean) >= -tolerance2 if correct_bool_leak_o else correct_bool_fitness_o
        
    #     correct_bool_leak = True if np.isnan(np.nanmean(leak_error_abs_mean_last_window)) else correct_bool_leak
    #     correct_bool_fitness = True if np.isnan(np.nanmean(best_fitness_last_window)) else correct_bool_fitness
        
    #     correct_bool = correct_bool_leak and correct_bool_fitness
    
    # correct_citeria_input 2
    correct_bool = leak_error_abs_mean_all_correct_list[-1]["monthly"] > leak_error_abs_mean_all_correct_list[0]["monthly"] * 0.1
    
    return correct_bool


def correct_citeria_output(iter_correct, inconsistency_error_abs_mean_all_correct_list, weighted_best_fitness_correct_list, citeria_window, tolerance1=1.05, tolerance2=0.1):
    # correct_citeria_output 1
    # if iter_correct < citeria_window + 1:
    #     correct_bool = True
    # else:
    #     citeria_window = citeria_window if citeria_window < iter_correct else iter_correct
        
    #     inconsistency_abs_mean_last_window = [r["monthly"] for r in inconsistency_error_abs_mean_all_correct_list[-citeria_window - 1: -1]]
    #     best_fitness_last_window = weighted_best_fitness_correct_list[-citeria_window - 1: -1]
        
    #     inconsistency_abs_mean_current = inconsistency_error_abs_mean_all_correct_list[-1]["monthly"]
    #     best_fitness_current = weighted_best_fitness_correct_list[-1]
        
    #     inconsistency_last_window_mean = np.nanmean(inconsistency_abs_mean_last_window)
    #     best_fitness_last_window_mean = np.nanmean(best_fitness_last_window)
        
    #     correct_bool_inconsistency_o = inconsistency_abs_mean_current <= inconsistency_last_window_mean
    #     correct_bool_fitness_o = best_fitness_current >= best_fitness_last_window_mean
        
    #     correct_bool_inconsistency = inconsistency_abs_mean_current <= inconsistency_last_window_mean * tolerance1 if correct_bool_fitness_o else correct_bool_inconsistency_o
        
    #     # *note the fitness can be negative
    #     correct_bool_fitness = (best_fitness_current - best_fitness_last_window_mean) >= -tolerance2 if correct_bool_inconsistency_o else correct_bool_fitness_o
        
    #     correct_bool_inconsistency = True if np.isnan(inconsistency_last_window_mean) else correct_bool_inconsistency
    #     correct_bool_fitness = True if np.isnan(best_fitness_last_window_mean) else correct_bool_fitness

    #     correct_bool = correct_bool_inconsistency and correct_bool_fitness
    
    # correct_citeria_output 2
    correct_bool = inconsistency_error_abs_mean_all_correct_list[-1]["monthly"] > inconsistency_error_abs_mean_all_correct_list[0]["monthly"] * 0.1
    
    return correct_bool
    

def loop_correct(forcing, initial_decomposition, correct_step=0.1, citeria_window=1):
    # preprocess
    # copy forcing
    forcing = deepcopy(forcing)
    forcing.index = pd.to_datetime(forcing.index)
    
    # sm: layer1 + layer2 + layer3; layer3 + layer4
    forcing.loc[:, "sm"] = forcing.loc[:, "swvl1"] + forcing.loc[:, "swvl2"] + forcing.loc[:, "swvl3"] * 22/ 72
    forcing.loc[:, "groundwater_reservoir"] = forcing.loc[:, "swvl3"] * 50 / 72 + forcing.loc[:, "swvl4"]
    forcing.loc[:, "all_sm"] = forcing.sm + forcing.groundwater_reservoir
    
    # get info
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    basinLatCen = forcing.loc[forcing.index[0], "basinLatCen"]
    
    # initial model and warm-up
    model = Lumod_HBV(area=basinArea, lat=basinLatCen)
    print("model warmup".center(50, "-"))
    for i in range(5):
        model.warm_up(forcing, start="19980101", end="19991231")  # ["19980101", "20101231"]
        print(f'{model.model.params["s0"]}, {model.model.params["w01"]}, {model.model.params["w02"]}')
    
    # list for saveing correct process and set the initial value
    forcing_correct_list = [forcing]
    best_params_dict_correct_list = [initial_decomposition["calibration_res"]["best_params_dict"]]
    best_fitness_correct_list = [initial_decomposition["calibration_res"]["best_fitness"]]
    weighted_best_fitness_correct_list = [initial_decomposition["calibration_res"]["weighted_best_fitness"]]
    sim_correct_list = [initial_decomposition["calibration_res"]["sim"]]
    front_correct_list = [initial_decomposition["calibration_res"]["front"]]
    gridPop_correct_list = [initial_decomposition["calibration_res"]["gridPop"]]
    
    res_correct_list = [initial_decomposition["res"]["res_all"]]
    leak_correct_list = [initial_decomposition["leakage"]["leakage_all"]]
    inconsistency_correct_list = [initial_decomposition["inconsistency"]["inconsistency_all"]]
    
    res_abs_mean_correct_list = [initial_decomposition["res"]["res_abs_mean_all"]]
    leak_error_abs_mean_all_correct_list = [initial_decomposition["leakage"]["leakage_abs_mean_all"]]
    inconsistency_error_abs_mean_all_correct_list = [initial_decomposition["inconsistency"]["inconsistency_abs_mean_all"]]

    # correct model input: reduce leak error
    # first correct
    sim = initial_decomposition["calibration_res"]["sim"]
    best_params_dict = initial_decomposition["calibration_res"]["best_params_dict"]
    fc = best_params_dict["fc"]

    forcing_correct, leak_error_all_correct, leak_error_abs_mean_all_correct = correct_model_input(forcing, sim, fc, correct_step)
    res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing_correct)
    
    inconsistency_all = {}
    inconsistency_all["daily"] = res_all["daily"] - leak_error_all_correct["daily"]
    inconsistency_all["monthly"] = res_all["monthly"] - leak_error_all_correct["monthly"]
    inconsistency_all["yearly"] = res_all["yearly"] - leak_error_all_correct["yearly"]
    
    inconsistency_error_abs_mean_all_correct = {"daily": np.mean(abs(inconsistency_all["daily"])),
                                                "monthly": np.mean(abs(inconsistency_all["monthly"])),
                                                "yearly": np.mean(abs(inconsistency_all["yearly"]))}
    
    forcing_correct_list.append(forcing_correct)
    best_params_dict_correct_list.append(initial_decomposition["calibration_res"]["best_params_dict"])
    best_fitness_correct_list.append(initial_decomposition["calibration_res"]["best_fitness"])
    weighted_best_fitness_correct_list.append(initial_decomposition["calibration_res"]["weighted_best_fitness"])
    sim_correct_list.append(initial_decomposition["calibration_res"]["sim"])
    front_correct_list.append(initial_decomposition["calibration_res"]["front"])
    gridPop_correct_list.append(initial_decomposition["calibration_res"]["gridPop"])
    
    res_correct_list.append(res_all)
    leak_correct_list.append(leak_error_all_correct)
    inconsistency_correct_list.append(inconsistency_all)
    
    res_abs_mean_correct_list.append(res_abs_mean_all)
    leak_error_abs_mean_all_correct_list.append(leak_error_abs_mean_all_correct)
    inconsistency_error_abs_mean_all_correct_list.append(inconsistency_error_abs_mean_all_correct)
    
    # loop correct
    iter_correct = 2
    forcing_correct = forcing
    print("----correct model input to reduce leak error----")
    while True:
        if correct_citeria_input(iter_correct, leak_error_abs_mean_all_correct_list, weighted_best_fitness_correct_list, citeria_window):
            forcing_correct, leak_error_all_correct, leak_error_abs_mean_all_correct, best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = correct_process(forcing_correct,
                                                                                                                                                           model,
                                                                                                                                                           correct_model_input,
                                                                                                                                                           correct_step)
            # cal Residual (leakage error + inconsistency error)
            res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing_correct)
            
            # cal inconsistency error (Residual - leakage error) 
            inconsistency_all = {}
            inconsistency_all["daily"] = res_all["daily"] - leak_error_all_correct["daily"]
            inconsistency_all["monthly"] = res_all["monthly"] - leak_error_all_correct["monthly"]
            inconsistency_all["yearly"] = res_all["yearly"] - leak_error_all_correct["yearly"]
            
            inconsistency_error_abs_mean_all_correct = {"daily": np.mean(abs(inconsistency_all["daily"])),
                                                        "monthly": np.mean(abs(inconsistency_all["monthly"])),
                                                        "yearly": np.mean(abs(inconsistency_all["yearly"]))}
            
            # append
            forcing_correct_list.append(forcing_correct)
            best_params_dict_correct_list.append(best_params_dict)
            best_fitness_correct_list.append(best_fitness)
            weighted_best_fitness_correct_list.append(weighted_best_fitness)
            sim_correct_list.append(sim)
            front_correct_list.append(front)
            gridPop_correct_list.append(gridPop)
            
            res_correct_list.append(res_all)
            leak_correct_list.append(leak_error_all_correct)
            inconsistency_correct_list.append(inconsistency_all)
            
            res_abs_mean_correct_list.append(res_abs_mean_all)
            leak_error_abs_mean_all_correct_list.append(leak_error_abs_mean_all_correct)
            inconsistency_error_abs_mean_all_correct_list.append(inconsistency_error_abs_mean_all_correct)
            
            iter_correct += 1
            print(f"iter{iter_correct}: leak error({leak_error_abs_mean_all_correct}), best_fitness({best_fitness})")
            
        else:
            forcing_correct_list = forcing_correct_list[:-1]
            best_params_dict_correct_list = best_params_dict_correct_list[:-1]
            best_fitness_correct_list = best_fitness_correct_list[:-1]
            weighted_best_fitness_correct_list = weighted_best_fitness_correct_list[:-1]
            sim_correct_list = sim_correct_list[:-1]
            front_correct_list = front_correct_list[:-1]
            gridPop_correct_list = gridPop_correct_list[:-1]
            
            res_correct_list = res_correct_list[:-1]
            leak_correct_list = leak_correct_list[:-1]
            inconsistency_correct_list = inconsistency_correct_list[:-1]
            
            res_abs_mean_correct_list = res_abs_mean_correct_list[:-1]
            leak_error_abs_mean_all_correct_list = leak_error_abs_mean_all_correct_list[:-1]
            inconsistency_error_abs_mean_all_correct_list = inconsistency_error_abs_mean_all_correct_list[:-1]
            break
    
    iter_correct_input = iter_correct
    # correct model output: reduce inconsistency error
    # first correct
    sim = sim_correct_list[-1]
    best_params_dict = best_params_dict_correct_list[-1]
    fc = best_params_dict["fc"]
    
    forcing_corrected, leak_error_all_correct, leak_error_abs_mean_all_correct = correct_model_output(forcing, sim, fc, correct_step=0.1)
    res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing_correct)
    
    inconsistency_all = {}
    inconsistency_all["daily"] = res_all["daily"] - leak_error_all_correct["daily"]
    inconsistency_all["monthly"] = res_all["monthly"] - leak_error_all_correct["monthly"]
    inconsistency_all["yearly"] = res_all["yearly"] - leak_error_all_correct["yearly"]
    
    inconsistency_error_abs_mean_all_correct = {"daily": np.mean(abs(inconsistency_all["daily"])),
                                                "monthly": np.mean(abs(inconsistency_all["monthly"])),
                                                "yearly": np.mean(abs(inconsistency_all["yearly"]))}
    
    forcing_correct_list.append(forcing_correct)
    best_params_dict_correct_list.append(best_params_dict)
    best_fitness_correct_list.append(best_fitness_correct_list[-1])
    weighted_best_fitness_correct_list.append(weighted_best_fitness_correct_list[-1])
    sim_correct_list.append(sim_correct_list[-1])
    front_correct_list.append(front_correct_list[-1])
    gridPop_correct_list.append(gridPop_correct_list[-1])
    
    res_correct_list.append(res_all)
    leak_correct_list.append(leak_error_all_correct)
    inconsistency_correct_list.append(inconsistency_all)
    
    res_abs_mean_correct_list.append(res_abs_mean_all)
    leak_error_abs_mean_all_correct_list.append(leak_error_abs_mean_all_correct)
    inconsistency_error_abs_mean_all_correct_list.append(inconsistency_error_abs_mean_all_correct)
    
    iter_correct = 2
    forcing_correct = forcing_correct_list[-1]
    print("----correct model output to reduce inconsistency error----")
    while True:
        if correct_citeria_output(iter_correct, inconsistency_error_abs_mean_all_correct_list, weighted_best_fitness_correct_list, citeria_window):
            forcing_correct, leak_error_all_correct, leak_error_abs_mean_all_correct, best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = correct_process(forcing_correct,
                                                                                                                                                           model,
                                                                                                                                                           correct_model_output,
                                                                                                                                                           correct_step)
            
            # cal Residual (leakage error + inconsistency error)
            res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing_correct)
            
            # cal inconsistency error (Residual - leakage error) 
            inconsistency_all = {}
            inconsistency_all["daily"] = res_all["daily"] - leak_error_all_correct["daily"]
            inconsistency_all["monthly"] = res_all["monthly"] - leak_error_all_correct["monthly"]
            inconsistency_all["yearly"] = res_all["yearly"] - leak_error_all_correct["yearly"]
            
            inconsistency_error_abs_mean_all_correct = {"daily": np.mean(abs(inconsistency_all["daily"])),
                                                        "monthly": np.mean(abs(inconsistency_all["monthly"])),
                                                        "yearly": np.mean(abs(inconsistency_all["yearly"]))}
            
            # append
            forcing_correct_list.append(forcing_correct)
            best_params_dict_correct_list.append(best_params_dict)
            best_fitness_correct_list.append(best_fitness)
            weighted_best_fitness_correct_list.append(weighted_best_fitness)
            sim_correct_list.append(sim)
            front_correct_list.append(front)
            gridPop_correct_list.append(gridPop)
            
            res_correct_list.append(res_all)
            leak_correct_list.append(leak_error_all_correct)
            inconsistency_correct_list.append(inconsistency_all)
            
            res_abs_mean_correct_list.append(res_abs_mean_all)
            leak_error_abs_mean_all_correct_list.append(leak_error_abs_mean_all_correct)
            inconsistency_error_abs_mean_all_correct_list.append(inconsistency_error_abs_mean_all_correct)
            
            iter_correct += 1
            print(f"iter{iter_correct}: inconsistency error({inconsistency_error_abs_mean_all_correct}), best_fitness({best_fitness})")
            
        else:        
            forcing_correct_list = forcing_correct_list[:-1]
            best_params_dict_correct_list = best_params_dict_correct_list[:-1]
            best_fitness_correct_list = best_fitness_correct_list[:-1]
            weighted_best_fitness_correct_list = weighted_best_fitness_correct_list[:-1]
            sim_correct_list = sim_correct_list[:-1]
            front_correct_list = front_correct_list[:-1]
            gridPop_correct_list = gridPop_correct_list[:-1]
            
            res_correct_list = res_correct_list[:-1]
            leak_correct_list = leak_correct_list[:-1]
            inconsistency_correct_list = inconsistency_correct_list[:-1]
            
            res_abs_mean_correct_list = res_abs_mean_correct_list[:-1]
            leak_error_abs_mean_all_correct_list = leak_error_abs_mean_all_correct_list[:-1]
            inconsistency_error_abs_mean_all_correct_list = inconsistency_error_abs_mean_all_correct_list[:-1]
            break
    
    iter_correct_output = iter_correct
    iter_correct = {"iter_correct_input": iter_correct_input,
                    "iter_correct_output": iter_correct_output}
    # dict
    correct_dict = {"iter_correct": iter_correct,
                    "forcing_correct_list": forcing_correct_list,
                    "best_params_dict_correct_list": best_params_dict_correct_list,
                    "best_fitness_correct_list": best_fitness_correct_list,
                    "weighted_best_fitness_correct_list": weighted_best_fitness_correct_list,
                    "sim_correct_list": sim_correct_list,
                    "front_correct_list": front_correct_list,
                    "gridPop_correct_list": gridPop_correct_list,
                    "res_correct_list": res_correct_list,
                    "leak_correct_list": leak_correct_list,
                    "inconsistency_correct_list": inconsistency_correct_list,
                    "res_abs_mean_correct_list": res_abs_mean_correct_list,
                    "leak_error_abs_mean_all_correct_list": leak_error_abs_mean_all_correct_list,
                    "inconsistency_error_abs_mean_all_correct_list": inconsistency_error_abs_mean_all_correct_list
                    }
    
    return correct_dict


def Decomposition_residual(forcing):
    # copy forcing
    forcing = deepcopy(forcing)
    forcing.index = pd.to_datetime(forcing.index)
    
    # sm: layer1 + layer2 + layer3; layer3 + layer4
    forcing.loc[:, "sm"] = forcing.loc[:, "swvl1"] + forcing.loc[:, "swvl2"] + forcing.loc[:, "swvl3"] * 22/ 72
    forcing.loc[:, "groundwater_reservoir"] = forcing.loc[:, "swvl3"] * 50 / 72 + forcing.loc[:, "swvl4"]
    forcing.loc[:, "all_sm"] = forcing.sm + forcing.groundwater_reservoir
    
    # get info
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    basinLatCen = forcing.loc[forcing.index[0], "basinLatCen"]
    
    # initial model and warm-up
    model = Lumod_HBV(area=basinArea, lat=basinLatCen)
    print("model warmup".center(50, "-"))
    
    for i in range(5):
        model.warm_up(forcing, start="19980101", end="19991231")  # ["19980101", "20101231"]
        print(f'{model.model.params["s0"]}, {model.model.params["w01"]}, {model.model.params["w02"]}')
    
    # calibration
    best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = calibration_basin(model, forcing, save=None)
    print(f"best fitness: {best_fitness}")
    print(sim)
    fc = best_params_dict["fc"]
    sim.loc[:, "sm"] = sim.ws * fc
    sim.loc[:, "groundwater_reservoir"] = sim.ws1 + sim.ws2
    sim.loc[:, "all_sm"] = sim.sm + sim.groundwater_reservoir
    
    calibration_res = {"best_params_dict": best_params_dict,
                       "weighted_best_fitness": weighted_best_fitness,
                       "best_fitness": best_fitness,
                       "sim": sim,
                       "front": front,
                       "gridPop": gridPop
                       }

    # init wbac
    wbac = WaterBalanceAnalysis_correct()
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    
    # obs res (leak error + inconsistency error)
    res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing)
    res = {"res_all": res_all,
           "res_quantile_all": res_quantile_all,
           "res_std_all": res_std_all,
           "res_abs_mean_all": res_abs_mean_all
           }
    
    # sim res (leakage error)
    forcing_sim = deepcopy(forcing)
    forcing_sim.all_sm = sim.all_sm
    forcing_sim.swe = sim.snow
    forcing_sim.streamflow = sim.qt
    forcing_sim.E = sim.et
    leakage_all, leakage_quantile_all, leakage_std_all, leakage_abs_mean_all = calResidual(forcing_sim, factor_feet2meter=1)
    leakage = {"leakage_all": leakage_all,
               "leakage_quantile_all": leakage_quantile_all,
               "leakage_std_all": leakage_std_all,
               "leakage_abs_mean_all": leakage_abs_mean_all
               }

    # obs res - sim res (inconsistency error)
    inconsistency_all = dict()
    inconsistency_quantile_all = dict()
    inconsistency_std_all = dict()
    inconsistency_abs_mean_all = dict()
    
    for key in res_all.keys():
        inconsistency_ = res_all[key] - leakage_all[key]
        inconsistency_all[key] = inconsistency_
        inconsistency_quantile_all[key] = np.percentile(inconsistency_, (0, 25, 50, 75, 100), interpolation='midpoint')
        inconsistency_std_all[key] = np.std(inconsistency_.values, ddof=1)
        inconsistency_abs_mean_all[key] = np.mean(abs(inconsistency_))
    
    inconsistency = {"inconsistency_all": inconsistency_all,
                     "inconsistency_quantile_all": inconsistency_quantile_all,
                     "inconsistency_std_all": inconsistency_std_all,
                     "inconsistency_abs_mean_all": inconsistency_abs_mean_all
                     }
    
    return res, leakage, inconsistency, calibration_res

def Decomposition_residual_basins():
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # loop for basins to build model
    for i in tqdm(range(len(fnames)), desc="loop for basins to build model", colour="green"):

        # read
        fpath = fpaths[i]
        fname = fnames[i]
        
        forcing = pd.read_csv(fpath, index_col=0)
        
        # decomposition residual
        res, leakage, inconsistency, calibration_res = Decomposition_residual(forcing)
        
        decomposition_basin = {"res": res,
                               "leakage": leakage,
                               "inconsistency": inconsistency,
                               "calibration_res": calibration_res
                               }
        
        # save
        save_home = "F:/research/WaterBudgetClosure/Decomposition/Basins"
        with open(os.path.join(save_home, f"{fname[: fname.find('.')]}_decomposition.pkl"), "wb") as f:
            pickle.dump(decomposition_basin, f)
            

def select_basins_to_correct(plot=True):
    # read decomposition_dict
    decomposition_dict = read_decomposition_func(f"F:/research/WaterBudgetClosure/Decomposition")
    basin_index = np.array([key for key in decomposition_dict.keys()])
    
    # data formation
    res_monthly_list = [decomposition_dict[key]["res"]["monthly"] for key in decomposition_dict]
    leakage_monthly_list = [decomposition_dict[key]["leakage"]["monthly"] for key in decomposition_dict]
    inconsistency_monthly_list = [decomposition_dict[key]["inconsistency"]["monthly"] for key in decomposition_dict]
    
    res_monthly_mean = np.array([res_.mean() for res_ in res_monthly_list])  # multi-year mean
    leakage_monthly_mean = np.array([leakage_.mean() for leakage_ in leakage_monthly_list])
    inconsistency_monthly_mean = np.array([inconsistency_.mean() for inconsistency_ in inconsistency_monthly_list])
    
    res_monthly_std = np.array([res_.std() for res_ in res_monthly_list])  # multi-year std
    leakage_monthly_std = np.array([leakage_.std() for leakage_ in leakage_monthly_list])
    inconsistency_monthly_std = np.array([inconsistency_.std() for inconsistency_ in inconsistency_monthly_list])

    res_monthly_range = np.hstack([basin_index.reshape(-1, 1), res_monthly_mean, res_monthly_mean + res_monthly_std, res_monthly_mean - res_monthly_std])
    leakage_monthly_range = np.hstack([basin_index.reshape(-1, 1), leakage_monthly_mean, leakage_monthly_mean + leakage_monthly_std, leakage_monthly_mean - leakage_monthly_std])
    inconsistency_monthly_range = np.hstack([basin_index.reshape(-1, 1), inconsistency_monthly_mean, inconsistency_monthly_mean + inconsistency_monthly_std, inconsistency_monthly_mean - inconsistency_monthly_std])
    
    res_monthly_range_sorted = res_monthly_range[np.lexsort(res_monthly_range[:, ::-1].T)]
    leakage_monthly_range_sorted = leakage_monthly_range[np.lexsort(leakage_monthly_range[:, ::-1].T)]
    inconsistency_monthly_range_sorted = inconsistency_monthly_range[np.lexsort(inconsistency_monthly_range[:, ::-1].T)]
    
    # plot
    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=1, dpi=300, sharex=True)

        axes[0].plot(res_monthly_range_sorted[:, 0], res_monthly_range_sorted[:, 1],
                    "k--",
                    linewidth=0.5)
        axes[0].fill_between(x=res_monthly_range_sorted[:, 0],
                            y1=res_monthly_range_sorted[:, 2], y2=res_monthly_range_sorted[:, 3],
                            color="#C82423",
                            alpha=0.5,
                            linestyle="-",
                            linewidth=0.05)
        
        axes[1].plot(inconsistency_monthly_range_sorted[:, 0], inconsistency_monthly_range_sorted[:, 1],
                    "k--",
                    linewidth=0.5)
        axes[1].fill_between(x=inconsistency_monthly_range_sorted[:, 0],
                            y1=inconsistency_monthly_range_sorted[:, 2], y2=inconsistency_monthly_range_sorted[:, 3],
                            color="#2878B5",
                            alpha=0.5,
                            linestyle="-",
                            linewidth=0.05)
        
        axes[2].plot(leakage_monthly_range_sorted[:, 0], leakage_monthly_range_sorted[:, 1],
                    "k--",
                    linewidth=0.5)
        axes[2].fill_between(x=leakage_monthly_range_sorted[:, 0],
                            y1=leakage_monthly_range_sorted[:, 2], y2=leakage_monthly_range_sorted[:, 3],
                            color="#9AC9DB",
                            alpha=0.5,
                            linestyle="-",
                            linewidth=0.05)
        
        axes[0].set_ylim([-300, 100])
        axes[1].set_ylim([-300, 100])
        axes[2].set_ylim([-300, 100])
        axes[2].set_xlim([0, len(res_monthly_list) + 1])
        for ax in axes:
            ax.tick_params(labelsize=8, labelfontfamily="Arial", length=2)
            
        plt.show()
        fig.savefig("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/decomposition_basins_for_select.tiff")
        
    else:
        fig = None
        
    # sorted: max abs, then, max range(std)
    sorted_func_res_max_abs_range = lambda element: (abs(element["res_monthly_mean"].values[0]), abs(element["res_monthly_std"].values[0]))
    sorted_func_leakage_max_abs_range = lambda element: (abs(element["leakage_monthly_mean"].values[0]), abs(element["leakage_monthly_std"].values[0]))
    sorted_func_inconsistency_max_abs_range = lambda element: (abs(element["inconsistency_monthly_mean"].values[0]), abs(element["inconsistency_monthly_std"].values[0]))
    
    combined_list = []
    for i in range(len(basin_index)):
        basin_index_ = basin_index[i]
        combined_list.append({"basin_index": basin_index_,
                               "res_monthly_mean": decomposition_dict[basin_index_]["res"]["monthly"].mean(),
                               "res_monthly_std": decomposition_dict[basin_index_]["res"]["monthly"].std(),
                               "leakage_monthly_mean": decomposition_dict[basin_index_]["leakage"]["monthly"].mean(),
                               "leakage_monthly_std": decomposition_dict[basin_index_]["leakage"]["monthly"].std(),
                               "inconsistency_monthly_mean": decomposition_dict[basin_index_]["inconsistency"]["monthly"].mean(),
                               "inconsistency_monthly_std": decomposition_dict[basin_index_]["inconsistency"]["monthly"].std(),
                               }
                             )
    
    sorted_combined_list_based_on_res = sorted(combined_list, key=sorted_func_res_max_abs_range, reverse=True)
    sorted_combined_list_based_on_leakage = sorted(combined_list, key=sorted_func_leakage_max_abs_range, reverse=True)
    sorted_combined_list_based_on_inconsistency = sorted(combined_list, key=sorted_func_inconsistency_max_abs_range, reverse=True)
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_res.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - res_monthly_mean: {e['res_monthly_mean'].values[0]} - res_monthly_std: {e['res_monthly_std'].values[0]}", file=f) for e in sorted_combined_list_based_on_res]
        
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_leakage.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - leakage_monthly_mean: {e['leakage_monthly_mean'].values[0]} - leakage_monthly_std: {e['leakage_monthly_std'].values[0]}", file=f) for e in sorted_combined_list_based_on_leakage]
        
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_inconsistency.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - inconsistency_monthly_mean: {e['inconsistency_monthly_mean'].values[0]} - inconsistency_monthly_std: {e['inconsistency_monthly_std'].values[0]}", file=f) for e in sorted_combined_list_based_on_inconsistency]
    
    return decomposition_dict, fig


def select_basins_to_correct_make_sure(selected_basin_index, 
                                       selected_decomposition_fnames,
                                       selected_decomposition_fpaths):
    for i in range(len(selected_basin_index)):
        basin_index = selected_basin_index[i]
        selected_decomposition_fname = selected_decomposition_fnames[i]
        selected_decomposition_fpath = selected_decomposition_fpaths[i]
        with open(selected_decomposition_fpath, "rb") as f:
            initial_decomposition = pickle.load(f)
        print(f"basin_index {basin_index}", initial_decomposition["calibration_res"]["best_fitness"])
        print(f"basin_index {basin_index}", initial_decomposition["calibration_res"]["sim"])
    

def select_basins_to_correct_plot_map(basin_index, dpc_base, save=True):
    from decomposition_residuals import plot_base_map
    
    # read
    basinLatCen = np.array([dpc_base.basin_shp.loc[key, "lat_cen"] for key in basin_index])
    basinLonCen = np.array([dpc_base.basin_shp.loc[key, "lon_cen"] for key in basin_index])
    
    # plot
    fig, ax = plot_base_map()
    ax.scatter(basinLonCen, basinLatCen, marker=None, edgecolor="k", linewidths=0.1, color="r", s=200)
    
    if save:
        fig.savefig("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/decomposition_basins_for_select_map.tiff")
    

def correct_basin(forcing_fname, forcing_fpath, decomposition_fname, decomposition_fpath):
    # read force
    forcing = pd.read_csv(forcing_fpath, index_col=0)
    
    # read decomposition
    with open(decomposition_fpath, "rb") as f:
        initial_decomposition = pickle.load(f)
    
    # loop for correct
    correct_dict = loop_correct(forcing, initial_decomposition, correct_step=0.1)
    
    # save
    save_home = "F:/research/WaterBudgetClosure/Correct/Basins"
    for key in correct_dict.keys():
        with open(os.path.join(save_home, f"{forcing_fname[: forcing_fname.find('.')]}_{key}.pkl"), "wb") as f:
            pickle.dump(correct_dict[key], f)
    
    
def main():
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # loop for basins to build model
    for i in tqdm(range(len(fnames)), desc="loop for basins to build model", colour="green"):

        # read
        fpath = fpaths[i]
        fname = fnames[i]
        
        forcing = pd.read_csv(fpath, index_col=0)
        
        # loop for correct
        correct_dict = loop_correct(forcing, correct_step=0.1)
        
        # save
        save_home = "F:/research/WaterBudgetClosure/Correct/Basins"
        for key in correct_dict.keys():
            with open(os.path.join(save_home, f"{fname[: fname.find('.')]}_{key}.pkl"), "wb") as f:
                pickle.dump(correct_dict[key], f)


if __name__ == "__main__":
    # 1998-2010
    analysis_period = ["19980101", "20101231"]
    root, home = setHomePath(root="E:")
    subdir = "WaterBalanceAnalysis"
    
    # forcing path
    home_forcing = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing"
    forcing_fnames = [n for n in os.listdir(home_forcing) if n.endswith(".csv")]
    forcing_fpaths = [os.path.join(home_forcing, n) for n in forcing_fnames]
    
    # decomposition path
    home_decomposition = "F:/research/WaterBudgetClosure/Decomposition/Basins"
    decomposition_fnames = [n for n in os.listdir(home_decomposition) if n.endswith(".pkl")]
    decomposition_fpaths = [os.path.join(home_decomposition, n) for n in decomposition_fnames]
    
    # instance dpc_base
    with open(os.path.join("E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis", "dpc_base.pkl"), "rb") as f:
        dpc_base = pickle.load(f)

    # Decomposition_residual_basins
    # Decomposition_residual_basins()
    
    # select basins to correct, this use mean, however, below correct use mean(abs)
    # select_basins_to_correct()
    
    # read
    selected_basin_index = [0] #  [0, 36, 6]
    selected_forcing_fnames = [fn for fn in forcing_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
    selected_forcing_fpaths = [os.path.join(home_forcing, n) for n in selected_forcing_fnames]
    selected_decomposition_fnames = [fn for fn in decomposition_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
    selected_decomposition_fpaths = [os.path.join(home_decomposition, n) for n in selected_decomposition_fnames]
    selected_basin_index = [int(fn[: fn.find('_')]) for fn in selected_forcing_fnames]
    
    # get performance
    # select_basins_to_correct_make_sure(selected_basin_index,
    #                                    selected_decomposition_fnames,
    #                                    selected_decomposition_fpaths)
    
    # # plot map
    # select_basins_to_correct_plot_map(selected_basin_index, dpc_base, save=False)
    
    # correct
    correct_bool = True
    if correct_bool:
        for i in range(len(selected_basin_index)):
            forcing_fname = selected_forcing_fnames[i]
            forcing_fpath = selected_forcing_fpaths[i]
            decomposition_fname = selected_decomposition_fnames[i]
            decomposition_fpath = selected_decomposition_fpaths[i]
            
            correct_basin(forcing_fname, forcing_fpath, decomposition_fname, decomposition_fpath)
    
    # plot correct result
    # correct_dict_list = []
    # for i in range(len(selected_basin_index)):
    #     forcing_fname = selected_forcing_fnames[i]
    #     correct_dict = read_correct_result(forcing_fname[:forcing_fname.find(".csv")])
    #     correct_dict_list.append(correct_dict)
    
    
    # main: correct all basins
    # main()
    
    # read and plot
    # basin_name="106_forcing_basin_2055100"
    # correct_dict = read_correct_result(basin_name)
    # save_prefix = os.path.join("F:/research/WaterBudgetClosure/Correct/Basins", basin_name)
    # plot_correct_dict(correct_dict, save_prefix=save_prefix)
    