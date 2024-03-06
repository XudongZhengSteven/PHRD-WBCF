# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from WaterBudgetClosure.dataPreprocess_CAMELS_functions import setHomePath
import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
from WaterBudgetClosure.lumod_hbv.BuildModel import Lumod_HBV
from WaterBudgetClosure.WaterBalanceAnalysis import calResidual, WaterBalanceAnalysis_correct
from WaterBudgetClosure.lumod_hbv.CalibrateModel import calibration_basin, cal_fitness
import matplotlib.pyplot as plt
from Static_func.correlation_analysis import T_test
from pyecharts import options as opts
from pyecharts.charts import Sankey
plt.rcParams['font.family']='Arial'
plt.rcParams["font.weight"] = "bold"


def read_correct_dict(correct_dict_fpath):
    with open(correct_dict_fpath, "rb") as f:
        correct_dict = pickle.load(f)

    return correct_dict  


def plotPopfront(correct_dict):
    gridPop_correct_list = correct_dict["gridPop_correct_list"]
    front_correct_list = correct_dict["front_correct_list"]
    
    fitness_gridPop_correct_list = [[x.fitness.values for x in g] for g in gridPop_correct_list]
    fitness_front_correct_list = [[x.fitness.values for x in f] for f in  front_correct_list]
    
    correct_start_Pop = fitness_gridPop_correct_list[0]
    correct_start_front =fitness_front_correct_list[0]
    correct_end_Pop = fitness_gridPop_correct_list[-1]
    correct_end_front = fitness_front_correct_list[-1]
    
    names = ["qt", "sm", "grs", "E", "swe"]
    
    f, axes = plt.subplots(nrows=len(names), ncols=len(names),
                           sharex=True, sharey=True,
                           gridspec_kw={"wspace": 0.3, "hspace": 0.3,
                                        "left":0.1, "right": 0.95,
                                        "bottom": 0.1, "top": 0.95})
    
    for i in range(len(names)):
        name_x = names[i]
        for j in range(len(names)):
            name_y = names[j]
            
            ax = axes[i, j]
            
            # plot start
            correct_start_Pop_y = [g[i] for g in correct_start_Pop]
            correct_start_Pop_x = [g[j] for g in correct_start_Pop]
            correct_start_front_y = [f[i] for f in correct_start_front]
            correct_start_front_x = [f[j] for f in correct_start_front]
            
            ax.plot(correct_start_Pop_x, correct_start_Pop_y, '.', color="lightgrey", ms=3, zorder=5, alpha=0.3)
            ax.plot(correct_start_front_x, correct_start_front_y, 'b.', zorder=10, ms=4)

            # plot end
            correct_end_Pop_y = [g[i] for g in correct_end_Pop]
            correct_end_Pop_x = [g[j] for g in correct_end_Pop]
            correct_end_front_y = [f[i] for f in correct_end_front]
            correct_end_front_x = [f[j] for f in correct_end_front]
            
            ax.plot(correct_end_Pop_x, correct_end_Pop_y, '.', color="grey", ms=3, zorder=5, alpha=0.3)
            ax.plot(correct_end_front_x, correct_end_front_y, 'r.', zorder=10, ms=4)
            
            # set
            if i == len(names) - 1:
                ax.set_xlabel(names[j], fontdict={'family':'Arial', 'weight':'bold'})
            if j == 0:
                ax.set_ylabel(names[i], fontdict={'family':'Arial', 'weight':'bold'})
                
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    return f


def plotErrors(correct_dict):
    errors_correct_list = correct_dict["errors_correct_list"]
    res = [e["res"] for e in errors_correct_list]
    leak = [e["leak"] for e in errors_correct_list]
    inconsistency = [e["inconsistency"] for e in errors_correct_list]
    errors_correct_input_list = correct_dict["errors_correct_input_list"]
    errors_correct_output_list = correct_dict["errors_correct_output_list"]
    
    def plotErrors_timeseries(timescale, res, leak, inconsistency):
        f, axes = plt.subplots(nrows=3, ncols=1,
                               gridspec_kw={"wspace": 0.3, "hspace": 0.3,
                                            "left":0.1, "right": 0.95,
                                            "bottom": 0.1, "top": 0.95})
        for i in range(len(res)):
            if i == 0:
                alpha = 1
                color = "b"
                zorder = 10
            elif i == len(res) - 1:
                alpha = 1
                color = "r"
                zorder = 10
            else:
                alpha = 0.3
                color = "grey"
                zorder = 5
                
            axes[0].plot(res[i]["res_all_correct"][timescale], "-", color=color, alpha=alpha, zorder=zorder)
            axes[1].plot(leak[i]["leak_all_correct"][timescale], "-", color=color, alpha=alpha, zorder=zorder)
            axes[2].plot(inconsistency[i]["inconsistency_all_correct"][timescale], "-", color=color, alpha=alpha, zorder=zorder)
            
            [ax.set_xlim([res[i]["res_all_correct"][timescale].index[0], res[i]["res_all_correct"][timescale].index[-1]]) for ax in axes]
            
        return f
    
    def plotErrors_bar(timescale, res, leak, inconsistency):
        f_dict = dict()
        
        f_res, ax_res = plt.subplots()
        ax_res.bar(range(len(res)), [res[i]["res_abs_mean_all_correct"][timescale] for i in range(len(res))])
        ax_res.tick_params(labelsize=20)
        ax_res.set_xticks(range(len(res)))
        f_dict["f_res"] = f_res
        
        f_leak, ax_leak = plt.subplots()
        ax_leak.bar(range(len(leak)), [leak[i]["leak_abs_mean_all_correct"][timescale] for i in range(len(leak))])
        ax_leak.plot(range(len(leak)-1), [errors_correct_input_list[i]["leak"]["leak_abs_mean_all_correct"][timescale] for i in range(1, len(leak))],
                     "r", marker="o", linestyle="", ms=15, mfc="none", markeredgewidth=3)
        ax_leak.tick_params(labelsize=20)
        ax_leak.set_xticks(range(len(res)))
        f_dict["f_leak"] = f_leak
        
        f_inconsistency, ax_inconsistency = plt.subplots()
        ax_inconsistency.bar(range(len(inconsistency)), [inconsistency[i]["inconsistency_abs_mean_all_correct"][timescale] for i in range(len(inconsistency))])
        ax_inconsistency.tick_params(labelsize=20)
        ax_inconsistency.set_xticks(range(len(res)))
        f_dict["f_inconsistency"] = f_inconsistency
        
        return f_dict
    
    f_timeseries = dict()
    f_timeseries["daily"] = plotErrors_timeseries("daily", res, leak, inconsistency)
    f_timeseries["monthly"] = plotErrors_timeseries("monthly", res, leak, inconsistency)
    f_timeseries["yearly"] = plotErrors_timeseries("yearly", res, leak, inconsistency)
    
    f_bars_dict_dict = dict()
    f_bars_dict_dict["daily"] = plotErrors_bar("daily", res, leak, inconsistency)
    f_bars_dict_dict["monthly"] = plotErrors_bar("monthly", res, leak, inconsistency)
    f_bars_dict_dict["yearly"] = plotErrors_bar("yearly", res, leak, inconsistency)
    
    return f_timeseries, f_bars_dict_dict


def plotCorrectedforcings(correct_dict):
    forcing_correct_list = correct_dict["forcing_correct_list"]
    
    names_plot = ["prec", "qt", "E", "sm", "grs", "swe"]
    names_stand = ["prec", "qt", "E", "sm", "groundwater_reservoir", "swe"]
    names_transfer = dict(zip(names_plot, names_stand))
    
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                           gridspec_kw={"wspace": 0.1, "hspace": 0.2,
                                        "left":0.03, "right": 0.99,
                                        "bottom": 0.05, "top": 0.99})
    axes = axes.flatten()
    
    for i in range(len(names_plot)):
        name_plot = names_plot[i]
        name_stand = names_transfer[name_plot]
        ax = axes[i]
        
        for j in range(len(forcing_correct_list)):
            forcing_correct = forcing_correct_list[j]
            
            if j == 0:
                alpha = 1
                color = "b"
                zorder = 10
            elif j == len(forcing_correct_list) - 1:
                alpha = 1
                color = "r"
                zorder = 10
            else:
                alpha = 0.3
                color = "grey"
                zorder = 5

            ax.plot(forcing_correct[name_stand], "-", color=color, alpha=alpha, zorder=zorder)
            
        ax.set_xlim([forcing_correct.index[0], forcing_correct.index[-1]])
        # ax.set_ylabel(name_plot)
    
    return f


def plotCorrectedsims(correct_dict):
    sim_correct_list = correct_dict["sim_correct_list"]
    
    names_plot = ["baseflow", "qt", "et", "sm", "grs", "snow"]
    names_stand = ["baseflow", "qt", "et", "sm", "groundwater_reservoir", "snow"]
    names_transfer = dict(zip(names_plot, names_stand))
    
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                           gridspec_kw={"wspace": 0.1, "hspace": 0.2,
                                        "left":0.03, "right": 0.99,
                                        "bottom": 0.05, "top": 0.99})
    axes = axes.flatten()
    
    for i in range(len(names_plot)):
        name_plot = names_plot[i]
        name_stand = names_transfer[name_plot]
        ax = axes[i]
        
        for j in range(len(sim_correct_list)):
            forcing_correct = sim_correct_list[j]
            
            if j == 0:
                alpha = 1
                color = "#2878B5"
                zorder = 10
            elif j == len(sim_correct_list) - 1:
                alpha = 1
                color = "#C82423"
                zorder = 10
            else:
                alpha = 0.3
                color = "grey"
                zorder = 5

            ax.plot(forcing_correct[name_stand], "-", color=color, alpha=alpha, zorder=zorder)
            
        ax.set_xlim([forcing_correct.index[0], forcing_correct.index[-1]])
    
    return f
    
    
def plot_correct_dict(correct_dict):
    f_Popfront = plotPopfront(correct_dict)
    f_Errors_timeseries, f_Errors_bars_dict_dict = plotErrors(correct_dict)
    f_Correctedforcings = plotCorrectedforcings(correct_dict)
    f_Correctedsims = plotCorrectedsims(correct_dict)
    return f_Popfront, f_Errors_timeseries, f_Errors_bars_dict_dict, f_Correctedforcings, f_Correctedsims


def statistics_correct_dict(correct_dict):
    errors_correct_list = correct_dict["errors_correct_list"]
    best_cal_fitness_correct_list = correct_dict["best_cal_fitness_correct_list"]
    
    timescales = ["daily", "monthly", "yearly"]
    
    best_cal_fitness_change_ratio_dict = dict()
    best_cal_fitness_start_dict = dict()
    best_cal_fitness_end_dict = dict()
    statistics_errors = dict()
    res_change_ratio_dict = dict()
    res_start_dict = dict()
    inconsistency_start_dict = dict()
    res_end_dict = dict()
    inconsistency_end_dict = dict()
    inconsistency_change_ratio_dict = dict()
    
    for timescale in timescales:
        res_start = errors_correct_list[0]["res"]["res_abs_mean_all_correct"][timescale]
        res_end = errors_correct_list[-1]["res"]["res_abs_mean_all_correct"][timescale]
        res_change_ratio = (res_end - res_start) / res_start
        
        inconsistency_start = errors_correct_list[0]["inconsistency"]["inconsistency_abs_mean_all_correct"][timescale]
        inconsistency_end = errors_correct_list[-1]["inconsistency"]["inconsistency_abs_mean_all_correct"][timescale]
        inconsistency_change_ratio = (inconsistency_end - inconsistency_start) / inconsistency_start
        
        res_change_ratio_dict[timescale] = res_change_ratio
        res_start_dict[timescale] = res_start
        res_end_dict[timescale] = res_end
        
        inconsistency_change_ratio_dict[timescale] = inconsistency_change_ratio
        inconsistency_start_dict[timescale] = inconsistency_start
        inconsistency_end_dict[timescale] = inconsistency_end
    
    statistics_errors["res_change_ratio_dict"] = res_change_ratio_dict
    statistics_errors["inconsistency_change_ratio_dict"] = inconsistency_change_ratio_dict
    statistics_errors["res_start_dict"] = res_start_dict
    statistics_errors["res_end_dict"] = res_end_dict
    statistics_errors["inconsistency_start_dict"] = inconsistency_start_dict
    statistics_errors["inconsistency_end_dict"] = inconsistency_end_dict
    
    best_cal_fitness_start = np.array(best_cal_fitness_correct_list[0])
    best_cal_fitness_end = np.array(best_cal_fitness_correct_list[-1])
    best_cal_fitness_change_ratio_dict["best_cal_fitness_change_ratio"] = (best_cal_fitness_end - best_cal_fitness_start) / best_cal_fitness_start
    best_cal_fitness_start_dict["best_cal_fitness_start"] = best_cal_fitness_start
    best_cal_fitness_end_dict["best_cal_fitness_end"] = best_cal_fitness_end
    
    return statistics_errors, best_cal_fitness_change_ratio_dict, best_cal_fitness_start_dict, best_cal_fitness_end_dict
    
    
def valid_performance(fitness):
    metrics_qt, metrics_sm, metrics_groundwater_reservoir, metrics_E, metrics_swe = fitness
    N = 4748
    valid_qt = metrics_qt > -0.41
    valid_sm = T_test(metrics_sm, N, two_side=False, right=True, alpha=0.05)[0]
    valid_grs = T_test(metrics_groundwater_reservoir, N, two_side=False, right=True, alpha=0.05)[0]
    valid_E = metrics_E > -0.41
    valid_swe = metrics_swe > -0.41
    
    valid = (valid_qt, valid_sm, valid_grs, valid_E, valid_swe)
    valid_first_four_variable = all((valid_qt, valid_sm, valid_grs, valid_E))
    
    return valid, valid_first_four_variable, valid_swe


def preprocess_forcing(forcing):
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
    
    return forcing, basinArea, basinLatCen


def warmup_Model(model, forcing, start="19980101", end="19991231"):
    print("model warmup".center(50, "-"))
    for i in range(5):
        model.warm_up(forcing, start=start, end=end)  # ["19980101", "20101231"]
        print(f'{model.model.params["s0"]}, {model.model.params["w01"]}, {model.model.params["w02"]}')
    return model


def cal_inconsistency(res_all_correct, leak_all_correct):
    inconsistency_all_correct = {}
    inconsistency_all_correct["daily"] = res_all_correct["daily"] - leak_all_correct["daily"]
    inconsistency_all_correct["monthly"] = res_all_correct["monthly"] - leak_all_correct["monthly"]
    inconsistency_all_correct["yearly"] = res_all_correct["yearly"] - leak_all_correct["yearly"]
    
    inconsistency_abs_mean_all_correct = {"daily": np.mean(abs(inconsistency_all_correct["daily"])),
                                                "monthly": np.mean(abs(inconsistency_all_correct["monthly"])),
                                                "yearly": np.mean(abs(inconsistency_all_correct["yearly"]))}
    
    return inconsistency_all_correct, inconsistency_abs_mean_all_correct

def correct_model_input(forcing, sim, correct_step=0.1):
    # res of sim: leak error
    # res of obs: leak error + inconsistency error
    # use pre to reduce leak error, namely, reduce res of sim
    # note: the signal of leak error is not must be same as inconsistency error
    # copy
    forcing = deepcopy(forcing)
    forcing_corrected = deepcopy(forcing)
    
    # init wbac
    wbac = WaterBalanceAnalysis_correct()
    
    # sim res (leak error)
    factor_feet2meter = 0.0283168
    forcing_sim = deepcopy(forcing)
    forcing_sim.swe = sim.snow
    forcing_sim.qt = sim.qt
    forcing_sim.streamflow = sim.qt / factor_feet2meter
    forcing_sim.E = sim.et
    forcing_sim.sm = sim.sm
    forcing_sim.groundwater_reservoir = sim.groundwater_reservoir
    forcing_sim.all_sm = sim.all_sm
    
    forcing_sim.sm = forcing_sim.sm - (forcing_sim.sm.mean() - forcing.sm.mean())
    forcing_sim.groundwater_reservoir = forcing_sim.groundwater_reservoir - (forcing_sim.groundwater_reservoir.mean() - forcing.groundwater_reservoir.mean())
    forcing_sim.all_sm = forcing_sim.all_sm - (forcing_sim.all_sm.mean() - forcing.all_sm.mean())
    
    leak_all_sim, leak_quantile_all_sim, leak_std_all_sim, leak_abs_mean_all_sim = calResidual(forcing_sim)
    
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
    
    forcing_sim.prec[forcing_sim.prec < 0] = 0
    
    # save the corrected values
    forcing_corrected.prec = forcing_sim.prec
    
    # cal errors after corrected input/pre (others keep same with sim)
    
    # leak
    leak_all_correct, leak_quantile_all_correct, leak_std_all_correct, leak_abs_mean_all_correct = calResidual(forcing_sim)
    
    # res
    res_all_correct, res_quantile_all_correct, res_std_all_correct, res_abs_mean_all_correct = calResidual(forcing_corrected)
    
    # inconsistency
    inconsistency_all_correct, inconsistency_abs_mean_all_correct = cal_inconsistency(res_all_correct, leak_all_correct)
    
    # errors_correct dict
    errors_correct = {"res": {"res_all_correct": res_all_correct, "res_abs_mean_all_correct": res_abs_mean_all_correct},
                      "leak": {"leak_all_correct":leak_all_correct, "leak_abs_mean_all_correct": leak_abs_mean_all_correct},
                      "inconsistency": {"inconsistency_all_correct": inconsistency_all_correct, "inconsistency_abs_mean_all_correct": inconsistency_abs_mean_all_correct}}
    
    return forcing_corrected, errors_correct
 

def correct_model_output(forcing, sim, best_cal_fitness, correct_step=0.1):
    # *do not correct qt/streamflow (see it as truth)
    # TODO  使用相关性类型的校准, copula?
    valid, _, _ = valid_performance(best_cal_fitness)
    
    # copy
    forcing = deepcopy(forcing)
    forcing_corrected = deepcopy(forcing)
    
    # sim res (leakage error)
    factor_feet2meter = 0.0283168
    forcing_sim = deepcopy(forcing)
    forcing_sim.swe = sim.snow
    forcing_sim.qt = sim.qt
    forcing_sim.streamflow = sim.qt / factor_feet2meter
    forcing_sim.E = sim.et
    forcing_sim.sm = sim.sm
    forcing_sim.groundwater_reservoir = sim.groundwater_reservoir
    forcing_sim.all_sm = sim.all_sm
    
    forcing_sim.sm = forcing_sim.sm - (forcing_sim.sm.mean() - forcing.sm.mean())
    forcing_sim.groundwater_reservoir = forcing_sim.groundwater_reservoir - (forcing_sim.groundwater_reservoir.mean() - forcing.groundwater_reservoir.mean())
    forcing_sim.all_sm = forcing_sim.all_sm - (forcing_sim.all_sm.mean() - forcing.all_sm.mean())
    
    # correct for output: et, swe
    if valid[3]:
        forcing_corrected.E = forcing_corrected.E + (forcing_sim.E - forcing_corrected.E) * correct_step
    if valid[4]:
        forcing_corrected.swe = forcing_corrected.swe + (forcing_sim.swe - forcing_corrected.swe) * correct_step
    
    # correct for output: sm and groundwater_reservoir
    if valid[1]:
        forcing_corrected.sm = forcing_corrected.sm + (forcing_sim.sm - forcing_corrected.sm) * correct_step
    if valid[2]:
        forcing_corrected.groundwater_reservoir = forcing_corrected.groundwater_reservoir + (forcing_sim.groundwater_reservoir - forcing_corrected.groundwater_reservoir) * correct_step
    
        # stand_sim_sm = (sim.sm - sim.sm.mean()) / sim.sm.std()
        # stand_corrected_sm = (forcing_corrected.sm - forcing_corrected.sm.mean()) / forcing_corrected.sm.std()
        # forcing_corrected.sm = (stand_corrected_sm + (stand_sim_sm - stand_corrected_sm) * correct_step) * forcing_corrected.sm.std() + forcing_corrected.sm.mean()
        
        # stand_sim_groundwater_reservoir = (sim.groundwater_reservoir - sim.groundwater_reservoir.mean()) / sim.groundwater_reservoir.std()
        # stand_corrected_groundwater_reservoir = (forcing_corrected.groundwater_reservoir - forcing_corrected.groundwater_reservoir.mean()) / forcing_corrected.groundwater_reservoir.std()
        # forcing_corrected.groundwater_reservoir = (stand_corrected_groundwater_reservoir + (stand_sim_groundwater_reservoir - stand_corrected_groundwater_reservoir) * correct_step) * forcing_corrected.groundwater_reservoir.std() + forcing_corrected.groundwater_reservoir.mean()
    
        # sim_diff_sm = sim.sm - sim.sm.mean()
        # forcing_corrected_diff_sm = forcing_corrected.sm - forcing_corrected.sm.mean()
        # forcing_corrected.sm = forcing_corrected.sm + (sim_diff_sm - forcing_corrected_diff_sm) * correct_step
        
        # sim_diff_groundwater_reservoir = sim.groundwater_reservoir - sim.groundwater_reservoir.mean()
        # forcing_corrected_diff_groundwater_reservoir = forcing_corrected.groundwater_reservoir - forcing_corrected.groundwater_reservoir.mean()
        # forcing_corrected.groundwater_reservoir = forcing_corrected.groundwater_reservoir + (sim_diff_groundwater_reservoir - forcing_corrected_diff_groundwater_reservoir) * correct_step
        
    # cal all_sm = sm + groundwater_reservoir
    forcing_corrected.all_sm = forcing_corrected.sm + forcing_corrected.groundwater_reservoir
    
    # correct for output: streamflow, qt
    if valid[0]:
        forcing_corrected.streamflow = forcing_corrected.streamflow + (forcing_sim.streamflow - forcing_corrected.streamflow) * correct_step
        forcing_corrected.qt = forcing_corrected.qt + (forcing_sim.qt - forcing_corrected.qt) * correct_step
    
    # cal errors after corrected output/E, SWE, sm, groundwater_reservoir
    
    # leak #* note the identification of leak error (model sim leak error)
    leak_all_correct, leak_quantile_all_correct, leak_std_all_correct, leak_abs_mean_all_correct = calResidual(forcing_sim)
    
    # res
    res_all_correct, res_quantile_all_correct, res_std_all_correct, res_abs_mean_all_correct = calResidual(forcing_corrected)
    
    # inconsistency
    inconsistency_all_correct, inconsistency_abs_mean_all_correct = cal_inconsistency(res_all_correct, leak_all_correct)
    
    # errors_correct dict
    errors_correct = {"res": {"res_all_correct": res_all_correct, "res_abs_mean_all_correct": res_abs_mean_all_correct},
                      "leak": {"leak_all_correct":leak_all_correct, "leak_abs_mean_all_correct": leak_abs_mean_all_correct},
                      "inconsistency": {"inconsistency_all_correct": inconsistency_all_correct, "inconsistency_abs_mean_all_correct": inconsistency_abs_mean_all_correct}}

    return forcing_corrected, errors_correct


def correct_citeria(errors_correct_list, correct_step):
    if len(errors_correct_list) > 1:
        correct_bool_res = False if errors_correct_list[-1]['res']['res_abs_mean_all_correct']["monthly"] <= errors_correct_list[0]['res']['res_abs_mean_all_correct']["monthly"] * 0.1 else True
        # correct_bool_leak = True if errors_correct_list[-1]['leak']['leak_abs_mean_all_correct']["monthly"] <= errors_correct_list[0]['leak']['leak_abs_mean_all_correct']["monthly"] * 0.5 else False
        correct_bool = correct_bool_res
    else:
        correct_bool = True
    
    correct_bool = False if correct_step <= 0.04 else correct_bool
    
    return correct_bool

def loop_correct(forcing, initial_decomposition, correct_step=0.5):
    # preprocess
    forcing, basinArea, basinLatCen = preprocess_forcing(forcing)
    
    # initial model and warm-up
    model = Lumod_HBV(area=basinArea, lat=basinLatCen)
    # model = warmup_Model(model, forcing, start="19980101", end="19991231")
    
    # list for saveing correct process and set the initial value
    forcing_correct_list = [forcing]
    best_params_dict_correct_list = [initial_decomposition["calibration_res"]["best_params_dict"]]
    best_fitness_correct_list = [initial_decomposition["calibration_res"]["best_fitness"]]
    best_cal_fitness_correct_list = [initial_decomposition["calibration_res"]["best_cal_fitness"]]
    weighted_best_fitness_correct_list = [initial_decomposition["calibration_res"]["weighted_best_fitness"]]
    sim_correct_list = [initial_decomposition["calibration_res"]["sim"]]
    front_correct_list = [initial_decomposition["calibration_res"]["front"]]
    gridPop_correct_list = [initial_decomposition["calibration_res"]["gridPop"]]
    errors_correct_input_list = [None]
    errors_correct_output_list = [None]
    
    initial_errors = initial_decomposition["errors"]
    initial_errors_keys = list(initial_errors.keys())
    initial_errors_m = dict()
    for key in initial_errors_keys:
        initial_errors_ = initial_errors[key]
        original_k = [k for k in initial_errors_.keys()]
        original_values = [initial_errors_[k] for k in initial_errors_.keys()]
        modified_k = [k+"_correct" for k in original_k]
        updated_dict = dict(zip(modified_k, original_values)) 
        initial_errors_m[key] = updated_dict
    
    errors_correct_list = [initial_errors_m]

    print("\n---- initial ----")
    print(f"initial best_cal_fitness: {best_cal_fitness_correct_list[0]}")
    print(f"initial res_abs_mean: {errors_correct_list[0]['res']['res_abs_mean_all_correct']}")
    print(f"initial leak_abs_mean: {errors_correct_list[0]['leak']['leak_abs_mean_all_correct']}")
    print(f"initial inconsistency_abs_mean: {errors_correct_list[0]['inconsistency']['inconsistency_abs_mean_all_correct']}")
    
    # loop correct
    iter_correct = 1
    while correct_citeria(errors_correct_list, correct_step):
        # print correct_step
        print(f"correct step: {correct_step}")
        
        # get last sim, forcing
        sim = sim_correct_list[-1]
        forcing_corrected = forcing_correct_list[-1]
        errors_correct = errors_correct_list[-1]
        best_cal_fitness = best_cal_fitness_correct_list[-1]
        print(sim)
        
        # performance needs to be good
        if valid_performance(best_cal_fitness)[1]:
        # if best_cal_fitness[0] > -0.41:
            
            # correct input
            print("\n----correct model input to reduce leak error----")
            forcing_corrected, errors_correct = correct_model_input(forcing_corrected, sim, correct_step)
            print("res", errors_correct["res"]["res_abs_mean_all_correct"])
            print("leak", errors_correct["leak"]["leak_abs_mean_all_correct"])
            print("inconsistency", errors_correct["inconsistency"]["inconsistency_abs_mean_all_correct"])
            errors_correct_input = errors_correct
            
            # correct output
            print("\n----correct model output to reduce inconsistency error----")
            forcing_corrected, errors_correct = correct_model_output(forcing_corrected, sim, best_cal_fitness, correct_step)
            print("res", errors_correct["res"]["res_abs_mean_all_correct"])
            print("leak", errors_correct["leak"]["leak_abs_mean_all_correct"])
            print("inconsistency", errors_correct["inconsistency"]["inconsistency_abs_mean_all_correct"])
            errors_correct_output = errors_correct

        
        # calibration again
        best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = calibration_basin(model, forcing_corrected, save=None)
        fc = best_params_dict["fc"]
        sim.loc[:, "sm"] = sim.ws * fc
        sim.loc[:, "groundwater_reservoir"] = sim.ws1 + sim.ws2
        sim.loc[:, "all_sm"] = sim.sm + sim.groundwater_reservoir
        
        # cal_fitness
        best_cal_fitness = cal_fitness(forcing_corrected, sim, best_params_dict)
        
        # update errors: leak and inconsistency
        factor_feet2meter = 0.0283168
        forcing_sim = deepcopy(forcing_corrected)
        forcing_sim.swe = sim.snow
        forcing_sim.qt = sim.qt
        forcing_sim.streamflow = sim.qt / factor_feet2meter
        forcing_sim.E = sim.et
        forcing_sim.sm = sim.sm
        forcing_sim.groundwater_reservoir = sim.groundwater_reservoir
        forcing_sim.all_sm = sim.all_sm
        
        forcing_sim.sm = forcing_sim.sm - (forcing_sim.sm.mean() - forcing.sm.mean())
        forcing_sim.groundwater_reservoir = forcing_sim.groundwater_reservoir - (forcing_sim.groundwater_reservoir.mean() - forcing.groundwater_reservoir.mean())
        forcing_sim.all_sm = forcing_sim.all_sm - (forcing_sim.all_sm.mean() - forcing_corrected.all_sm.mean())
        
        leak_all_correct, leak_quantile_all_correct, leak_std_all_correct, leak_abs_mean_all_correct = calResidual(forcing_sim)
        inconsistency_all_correct, inconsistency_abs_mean_all_correct = cal_inconsistency(errors_correct["res"]["res_all_correct"], leak_all_correct)
        
        errors_correct.update({"leak": {"leak_all_correct": leak_all_correct, "leak_abs_mean_all_correct": leak_abs_mean_all_correct}})
        errors_correct.update({"inconsistency": {"inconsistency_all_correct": inconsistency_all_correct, "inconsistency_abs_mean_all_correct": inconsistency_abs_mean_all_correct}})
        
        # print correct result
        print(f"\n---- iter_correct {iter_correct} ----")
        print(f"best_fitness: {best_fitness}")
        print(f"best_cal_fitness: {best_cal_fitness}")
        print(f"res: {errors_correct['res']['res_abs_mean_all_correct']}")
        print(f"leak: {errors_correct['leak']['leak_abs_mean_all_correct']}")
        print(f"inconsistency: {errors_correct['inconsistency']['inconsistency_abs_mean_all_correct']}")
        
        # save
        if valid_performance(best_cal_fitness)[1]:
        # if best_cal_fitness[0] > -0.41:
            forcing_correct_list.append(forcing_corrected)
            best_params_dict_correct_list.append(best_params_dict)
            best_fitness_correct_list.append(best_fitness)
            best_cal_fitness_correct_list.append(best_cal_fitness)
            weighted_best_fitness_correct_list.append(weighted_best_fitness)
            sim_correct_list.append(sim)
            front_correct_list.append(front)
            gridPop_correct_list.append(gridPop)
            errors_correct_list.append(errors_correct)
            errors_correct_input_list.append(errors_correct_input)
            errors_correct_output_list.append(errors_correct_output)
        
            iter_correct += 1
            correct_step *= 2 if correct_step < 0.5 else 0.5
            
        else:
            print("\n---- !performance not good, re-correct ----")
            correct_step /= 2

    
    
    # dict
    correct_dict = {"iter_correct": iter_correct,
                    "forcing_correct_list": forcing_correct_list,
                    "best_params_dict_correct_list": best_params_dict_correct_list,
                    "best_fitness_correct_list": best_fitness_correct_list,
                    "best_cal_fitness_correct_list": best_cal_fitness_correct_list,
                    "weighted_best_fitness_correct_list": weighted_best_fitness_correct_list,
                    "sim_correct_list": sim_correct_list,
                    "front_correct_list": front_correct_list,
                    "gridPop_correct_list": gridPop_correct_list,
                    "errors_correct_list": errors_correct_list,
                    "errors_correct_input_list": errors_correct_input_list,
                    "errors_correct_output_list": errors_correct_output_list
                    }
    
    return correct_dict


def correct_basin(forcing_fname, forcing_fpath, decomposition_fpath, save_home):
    # print
    print(f"correct basin: {forcing_fname}")
    
    # read force
    forcing = pd.read_csv(forcing_fpath, index_col=0)
    
    # read decomposition
    with open(decomposition_fpath, "rb") as f:
        initial_decomposition = pickle.load(f)
    
    # initial performance effective
    if valid_performance(initial_decomposition["calibration_res"]["best_cal_fitness"])[1]:
        
        # loop for correct
        correct_dict = loop_correct(forcing, initial_decomposition, correct_step=0.5)
        
        # save
        with open(os.path.join(save_home, f"{forcing_fname[: forcing_fname.find('.')]}.pkl"), "wb") as f:
            pickle.dump(correct_dict, f)
            
    else:
        ineffective_readme_path = os.path.join(save_home, "basins_ineffective_initial_performance.txt")
        with open(ineffective_readme_path, "a") as f:
            f.write(forcing_fname[: forcing_fname.rfind('.csv')])
            f.write(f': {"best_cal_fitness"}: {initial_decomposition["calibration_res"]["best_cal_fitness"]}\n')
            
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
    
    # correct save path
    correct_home = "F:/research/WaterBudgetClosure/Correct/Basins"
    
    # instance dpc_base
    with open(os.path.join("E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis", "dpc_base.pkl"), "rb") as f:
        dpc_base = pickle.load(f)

    # read
    selected_basin_index = None #[8, 171, 27, 218, 33, 21] # [0, 6, 588, 610, 609, 611, 5, 274, 7, 36, 626, 596, 251, 4, 30]
    if selected_basin_index:
        selected_forcing_fnames = [fn for fn in forcing_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
    else:
        selected_forcing_fnames = forcing_fnames
    
    # exclude
    exclude_existing = True
    if exclude_existing:
        existing_fnames = [n for n in os.listdir(correct_home) if n.endswith(".pkl")]
        existing_index = [int(fn[: fn.find('_')]) for fn in existing_fnames]
        selected_forcing_fnames = [fn for fn in selected_forcing_fnames if int(fn[: fn.find('_')]) not in existing_index]
    
    selected_forcing_fpaths = [os.path.join(home_forcing, n) for n in selected_forcing_fnames]
    selected_decomposition_fnames = [f"{forcing_fnames[: forcing_fnames.rfind('.csv')]}_decomposition.pkl" for forcing_fnames in selected_forcing_fnames]
    selected_decomposition_fpaths = [os.path.join(home_decomposition, fn) for fn in selected_decomposition_fnames]
    selected_basin_index = [int(fn[: fn.find('_')]) for fn in selected_forcing_fnames]
    
    # correct
    correct_bool = True
    if correct_bool:
        for i in range(len(selected_basin_index)):
            forcing_fname = selected_forcing_fnames[i]
            forcing_fpath = selected_forcing_fpaths[i]
            decomposition_fname = selected_decomposition_fnames[i]
            decomposition_fpath = selected_decomposition_fpaths[i]
            
            correct_basin(forcing_fname, forcing_fpath, decomposition_fpath, correct_home)

    # read_correct_dict_
    selected_basin_index = None # [0]
    read_correct_dict_ = False
    correct_plot_ = True
    save_correct_plot = True
    correct_statistics_ = False
    save_correct_statistics_ = True
    if read_correct_dict_:
        home_correct_plot = "F:/research/WaterBudgetClosure/Correct/correct_plot"
        home_correct_dict = "F:/research/WaterBudgetClosure/Correct/Basins"
        correct_dict_fnames = [n for n in os.listdir(home_correct_dict) if n.endswith(".pkl")]
        if selected_basin_index:
            correct_dict_fnames = [fn for fn in correct_dict_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
        
        correct_dict_fpaths = [os.path.join(home_correct_dict, n) for n in correct_dict_fnames]
        
        for i in range(len(correct_dict_fnames)):
            correct_dict_fname = correct_dict_fnames[i]
            correct_dict_fpath = correct_dict_fpaths[i]
            
            correct_dict = read_correct_dict(correct_dict_fpath)
            
            # plot
            if correct_plot_:
                f_Popfront, f_Errors_timeseries, f_Errors_bars_dict_dict, f_Correctedforcings, f_Correctedsims = plot_correct_dict(correct_dict)
                plt.show()
                
                # save
                if save_correct_plot:
                    dir_plot = os.path.join(home_correct_plot, correct_dict_fname[:correct_dict_fname.find(".")])
                    if not os.path.exists(dir_plot):
                        os.mkdir(dir_plot)
                    
                    f_Popfront.savefig(os.path.join(dir_plot, "f_Popfront.tiff"))
                    timescale = ["daily", "monthly", "yearly"]
                    errors_ = ["f_res", "f_leak", "f_inconsistency"]
                    [f_Errors_timeseries[t].savefig(os.path.join(dir_plot, f"f_Errors_timeseries_{t}.tiff")) for t in timescale]
                    [f_Errors_bars_dict_dict[t][e].savefig(os.path.join(dir_plot, f"f_Errors_bars_{e}_{t}.tiff")) for t in timescale for e in errors_]
                    f_Correctedforcings.savefig(os.path.join(dir_plot, "f_Correctedforcings.tiff"))
                    f_Correctedsims.savefig(os.path.join(dir_plot, "f_Correctedsims.tiff"))
            
            # statistics
            if correct_statistics_:
                statistics_errors, best_cal_fitness_change_ratio, best_cal_fitness_start_dict, best_cal_fitness_end_dict = statistics_correct_dict(correct_dict)
                
                # save
                if save_correct_statistics_:
                    dir_plot = os.path.join(home_correct_plot, correct_dict_fname[:correct_dict_fname.find(".")])
                    if not os.path.exists(dir_plot):
                        os.mkdir(dir_plot)

                    with open(os.path.join(dir_plot, f"correct_statistics.txt"), "w") as f:
                        print(statistics_errors, file=f)
                        print(best_cal_fitness_change_ratio, file=f)
                        print(best_cal_fitness_start_dict, file=f)
                        print(best_cal_fitness_end_dict, file=f)
        