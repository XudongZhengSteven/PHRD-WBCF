# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lumod
from lumod import tools
from lumod import MonteCarlo
from Calibration import Calibration_Deap
from copy import deepcopy
from lumod.tools import metrics


def calibration_mc(model, forcing):
    # monte carlo
    # Initial requirements
    nsims = 1000  # number of simulations
    keepb = 20    # keep only the best simulations
    savevars = ["qt"]  # save only this variables from Monte Carlo simulation , "snow", "ws1", "ws2"
    xobs = forcing.loc[:, ["qt"]]
    
    # Bounds of Parameters
    bounds = {
        "beta": (0.0, 8.0),      
        "fc": (30.0, 600.0),
        "maxbas": (1, 10),
        'k0': (0.1, 0.8),
        'k1': (0.01, 0.5),
        'k2': (0.001, 0.1),
        'kp': (0.001, 5.0),
        'pwp': (0.2, 1.0),
        'lthres': (10., 60.0),
        'tthres': (-2.0, 2.0),
    }

    scale = {"k1": "log", "k2": "log", "kp": "log"}
    
    # Objective function
    score1 = {
        "var": "qt",
        "metric": "kge",
        "weight": 1.0
    }

    # score2 = {
    #     "var": "snow",
    #     "metric": "kge",
    #     "agg": "mean",
    #     "weight": 0.25
    # }
    
    # score3 = {
    #     "var": "ws1",
    #     "metric": "kge",
    #     "agg": "mean",
    #     "weight": 0.25
    # }
    
    # score4 = {
    #     "var": "ws2",
    #     "metric": "kge",
    #     "agg": "mean",
        # "weight": 0.25
    # }
    
    scores = [score1] # score2, score3, score4


    # Run Monte Carlo
    mc_res = lumod.MonteCarlo(
        model,
        forcing,
        bounds,
        param_scale=scale,
        numsimul=nsims,
        save_vars=savevars,
        xobs=xobs,
        scores=scores,
        keep_best=keepb,
    )

    # Simulations analysis
    qtsim      = mc_res["simulations"]["qt"]
    qtsim_min  = qtsim.min(axis=1)
    qtsim_mean = qtsim.mean(axis=1)
    qtsim_max  = qtsim.max(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x=qtsim_min.index, y1=qtsim_min, y2=qtsim_max, color="b", alpha=0.6)
    ax.plot(xobs.index, xobs["qt"], color="k", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Streamflow (m3/s)", fontsize=12)
    ax.grid(True)
    ax.set_yscale("log")
    fig.tight_layout()
    plt.show()
    return mc_res


def calibration_deap_SGA(model, forcings, maxGen=250):
    bounds = {
        # snow
        "dd": (1.0, 10.0),
        'tthres': (-2.5, 2.5),
        
        # soil moisture
        "beta": (1.0, 8.0),
        "fc": (10.0, 600.0),
        
        # ET
        'pwp': (0.2, 1.0),
        
        # runoff generate
        'lthres': (10., 200.0),
        'k0': (0.1, 0.8),
        'k1': (0.01, 0.5),
        'k2': (0.001, 0.15),
        'kp': (0.001, 5.0),
        
        # runoff concentration
        "maxbas": (1, 10),
    }
    
    param_names = list(bounds.keys())
    param_types = [type(bounds[k][0]) for k in bounds.keys()]
    low = [bounds[k][0] for k in bounds.keys()]
    up = [bounds[k][1] for k in bounds.keys()]
    
    calibrate = Calibration_Deap.Calibration_SGA_oneobj(low=low, up=up, maxGen=maxGen)
    r, r_format_type, gen, fit_maxs, fit_average, fig_result = calibrate(model, forcings, param_names, param_types, cp_path="F:/research/WaterBudgetClosure/HBV/Calibration/cp.pkl")
    r_format_type = dict(zip(param_names, r_format_type))
    
    return r_format_type, gen, fit_maxs, fit_average, fig_result


def calibration_deap_NSGAII(model, forcings, weight, start=None, end=None, maxGen=250, save=None):
    bounds = {
        # snow
        "dd": (1.0, 10.0),
        'tthres': (-2.5, 2.5),
        
        # soil moisture
        "beta": (1.0, 8.0),
        "fc": (10.0, 600.0),
        
        # ET
        'pwp': (0.2, 1.0),
        # 'cevp': 
        # 'cevpam': 
        # 'cevpph': 
        
        # runoff generate
        'lthres': (10., 200.0),
        'k0': (0.1, 0.8),
        'k1': (0.01, 0.5),
        'k2': (0.001, 0.15),
        'kp': (0.001, 5.0),
        
        # runoff concentration 
        "maxbas": (1, 10),
    }
    
    param_names = list(bounds.keys())
    param_types = [type(bounds[k][0]) for k in bounds.keys()]
    low = [bounds[k][0] for k in bounds.keys()]
    up = [bounds[k][1] for k in bounds.keys()]
    
    calibrate = Calibration_Deap.Calibration_NSGAII_multi_obj(low=low, up=up, maxGen=maxGen, weight=weight)
    front, gridPop, fig_result = calibrate(model, forcings, param_names, param_types,
                                           xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
                                           start=start, end=end,
                                           plotting=False, plotting_iter=20, plot_result=False, plot_result_show=False,
                                           save=save, print_bool=False,
                                           load_checkpoint=False)
    
    return front, gridPop, fig_result, param_names, param_types


def select_front_EqualWeightedSum_fitness(front, threshold_perc=50, missing_value=-9999):
    # remove missing_value
    front_no_missing = [ind for ind in front if missing_value not in ind.fitness.values]
    front = front if len(front_no_missing) == 0 else front_no_missing
    
    # fitness
    fitness = [ind.fitness.values for ind in front]
    
    # remove ind from front based on threshold
    for i in range(len(fitness[0])):
        threshold = np.percentile([fn[i] for fn in fitness], threshold_perc)
        try:
            front_threshold = [ind for ind in front if ind.fitness.values[i] >= threshold]
        except:
            raise ValueError("cannot remove ind based on threshold1 and threshold2")
    
    # sort front based on sorted_function
    weight = np.full((len(fitness[0]), ), fill_value=1 / len(fitness[0])) 
    sorted_function = lambda ind: sum(ind.fitness.values * weight)
    sorted_front = sorted(front_threshold, key=sorted_function, reverse=True)
    sorted_fitness = [sorted_function(ind) for ind in sorted_front]
    print(f"sorted_fitness: {sorted_fitness}")
    
    return sorted_front[0], sorted_fitness[0]


def select_front_weightedSum_fitness(front, threshold1=None, threshold2=None, weight1=0.5, weight2=0.5):
    fitness = [ind.fitness.values for ind in front]
    
    # remove ind from front based on threshold1 and threshold2
    threshold1 = threshold1 if threshold1 else np.mean([fn[0] for fn in fitness])
    threshold2 = threshold2 if threshold2 else np.mean([fn[1] for fn in fitness])
    try:
        front = [ind for ind in front if ind.fitness.values[0] >= threshold1 and ind.fitness.values[1] >= threshold2]
    except:
        raise ValueError("cannot remove ind based on threshold1 and threshold2")
    
    # sort front based on sorted_function
    sorted_function = lambda ind: ind.fitness.values[0] * weight1 + ind.fitness.values[1] * weight2
    sorted_front = sorted(front, key=sorted_function, reverse=True)
    sorted_fitness = [sorted_function(ind) for ind in sorted_front]
    print(f"sorted_fitness: {sorted_fitness}")
    
    return sorted_front[0], sorted_fitness[0]


def select_front_weighted_fitness(front, threshold_perc=50, weight=[0.4, 0.2, 0.2, 0.1, 0.1], missing_value=-9999):
    # remove missing_value
    front_no_missing = [ind for ind in front if missing_value not in ind.fitness.values]
    front = front if len(front_no_missing) == 0 else front_no_missing
    
    # fitness
    fitness = [ind.fitness.values for ind in front]
    
    # remove ind from front based on threshold
    for i in range(len(fitness[0])):
        threshold = np.percentile([fn[i] for fn in fitness], threshold_perc)
        try:
            front_threshold = [ind for ind in front if ind.fitness.values[i] >= threshold]
        except:
            raise ValueError("cannot remove ind based on threshold1 and threshold2")
    
    # sort front based on sorted_function
    sorted_function = lambda ind: sum(np.array(ind.fitness.values) * np.array(weight))
    sorted_front = sorted(front_threshold, key=sorted_function, reverse=True)
    sorted_fitness = [sorted_function(ind) for ind in sorted_front]
    # print(f"sorted_fitness: {sorted_fitness}")
    
    return sorted_front[0], sorted_fitness[0]


def calibration_basin(model, forcing, save=None, input_pet=True):
    # input pet or not
    if not input_pet:
        forcing.rename(columns={"pet": "Ep"}, inplace=True)
    
    # copy model
    model = deepcopy(model)
    
    # calibration
    weight = (0.4, 0.2, 0.2, 0.1, 0.1) if input_pet else (0.3, 0.2, 0.2, 0.2, 0.1)
    front, gridPop, fig_result, param_names, param_types  = calibration_deap_NSGAII(model.model, forcing, weight=weight,
                                                                                    start="19980101", end="20101231",
                                                                                    maxGen=500, save=save)
    
    best_params, weighted_best_fitness = select_front_weighted_fitness(front, threshold_perc=50, weight=weight)
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

def cal_fitness(forcing, sim, best_params_dict):
    # sim
    sim_qt = sim.qt
    sim_sm = sim.ws * best_params_dict["fc"]
    sim_groundwater_reservoir = sim.ws1 + sim.ws2
    sim_E = sim.et
    sim_swe = sim.snow
    
    # metrics
    metrics_qt = metrics.kling_gupta_efficiency(yobs=forcing.qt, ysim=sim_qt)
    metrics_sm = metrics.correlation(yobs=forcing.sm, ysim=sim_sm)
    metrics_groundwater_reservoir = metrics.correlation(yobs=forcing.groundwater_reservoir, ysim=sim_groundwater_reservoir)
    metrics_E = metrics.kling_gupta_efficiency(yobs=forcing.E, ysim=sim_E)
    metrics_swe = metrics.kling_gupta_efficiency(yobs=forcing.swe, ysim=sim_swe)
    
    metrics_qt = metrics_qt if not (np.isnan(metrics_qt) or np.isinf(metrics_qt)) else -9999.0
    metrics_sm = metrics_sm if not (np.isnan(metrics_sm) or np.isinf(metrics_sm)) else -9999.0
    metrics_groundwater_reservoir = metrics_groundwater_reservoir if not (np.isnan(metrics_groundwater_reservoir) or np.isinf(metrics_groundwater_reservoir)) else -9999.0
    metrics_E = metrics_E if not (np.isnan(metrics_E) or np.isinf(metrics_E)) else -9999.0
    metrics_swe = metrics_swe if not (np.isnan(metrics_swe) or np.isinf(metrics_swe)) else -9999.0
    
    # nan
    nan_num = sum(pd.isna(sim_qt)) + sum(pd.isna(sim_sm)) + sum(pd.isna(sim_groundwater_reservoir)) + sum(pd.isna(sim_E)) + sum(pd.isna(sim_swe))
    if nan_num > 0:
        metrics_qt = -9999.0
        metrics_sm = -9999.0
        metrics_groundwater_reservoir = -9999.0
        metrics_E = -9999.0
        metrics_swe = -9999.0
    
    fitness = (metrics_qt, metrics_sm, metrics_groundwater_reservoir, metrics_E, metrics_swe)
    
    return fitness
