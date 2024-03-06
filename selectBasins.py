# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from WaterBudgetClosure.decomposition_residuals import read_decomposition_func, read_fitness_func
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from Static_func.correlation_analysis import T_test

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
    

def select_basins_to_correct(plot=True):
    # read decomposition_dict
    decomposition_dict = read_decomposition_func("F:/research/WaterBudgetClosure/Decomposition")
    fitness_dict = read_fitness_func("F:/research/WaterBudgetClosure/Decomposition")
    basin_index = np.array([key for key in decomposition_dict.keys()])
    
    # data formation
    res_monthly_list = [decomposition_dict[key]["res"]["monthly"] for key in decomposition_dict]
    leakage_monthly_list = [decomposition_dict[key]["leakage"]["monthly"] for key in decomposition_dict]
    inconsistency_monthly_list = [decomposition_dict[key]["inconsistency"]["monthly"] for key in decomposition_dict]
    
    res_monthly_mean = np.array([np.mean(abs(res_)) for res_ in res_monthly_list])  # multi-year mean
    leakage_monthly_mean = np.array([np.mean(abs(leakage_)) for leakage_ in leakage_monthly_list])
    inconsistency_monthly_mean = np.array([np.mean(abs(inconsistency_)) for inconsistency_ in inconsistency_monthly_list])
    
    res_monthly_std = np.array([np.std(res_.values, ddof=1) for res_ in res_monthly_list])  # multi-year std
    leakage_monthly_std = np.array([np.std(leakage_.values, ddof=1) for leakage_ in leakage_monthly_list])
    inconsistency_monthly_std = np.array([np.std(inconsistency_.values, ddof=1) for inconsistency_ in inconsistency_monthly_list])

    res_monthly_range = np.vstack([basin_index, res_monthly_mean, res_monthly_mean + res_monthly_std, res_monthly_mean - res_monthly_std])
    leakage_monthly_range = np.vstack([basin_index, leakage_monthly_mean, leakage_monthly_mean + leakage_monthly_std, leakage_monthly_mean - leakage_monthly_std])
    inconsistency_monthly_range = np.vstack([basin_index, inconsistency_monthly_mean, inconsistency_monthly_mean + inconsistency_monthly_std, inconsistency_monthly_mean - inconsistency_monthly_std])
    res_monthly_range = res_monthly_range.T
    leakage_monthly_range = leakage_monthly_range.T
    inconsistency_monthly_range = inconsistency_monthly_range.T
    
    res_monthly_range_sorted = res_monthly_range[np.lexsort(res_monthly_range[:, ::-1].T)]
    leakage_monthly_range_sorted = leakage_monthly_range[np.lexsort(leakage_monthly_range[:, ::-1].T)]
    inconsistency_monthly_range_sorted = inconsistency_monthly_range[np.lexsort(inconsistency_monthly_range[:, ::-1].T)]
    
    # fitness
    valid_first_four_array = np.array([valid_performance(fitness_dict[key])[1] for key in fitness_dict])
    valid_swe_array = np.array([valid_performance(fitness_dict[key])[2] for key in fitness_dict])
    basin_index_ = np.array([key for key in fitness_dict.keys()])
    valid_first_four_array = np.vstack([basin_index_, valid_first_four_array])
    valid_swe_array = np.vstack([basin_index_, valid_swe_array])
    valid_first_four_array = valid_first_four_array.T
    valid_swe_array = valid_swe_array.T
    
    valid_first_four_array_sorted = valid_first_four_array[np.lexsort(valid_first_four_array[:, ::-1].T)]
    valid_swe_array_sorted = valid_swe_array[np.lexsort(valid_swe_array[:, ::-1].T)]
    
    flag_first_four = np.argwhere(valid_first_four_array_sorted[:, 1] == 0).flatten()
    flag_start_first_four = flag_first_four[np.argwhere(flag_first_four - np.roll(flag_first_four, 1).flatten() != 1)].flatten()
    flag_end_first_four = flag_first_four[np.argwhere(flag_first_four - np.roll(flag_first_four, -1).flatten() != -1)].flatten()
    
    flag_swe = np.argwhere(valid_swe_array_sorted[:, 1] == 0).flatten()
    flag_start_swe = flag_swe[np.argwhere(flag_swe - np.roll(flag_swe, 1).flatten() != 1)].flatten()
    flag_end_swe = flag_swe[np.argwhere(flag_swe - np.roll(flag_swe, -1).flatten() != -1)].flatten()
    
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
        
        # plot valid
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            y_plot = np.linspace(ymin, ymax, 100)
            
            # valid_first_four_array_sorted
            for i in range(len(flag_start_first_four)):
                start = flag_start_first_four[i]
                end = flag_end_first_four[i] + 1
                ax.fill_betweenx(y_plot, start, end, alpha=0.3, color="b", linewidth=0.05)
                ax.set_ylim(ymin, ymax)
                
            # valid_swe_array_sorted
            for i in range(len(flag_start_swe)):
                start = flag_start_swe[i]
                end = flag_end_swe[i] + 1
                ax.fill_betweenx(y_plot, start, end, alpha=0.3, color="k", linewidth=0.05)
                ax.set_ylim(ymin, ymax)
        
        # ax set
        axes[2].set_xlim([0, len(res_monthly_list) + 1])
        for ax in axes:
            ax.tick_params(labelsize=8, labelfontfamily="Arial", length=2)
            
        plt.show()
        fig.savefig("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/decomposition_basins_for_select.tiff")
        
    else:
        fig = None
        
    # sorted: max abs, then, max range(std)
    sorted_func_res_max_abs_range = lambda element: (element["res_monthly_mean"], element["res_monthly_std"])
    sorted_func_leakage_max_abs_range = lambda element: (element["leakage_monthly_mean"], element["leakage_monthly_std"])
    sorted_func_inconsistency_max_abs_range = lambda element: (element["inconsistency_monthly_mean"], element["inconsistency_monthly_std"])
    
    combined_list = []
    for i in range(len(basin_index)):
        basin_index_ = basin_index[i]
        combined_list.append({"basin_index": basin_index_,
                               "res_monthly_mean": np.mean(abs(decomposition_dict[basin_index_]["res"]["monthly"])),
                               "res_monthly_std": np.std(decomposition_dict[basin_index_]["res"]["monthly"].values, ddof=1),
                               "leakage_monthly_mean": np.mean(abs(decomposition_dict[basin_index_]["leakage"]["monthly"])),
                               "leakage_monthly_std": np.std(decomposition_dict[basin_index_]["leakage"]["monthly"].values, ddof=1),
                               "inconsistency_monthly_mean": np.mean(abs(decomposition_dict[basin_index_]["inconsistency"]["monthly"])),
                               "inconsistency_monthly_std": np.std(decomposition_dict[basin_index_]["inconsistency"]["monthly"].values, ddof=1),
                               "cal_fitness": fitness_dict[basin_index_],
                               }
                             )
    
    sorted_combined_list_based_on_res = sorted(combined_list, key=sorted_func_res_max_abs_range, reverse=True)
    sorted_combined_list_based_on_leakage = sorted(combined_list, key=sorted_func_leakage_max_abs_range, reverse=True)
    sorted_combined_list_based_on_inconsistency = sorted(combined_list, key=sorted_func_inconsistency_max_abs_range, reverse=True)
    
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_res.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - res_monthly_mean: {e['res_monthly_mean']} - res_monthly_std: {e['res_monthly_std']} - cal_fitness: {e['cal_fitness']}", file=f) for e in sorted_combined_list_based_on_res]
        
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_leakage.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - leakage_monthly_mean: {e['leakage_monthly_mean']} - leakage_monthly_std: {e['leakage_monthly_std']} - cal_fitness: {e['cal_fitness']}", file=f) for e in sorted_combined_list_based_on_leakage]
        
    with open("F:/research/WaterBudgetClosure/Correct/select_basins_to_correct/sorted_combined_list_based_on_inconsistency.txt", "a") as f:
        [print(f"basin_index: {e['basin_index']} - inconsistency_monthly_mean: {e['inconsistency_monthly_mean']} - inconsistency_monthly_std: {e['inconsistency_monthly_std']} - cal_fitness: {e['cal_fitness']}", file=f) for e in sorted_combined_list_based_on_inconsistency]
    
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
        print(f"basin_index {basin_index}", initial_decomposition["calibration_res"]["best_cal_fitness"])
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

if __name__ == "__main__":
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
    
    # select basins to correct, this use mean, however, below correct use mean(abs)
    # select_basins_to_correct(plot=True)
    
    # read and check
    selected_basin_index = [0, 6, 588] #  [0, 36, 6]
    selected_forcing_fnames = [fn for fn in forcing_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
    selected_forcing_fpaths = [os.path.join(home_forcing, n) for n in selected_forcing_fnames]
    selected_decomposition_fnames = [fn for fn in decomposition_fnames if int(fn[: fn.find('_')]) in selected_basin_index]
    selected_decomposition_fpaths = [os.path.join(home_decomposition, n) for n in selected_decomposition_fnames]
    selected_basin_index = [int(fn[: fn.find('_')]) for fn in selected_forcing_fnames]
    
    # # get performance
    select_basins_to_correct_make_sure(selected_basin_index,
                                       selected_decomposition_fnames,
                                       selected_decomposition_fpaths)
    
    # # plot map
    # select_basins_to_correct_plot_map(selected_basin_index, dpc_base, save=False)
    