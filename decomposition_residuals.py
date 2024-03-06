# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import *
from plot_func.colorbar_plot import colorbar2d
from matplotlib.colors import ListedColormap
from Static_func.correlation_analysis import T_test
from scipy import stats
import seaborn as sns
from matplotlib.path import Path
from matplotlib import cm
from distribution_func.Nonparamfit import EmpiricalDistribution
import matplotlib.dates as mdates
from copy import deepcopy
from WaterBudgetClosure.lumod_hbv.BuildModel import Lumod_HBV
from WaterBudgetClosure.lumod_hbv.CalibrateModel import calibration_basin, cal_fitness
from WaterBudgetClosure.WaterBalanceAnalysis import WaterBalanceAnalysis_correct, calResidual
from WaterBudgetClosure.correct import cal_inconsistency
from multiprocessing import Pool
plt.rcParams['font.family']='Arial'
plt.rcParams["font.weight"] = "bold"

def setHomePath(root="E:"):
    home = f"{root}/research/WaterBudgetClosure/Decomposition"
    return root, home


def Decomposition_residual(forcing, input_pet):
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
    # print("model warmup".center(50, "-"))
    
    # for i in range(5):
    #     model.warm_up(forcing, start="19980101", end="19991231")  # ["19980101", "20101231"]
    #     print(f'{model.model.params["s0"]}, {model.model.params["w01"]}, {model.model.params["w02"]}')
    
    # calibration
    best_params_dict, weighted_best_fitness, best_fitness, sim, front, gridPop = calibration_basin(model, forcing, save=None, input_pet=input_pet)
    fc = best_params_dict["fc"]
    sim.loc[:, "sm"] = sim.ws * fc
    sim.loc[:, "groundwater_reservoir"] = sim.ws1 + sim.ws2
    sim.loc[:, "all_sm"] = sim.sm + sim.groundwater_reservoir
    
    # cal_fitness
    best_cal_fitness = cal_fitness(forcing, sim, best_params_dict)
    
    # print
    print(f"best fitness: {best_fitness}")
    print(f"best_cal_fitness: {best_cal_fitness}")
    print(sim)
    
    calibration_res = {"best_params_dict": best_params_dict,
                       "weighted_best_fitness": weighted_best_fitness,
                       "best_fitness": best_fitness,
                       "best_cal_fitness": best_cal_fitness,
                       "sim": sim,
                       "front": front,
                       "gridPop": gridPop
                       }

    # init wbac
    wbac = WaterBalanceAnalysis_correct()
    basinArea = forcing.loc[forcing.index[0], "basinArea"]
    
    # cal errors
    # res
    res_all, res_quantile_all, res_std_all, res_abs_mean_all = calResidual(forcing)
    
    # leak
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
    
    leak_all, leak_quantile_all, leak_std_all, leak_abs_mean_all = calResidual(forcing_sim)

    # inconsistency
    inconsistency_all, inconsistency_abs_mean_all = cal_inconsistency(res_all, leak_all)

    errors = {"res": {"res_all": res_all, "res_abs_mean_all": res_abs_mean_all},
              "leak": {"leak_all":leak_all, "leak_abs_mean_all": leak_abs_mean_all},
              "inconsistency": {"inconsistency_all": inconsistency_all, "inconsistency_abs_mean_all": inconsistency_abs_mean_all}}
    
    
    return errors, calibration_res


def Decomposition_residual_basins_single(fpath, fname, input_pet):
    print(f"start: {fname}")
    
    # read forcing
    forcing = pd.read_csv(fpath, index_col=0)
    
    # decomposition residual
    errors, calibration_res = Decomposition_residual(forcing, input_pet)
    decomposition_basin = {"errors": errors,
                        "calibration_res": calibration_res
                        }
    
    # save
    save_home = "F:/research/WaterBudgetClosure/Decomposition/Basins"
    with open(os.path.join(save_home, f"{fname[: fname.find('.')]}_decomposition.pkl"), "wb") as f:
        pickle.dump(decomposition_basin, f)
    
    print(f"end: {fname}")
    
def Decomposition_residual_basins(cpu, input_pet):
    """
    input_pet: bool, whether to input pet as forcing, if so, weight is set as (0.3, 0.2, 0.2, 0.2, 0.1) else (0.4, 0.2, 0.2, 0.1, 0.1)

    Args:
        cpu (_type_): _description_
        input_pet (_type_): _description_
    """
    
    # general
    home = "E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing"
    fnames = [n for n in os.listdir(home) if n.endswith(".csv")]
    fpaths = [os.path.join(home, n) for n in fnames]
    
    # fname = '377_forcing_basin_6814000.csv'
    # fpath = 'E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/lumod_hbv/forcing\\377_forcing_basin_6814000.csv'
    # Decomposition_residual_basins_single(fpath, fname)

    # loop for basins to build model
    if cpu == 1:
        for i in tqdm(range(len(fnames)), desc="loop for basins to build model", colour="green"):

            # read
            fpath = fpaths[i]
            fname = fnames[i]
            
            Decomposition_residual_basins_single(fpath, fname, input_pet)

    else:
        po = Pool(cpu)
        res = [po.apply_async(Decomposition_residual_basins_single, (fpaths[i], fnames[i], input_pet)) for i in range(len(fnames))]
        po.close()
        po.join()
        
class read_decomposition:
    
    def __init__(self, home):
        decompositions_home = os.path.join(home, "Basins")
        self.fnames = [n for n in os.listdir(decompositions_home) if n.endswith(".pkl")]
        self.fpaths = [os.path.join(decompositions_home, n) for n in self.fnames]
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            self.fpath = self.fpaths[self.index]
            self.fname = self.fnames[self.index]
            
            # read
            with open(self.fpath, "rb") as f:
                self.decomposition_basin = pickle.load(f)
                
        except IndexError:
            raise StopIteration()
        
        self.index += 1
        
        return self.decomposition_basin, self.fpath, self.fname


def read_fitness_func(home):
    rd = read_decomposition(home)
    fitness_dict = dict()
    for decomposition_basin, fpath, fname in tqdm(rd, desc="loop for read cal fitness", colour="gray"):
        basin_cal_fitness = decomposition_basin["calibration_res"]["best_cal_fitness"]
        basin_index = int(fname[:fname.find("_")])
        fitness_dict[basin_index] = basin_cal_fitness
    
    return fitness_dict


def read_decomposition_func(home):
    rd = read_decomposition(home)
    decomposition_dict = dict()
    for decomposition_basin, fpath, fname in tqdm(rd, desc="loop for read decomposition", colour="gray"):
        basin_index = int(fname[:fname.find("_")])
        decomposition_dict[basin_index] = {"res": decomposition_basin["errors"]["res"]["res_all"],
                                           "leakage": decomposition_basin["errors"]["leak"]["leak_all"],
                                           "inconsistency": decomposition_basin["errors"]["inconsistency"]["inconsistency_all"]}
    
    return decomposition_dict


def plot_base_map():
    # background
    fig = plt.figure(dpi=300)
    proj = ccrs.PlateCarree()
    extent = [-125, -66.5, 24.5, 50.5]
    alpha=0.3
    ax = fig.add_axes([0.05, 0, 0.9, 1], projection=proj)
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    ax.add_feature(cfeature.LAND, alpha=alpha)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5, zorder=10, alpha=alpha)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=0.2, edgecolor="k", zorder=10, alpha=alpha)
    ax.set_extent(extent,crs=proj)
    
    return fig, ax
    
    
def plot_fitness(dpc_base, fitness_dict):
    # read
    basinArea_all = dpc_base.basin_shp.loc[:, "AREA_km2"]
    basinAreas = np.array([dpc_base.basin_shp.loc[key, "AREA_km2"] for key in fitness_dict])
    basinLatCen = np.array([dpc_base.basin_shp.loc[key, "lat_cen"] for key in fitness_dict])
    basinLonCen = np.array([dpc_base.basin_shp.loc[key, "lon_cen"] for key in fitness_dict])
    
    # plot fitness set
    radius_max = 90
    radius_min = 10
    radius_func = lambda basinArea: radius_min + (radius_max - radius_min) * (basinArea - basinArea_all.min()) / (basinArea_all.max()-basinArea_all.min())   
    radius = np.array([radius_func(area) for area in basinAreas])
    
    # fitness_qt: metrics_qt
    fig, ax = plot_base_map()
    fitness_qt = np.array([fitness_dict[key][0] for key in fitness_dict])
    cmap_qt = plt.cm.PuBu
    fitness_handle_qt = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=fitness_qt,
                                   cmap=cmap_qt,
                                   vmax=1, vmin=-0.41,
                                   edgecolor="k", linewidths=0.1)
    cmap_qt.set_under("r")
    
    # cb_qt = plt.colorbar(fitness_handle_qt, location="right", shrink=0.69, pad=0.02)
    # cb_qt.ax.tick_params(labelsize=5, direction='in')
    # fitness_qt_bool = np.array([f > -0.41 for f in fitness_qt])

    # fitness_area_qt = np.vstack((basinAreas[fitness_qt_bool], fitness_qt[fitness_qt_bool]))
    # fitness_area_qt_sorted = fitness_area_qt.T[np.lexsort(fitness_area_qt[::-1,:])].T

    # ax_sub = fig.add_axes([0.8, 0.18, 0.13, 0.2])
    
    # ax_sub.set_xlabel("sm", fontsize=5, loc="center", labelpad=0.5)
    # ax_sub.set_ylabel("grs", fontsize=5, loc="center")
    # ax_sub.tick_params(labelsize=5, direction='in', length=1)
    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance/qt.tiff")

    # fitness_sm/groundwater_reservoir storage : metrics_sm, metrics_groundwater_reservoir
    fig, ax = plot_base_map()
    
    fitness_sm = np.array([fitness_dict[key][1] for key in fitness_dict])
    fitness_grs = np.array([fitness_dict[key][2] for key in fitness_dict])
    fitness_sm_grs = np.vstack((fitness_sm, fitness_grs))
    
    # t-test, N=4748 (19980101-20101231)
    N = 4748
    fitness_sm_bool = np.array([T_test(f, N, two_side=False, right=True, alpha=0.05)[0] for f in fitness_sm])
    fitness_grs_bool = np.array([T_test(f, N, two_side=False, right=True, alpha=0.05)[0] for f in fitness_grs])
    fitness_sm_grs_bool = np.array([fitness_sm_bool[i] and fitness_grs_bool[i] for i in range(len(fitness_sm_bool))])
    
    fitness_sm_insig = fitness_sm[~fitness_sm_bool]
    fitness_grs_insig = fitness_grs[~fitness_grs_bool]
    fitness_sm_grs_sig = fitness_sm_grs[:, fitness_sm_grs_bool]
    
    # cb
    cmap_sm_grs = colorbar2d(fitness_sm_grs_sig[0, :], fitness_sm_grs_sig[1, :], 'g', 'b', maxv = 1)
    cmap_sm_grs_rgba = cmap_sm_grs.rgba()
    colorbar_x, colorbar_y, xy_color = cmap_sm_grs.colorbar(h=0.01)
    
    ax.scatter(np.array(basinLonCen)[fitness_sm_grs_bool], np.array(basinLatCen)[fitness_sm_grs_bool], marker=None,
               s=np.array(radius)[fitness_sm_grs_bool],
               c=cmap_sm_grs_rgba,
               linewidths=0.1, edgecolor="k")
    ax.scatter(np.array(basinLonCen)[~fitness_sm_bool], np.array(basinLatCen)[~fitness_sm_bool], s=15, linewidths=0.01, 
               marker="*", c="r")
    ax.scatter(np.array(basinLonCen)[~fitness_grs_bool], np.array(basinLatCen)[~fitness_grs_bool], s=10, linewidths=0.3, 
               marker="x", c="r")
    
    # ax_cb = fig.add_axes([0.8, 0.18, 0.13, 0.2])
    # ax_cb.scatter(colorbar_y, colorbar_x, c = xy_color)
    # ax_cb.set_xlim(colorbar_y.min(), colorbar_y.max())
    # ax_cb.set_ylim(colorbar_x.min(), colorbar_x.max())
    # ax_cb.set_xlabel("sm", fontsize=5, loc="right", labelpad=0.001)
    # ax_cb.set_ylabel("grs", fontsize=5, loc="top", labelpad=0.001)
    # ax_cb.tick_params(labelsize=5, direction='in', length=1)

    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance/sm_grs.tiff")
    
    # fitness_E: metrics_E
    fig, ax = plot_base_map()
    fitness_E = [fitness_dict[key][3] for key in fitness_dict]
    cmap_E = plt.cm.PuBu
    fitness_handle_E = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=fitness_E,
                                  cmap=cmap_E,
                                  vmax=1, vmin=-0.41,
                                  linewidths=0.1, edgecolor="k")
    cmap_E.set_under("r")
    # cb_E = plt.colorbar(fitness_handle_E, location="right", shrink=0.69, pad=0.02)
    # cb_E.ax.tick_params(labelsize=5, direction='in')

    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance/E.tiff")
    
    # fitness_swe: metrics_swe
    fig, ax = plot_base_map()
    fitness_swe = [fitness_dict[key][4] for key in fitness_dict]
    cmap_swe = plt.cm.PuBu
    fitness_handle_swe = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=fitness_swe,
                                    cmap=cmap_swe,
                                    vmax=1, vmin=-0.41,
                                    linewidths=0.1, edgecolor="k")
    cmap_swe.set_under("r")
    # cb_swe = plt.colorbar(fitness_handle_swe, location="right", shrink=0.69, pad=0.02)
    # cb_swe.ax.tick_params(labelsize=5, direction='in')

    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance/swe.tiff")


def factors_map(factors):
    gel1_class = factors.loc[:, 'camels_geol:geol_1st_class']
    gel2_class = factors.loc[:, 'camels_geol:geol_2nd_class']
    gel1_class_set = list(set(gel1_class))
    gel2_class_set = list(set(gel2_class))
    gel1_class_set_map = dict(zip(gel1_class_set, list(range(len(gel1_class_set)))))
    gel2_class_set_map = dict(zip(gel2_class_set, list(range(len(gel2_class_set)))))
    gel1_class_func = lambda gel1_class: gel1_class_set_map[gel1_class]
    gel2_class_func = lambda gel2_class: gel2_class_set_map[gel2_class]
    
    factors.loc[:, 'camels_geol:geol_1st_class'] = factors.loc[:, 'camels_geol:geol_1st_class'].apply(gel1_class_func)
    factors.loc[:, 'camels_geol:geol_2nd_class'] = factors.loc[:, 'camels_geol:geol_2nd_class'].apply(gel2_class_func)
    
    factors.loc[:, 'camels_geol:geol_1st_class'] = factors.loc[:, 'camels_geol:geol_1st_class'].astype(int)
    factors.loc[:, 'camels_geol:geol_2nd_class'] = factors.loc[:, 'camels_geol:geol_2nd_class'].astype(int)
    
    dom_land_cover = factors.loc[:, 'camels_vege:dom_land_cover']
    dom_land_cover_set = list(set(dom_land_cover))
    dom_land_cover_set_map = dict(zip(dom_land_cover_set, list(range(len(dom_land_cover_set)))))
    dom_land_cover_func = lambda dom_land_cover: dom_land_cover_set_map[dom_land_cover]
    factors.loc[:, 'camels_vege:dom_land_cover'] = factors.loc[:, 'camels_vege:dom_land_cover'].apply(dom_land_cover_func)
    factors.loc[:, 'camels_vege:dom_land_cover'] = factors.loc[:, 'camels_vege:dom_land_cover'].astype(int)
    
    return factors


def factors_Normalization(factors):
    factors_stand = (factors - factors.mean()) / factors.std()
    return factors_stand

def group_fitness(fitness, fitness_bool, basinIndex):
    fitness_effective = fitness[fitness_bool]
    fitness_effective_basinIndex = basinIndex[fitness_bool]
    
    fitness_ineffective = fitness[~fitness_bool]
    fitness_ineffective_basinIndex = basinIndex[~fitness_bool]
    
    fitness_first50_bool = fitness_effective > fitness_effective.mean()
    fitness_last50_bool = fitness_effective < fitness_effective.mean()
    
    fitness_first50 = fitness_effective[fitness_first50_bool]
    fitness_first50_basinIndex = fitness_effective_basinIndex[fitness_first50_bool]
    fitness_last50 = fitness_effective[fitness_last50_bool]
    fitness_last50_basinIndex = fitness_effective_basinIndex[fitness_last50_bool]
    
    fitness_effective_dict = {"fitness": fitness_effective, "basinIndex": fitness_effective_basinIndex}
    fitness_ineffective_dict = {"fitness": fitness_ineffective, "basinIndex": fitness_ineffective_basinIndex}
    fitness_first50_dict = {"fitness": fitness_first50, "basinIndex": fitness_first50_basinIndex}
    fitness_last50_dict = {"fitness": fitness_last50, "basinIndex": fitness_last50_basinIndex}
    
    return fitness_ineffective_dict, fitness_effective_dict, fitness_last50_dict, fitness_first50_dict
    

def create_stripplot_df(factors, fitness_group, factor_column):
    combined_df = pd.concat(objs=[factors.loc[fitness_group[0]["basinIndex"], factor_column],
                                  factors.loc[fitness_group[1]["basinIndex"], factor_column],
                                  factors.loc[fitness_group[2]["basinIndex"], factor_column],
                                  factors.loc[fitness_group[3]["basinIndex"], factor_column]],
                            axis=0)
    combined_df = pd.DataFrame(combined_df)
    combined_df.rename(columns={'factor_column': 'fitness'}, inplace=True)
    combined_df.loc[:, "type"] = np.hstack([np.full((len(factors.loc[fitness_group[0]["basinIndex"], factor_column]), ), fill_value="ineff", dtype="S5"),
                                            np.full((len(factors.loc[fitness_group[1]["basinIndex"], factor_column]), ), fill_value="eff", dtype="S5"),
                                            np.full((len(factors.loc[fitness_group[2]["basinIndex"], factor_column]), ), fill_value="B50%", dtype="S5"),
                                            np.full((len(factors.loc[fitness_group[3]["basinIndex"], factor_column]), ), fill_value="T50%", dtype="S5")])
    
    # remove nan
    combined_df = combined_df.dropna()
    
    return combined_df


def stripplot_factors(factors, fitness_group, column, ax):
    # stripplot
    stripplot_df = create_stripplot_df(factors, fitness_group, column)
    sns.stripplot(data=stripplot_df, x="type", y=column, hue="type",
                alpha=.25, zorder=1, legend=False, size=1.5, order=["ineff", "eff", "B50%", "T50%"],
                c='r', palette=['crimson', 'darkblue', 'cornflowerblue', 'royalblue'],
                ax=ax)
    
    # boxplot
    boxplot_list = [factors.loc[fitness_group[0]["basinIndex"], column],
                     factors.loc[fitness_group[1]["basinIndex"], column],
                     factors.loc[fitness_group[2]["basinIndex"], column],
                     factors.loc[fitness_group[3]["basinIndex"], column]]
    boxplot_list = [a.dropna() for a in boxplot_list]
    
    ax.boxplot(boxplot_list,
               labels=["ineff", "eff", "B50%", "T50%"],
               showbox=False, showcaps=False,
               whis=[0, 100], whiskerprops={"color": "k", "linestyle": None, "alpha": 1},
               medianprops={"color": "r", "linestyle": None, "alpha": 1},
               positions=[0, 1, 2, 3])
    ax.set_xlabel(["ineff", "eff", "B50%", "T50%"], font={"size": 5, "family": "Arial", "weight": "normal"})
    
    # lines
    ylim = ax.get_ylim()
    for x in [-0.5, 0.5, 1.5, 2.5]:
        ax.vlines(x=x, ymin=ylim[0], ymax=ylim[-1], color="gray", linewidth=0.5, alpha=0.5)
        ax.set_ylim(ylim)
        
    # set
    ax.tick_params(labelsize=5, labelfontfamily="Arial", length=1, pad=1)
    yticks = ax.get_yticks()[1:-1]
    ax.set_yticks(yticks, [f"{t:>4.1f}" if abs(t) < 10 else f"{t:>4.0f}" for t in yticks])

    
def annoationplot_factors(factors, fitness_group, column, ax):
    # get percent
    median_factors = [np.nanmedian(factors.loc[fitness_group[0]["basinIndex"], column]),
                      np.nanmedian(factors.loc[fitness_group[1]["basinIndex"], column]),
                      np.nanmedian(factors.loc[fitness_group[2]["basinIndex"], column]),
                      np.nanmedian(factors.loc[fitness_group[3]["basinIndex"], column])]
    all_factors = factors.loc[:, column]
    em = EmpiricalDistribution()
    em.fit(all_factors)
    percent = em.cdf(median_factors)
    # all_len = len(fitness_group[0]["fitness"]) + len(fitness_group[1]["fitness"])
    # percent = [len(fitness_group[0]["fitness"]) / all_len,
    #            len(fitness_group[1]["fitness"]) / all_len,
    #            len(fitness_group[2]["fitness"]) / all_len,
    #            len(fitness_group[3]["fitness"]) / all_len]
    percent_func = lambda pt: f"{pt:^5.0%}" if pt > 0.1 else f"{pt:^6.0%}"
    
    # plot
    cmap_colors =cm.get_cmap("BrBG", 100)
    font_color_func = lambda pt: "w" if ((int(pt * 100) < 25) or (int(pt * 100) > 75)) else "k"
    ax.annotate(percent_func(percent[0]), xy=(-0.4, 1.07), xycoords=('data', 'axes fraction'),
                 fontfamily="Arial",  fontsize=5, color=font_color_func(percent[0]), fontweight="bold",
                 bbox={"boxstyle": "Square", "alpha": 1, "fc":cmap_colors(int(percent[0] * 100)), "ec": "w", "lw": 0.3},
                 )

    ax.annotate(percent_func(percent[1]), xy=(0.278, 1.07), xycoords=('axes fraction', 'axes fraction'),
                fontfamily="Arial",  fontsize=5, color=font_color_func(percent[1]), fontweight="bold",
                bbox={"boxstyle": "Square", "alpha": 1, "fc":cmap_colors(int(percent[1] * 100)), "ec": "w", "lw": 0.3},
                )
    
    ax.annotate(percent_func(percent[2]), xy=(0.535, 1.07), xycoords=('axes fraction', 'axes fraction'),
                fontfamily="Arial",  fontsize=5, color=font_color_func(percent[2]), fontweight="bold",
                bbox={"boxstyle": "Square", "alpha": 1, "fc":cmap_colors(int(percent[2] * 100)), "ec": "w", "lw": 0.5},
                )
    
    ax.annotate(percent_func(percent[3]), xy=(0.785, 1.07), xycoords=('axes fraction', 'axes fraction'),
                fontfamily="Arial",  fontsize=5, color=font_color_func(percent[3]), fontweight="bold",
                bbox={"boxstyle": "Square", "alpha": 1, "fc":cmap_colors(int(percent[3] * 100)), "ec": "w", "lw": 0.5},
                )
    

def ttest_factors(factors, fitness_group):
    factors_effective = factors.loc[fitness_group[0]["basinIndex"], :] 
    factors_ineffective = factors.loc[fitness_group[1]["basinIndex"], :]
    
    ttest_ret = pd.DataFrame(index=["statistic", "pvalue", "significance"], columns=factors_effective.columns)
    alpha = 0.05
    for column in factors_effective.columns:
        factors_effective_column = factors_effective.loc[:, column]
        factors_ineffective_column = factors_ineffective.loc[:, column]
        statistic, pvalue = stats.ttest_ind_from_stats(factors_effective_column.mean(),
                                                        factors_effective_column.std(),
                                                        len(factors_effective_column),
                                                        factors_ineffective_column.mean(),
                                                        factors_ineffective_column.std(),
                                                        len(factors_ineffective_column),
                                                        equal_var=False,
                                                        alternative="two-sided")
        ttest_ret.loc["statistic", column] = statistic
        ttest_ret.loc["pvalue", column] = pvalue
        ttest_ret.loc["significance", column] = pvalue < alpha
    
    significant_factors = [column for column in ttest_ret.columns if ttest_ret.loc["significance", column]]
    print(significant_factors)
    
    return significant_factors

def performance_analysis(dpc_base, fitness_dict):
    # read
    N = 4748
    
    basinIndex = np.array(list(fitness_dict.keys()))
    basinArea_all = dpc_base.basin_shp.loc[:, "AREA_km2"]
    basinAreas = np.array([dpc_base.basin_shp.loc[key, "AREA_km2"] for key in fitness_dict])
    basinLatCen = np.array([dpc_base.basin_shp.loc[key, "lat_cen"] for key in fitness_dict])
    basinLonCen = np.array([dpc_base.basin_shp.loc[key, "lon_cen"] for key in fitness_dict])
    fitness_qt = np.array([fitness_dict[key][0] for key in fitness_dict])
    fitness_sm = np.array([fitness_dict[key][1] for key in fitness_dict])
    fitness_grs = np.array([fitness_dict[key][2] for key in fitness_dict])
    fitness_E = np.array([fitness_dict[key][3] for key in fitness_dict])
    fitness_swe = np.array([fitness_dict[key][4] for key in fitness_dict])
    
    fitness_qt_bool = np.array([f > -0.41 for f in fitness_qt])
    fitness_E_bool = np.array([f > -0.41 for f in fitness_E])
    fitness_swe_bool = np.array([f > -0.41 for f in fitness_swe])
    fitness_sm_bool = np.array([T_test(f, N, two_side=False, right=True, alpha=0.05)[0] for f in fitness_sm])
    fitness_grs_bool = np.array([T_test(f, N, two_side=False, right=True, alpha=0.05)[0] for f in fitness_grs])
    
    # possible factor
    factors_keys = ['elev_mean', 'AREA_km2',
       'camels_clim:p_mean', 'camels_clim:pet_mean',
       'camels_clim:p_seasonality', 'camels_clim:frac_snow',
       'camels_clim:aridity', 'camels_clim:high_prec_freq',
       'camels_clim:high_prec_dur',
       'camels_clim:low_prec_freq', 'camels_clim:low_prec_dur',
       'camels_geol:geol_1st_class', 'camels_geol:glim_1st_class_frac',
       'camels_geol:geol_2nd_class', 'camels_geol:glim_2nd_class_frac',
       'camels_geol:carbonate_rocks_frac', 'camels_geol:geol_porostiy',
       'camels_geol:geol_permeability',
       'camels_hydro:q_mean', 'camels_hydro:runoff_ratio',
       'camels_hydro:slope_fdc', 'camels_hydro:baseflow_index',
       'camels_hydro:stream_elas', 'camels_hydro:q5', 'camels_hydro:q95',
       'camels_hydro:high_q_freq', 'camels_hydro:high_q_dur',
       'camels_hydro:low_q_freq', 'camels_hydro:low_q_dur',
       'camels_hydro:zero_q_freq', 'camels_hydro:hfd_mean',
       'camels_soil:soil_depth_pelletier',
       'camels_soil:soil_depth_statsgo', 'camels_soil:soil_porosity',
       'camels_soil:soil_conductivity', 'camels_soil:max_water_content',
       'camels_soil:sand_frac', 'camels_soil:silt_frac',
       'camels_soil:clay_frac', 'camels_soil:water_frac',
       'camels_soil:organic_frac', 'camels_soil:other_frac',
       'camels_topo:gauge_lat',
       'camels_topo:gauge_lon', 'camels_topo:elev_mean',
       'camels_topo:slope_mean', 'camels_topo:area_gages2',
       'camels_topo:area_geospa_fabric',
       'camels_vege:frac_forest', 'camels_vege:lai_max',
       'camels_vege:lai_diff', 'camels_vege:gvf_max', 'camels_vege:gvf_diff',
       'camels_vege:dom_land_cover_frac', 'camels_vege:dom_land_cover',
       'camels_vege:root_depth_50', 'camels_vege:root_depth_99']
    
    # factors   
    factors = dpc_base.basin_shp.loc[:, factors_keys]
    factors = factors_map(factors)
    
    # group: fitness_ineffective_dict, fitness_effective_dict, fitness_last50_dict, fitness_first50_dict
    fitness_qt_group = group_fitness(fitness_qt, fitness_qt_bool, basinIndex)
    fitness_sm_group = group_fitness(fitness_sm, fitness_sm_bool, basinIndex)
    fitness_grs_group = group_fitness(fitness_grs, fitness_grs_bool, basinIndex)
    fitness_E_group = group_fitness(fitness_E, fitness_E_bool, basinIndex)
    fitness_swe_group = group_fitness(fitness_swe, fitness_swe_bool, basinIndex)
    
    # static
    # np.median(fitness_qt_group[1]["fitness"])  # 0.4756291201917883
    # (fitness_qt_group[1]["fitness"].min(), fitness_qt_group[1]["fitness"].max())  # (-0.391322419938694, 0.8482530715907253)
    # len(fitness_qt_group[1]["fitness"]), len(fitness_qt_group[1]["fitness"]) / len(fitness_qt)  # 508, 0.777947932618683
    
    # np.median(fitness_sm_group[1]["fitness"])  # 0.7902663627714426
    # (fitness_sm_group[1]["fitness"].min(), fitness_sm_group[1]["fitness"].max())  # (0.0344049868906197, 0.9471876735142971)
    # len(fitness_sm_group[1]["fitness"]), len(fitness_sm_group[1]["fitness"]) / len(fitness_sm)  # 570, 0.8728943338437979
    
    # np.median(fitness_grs_group[1]["fitness"])  # 0.7243615708257609
    # (fitness_grs_group[1]["fitness"].min(), fitness_grs_group[1]["fitness"].max())  # (0.03099597704187763, 0.9352658896295106)
    # len(fitness_grs_group[1]["fitness"]), len(fitness_grs_group[1]["fitness"]) / len(fitness_grs)  # 567, 0.8683001531393568
    
    # np.median(fitness_E_group[1]["fitness"])  # 0.9508942347148424
    # (fitness_E_group[1]["fitness"].min(), fitness_E_group[1]["fitness"].max())  # (-0.4099023727190634, 0.998071303296753)
    # len(fitness_E_group[1]["fitness"]), len(fitness_E_group[1]["fitness"]) / len(fitness_E)  # 511, 0.7825421133231241
    
    # np.median(fitness_swe_group[1]["fitness"])  # 0.4083550558271627
    # (fitness_swe_group[1]["fitness"].min(), fitness_swe_group[1]["fitness"].max())  # (-0.4038809916549724, 0.8664992699051783)
    # len(fitness_swe_group[1]["fitness"]), len(fitness_swe_group[1]["fitness"]) / len(fitness_swe)  # 303, 0.46401225114854516
    
    # analysis: qt
    significant_factors_all_qt = ttest_factors(factors, fitness_qt_group)
    significant_factors = np.array([['camels_clim:p_mean', 'camels_clim:pet_mean', 'camels_clim:p_seasonality', 'camels_clim:frac_snow', 'camels_clim:aridity'],
                           ['camels_geol:geol_porostiy', 'camels_geol:geol_permeability', 'camels_soil:soil_depth_statsgo', 'camels_soil:silt_frac', 'camels_soil:max_water_content'],
                           ['camels_hydro:q_mean', 'camels_hydro:runoff_ratio', 'camels_hydro:baseflow_index', 'camels_hydro:zero_q_freq', 'camels_hydro:high_q_freq'],
                           ['camels_vege:frac_forest', 'camels_vege:lai_max', 'camels_vege:lai_diff', 'camels_vege:gvf_max', 'camels_vege:gvf_diff']])
    
    y_labels = np.array([['Clim: P Mean', 'Clim : Pet Mean', 'Clim: P Seasonality', 'Clim: Snow Frac', 'Clim: Aridity'],
                ['Geol: Porostiy', 'Geol: Permeability', 'Soil: Soil Depth', 'Soil: Silt Frac', 'Soil: Max Water Content'],
                ['Hydro: Q mean', 'Hydro: Runoff Ratio', 'Hydro: Baseflow Index', 'Hydro: Zero Q Freq', 'Hydro: High Q Freq'],
                ['Vege: Forest Frac', 'Vege: Lai Max', 'Vege: Lai Diff', 'Vege: GVF Max', 'Vege: GVF Diff']])
    
    # plot: qt
    fig, ax = plt.subplots(4, 5, dpi=300, gridspec_kw={"left":0.05, "right":0.98, "bottom":0.05, "top":0.95, "wspace": 0.38, "hspace": 0.2}, sharex=False)    
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax_ = ax[i, j]
            column_ = significant_factors[i, j]
            stripplot_factors(factors, fitness_qt_group, column_, ax_)
            annoationplot_factors(factors, fitness_qt_group, column_, ax_)
            ax_.set_ylabel(y_labels[i, j], fontsize="5", family="Arial", labelpad=0.5)
            ax_.set_xlabel(None)

    # fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance_analysis/performance_analysis_qt.tiff")
    
    # analysis: E
    significant_factors_all_E = ttest_factors(factors, fitness_E_group)
    # TODO
    significant_factors = np.array([['camels_clim:p_mean', 'camels_clim:pet_mean', 'camels_clim:p_seasonality', 'camels_clim:aridity'],
                           ['camels_soil:soil_depth_statsgo', 'camels_soil:silt_frac', 'camels_soil:other_frac'],
                           ['camels_hydro:q_mean', 'camels_hydro:runoff_ratio', 'camels_hydro:slope_fdc', 'camels_hydro:baseflow_index', 'camels_hydro:high_q_freq'],
                           ['camels_vege:frac_forest', 'camels_vege:lai_max', 'camels_vege:lai_diff', 'camels_vege:gvf_max', 'camels_vege:gvf_diff']])
    
    y_labels = np.array([['Clim: P Mean', 'Clim : Pet Mean', 'Clim: P Seasonality', 'Clim: Snow Frac', 'Clim: Aridity'],
                ['Geol: Porostiy', 'Geol: Permeability', 'Soil: Soil Depth', 'Soil: Silt Frac', 'Soil: Other Frac'],
                ['Hydro: Q mean', 'Hydro: Runoff Ratio', 'Hydro: FDC Slope', 'Hydro: Baseflow Index', 'Hydro: High Q Freq'],
                ['Vege: Forest Frac', 'Vege: Lai Max', 'Vege: Lai Diff', 'Vege: GVF Max', 'Vege: GVF Diff']])
    
    # plot: E
    fig, ax = plt.subplots(4, 5, dpi=300, gridspec_kw={"left":0.05, "right":0.98, "bottom":0.05, "top":0.95, "wspace": 0.38, "hspace": 0.2}, sharex=False)    
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax_ = ax[i, j]
            column_ = significant_factors[i, j]
            stripplot_factors(factors, fitness_qt_group, column_, ax_)
            annoationplot_factors(factors, fitness_qt_group, column_, ax_)
            ax_.set_ylabel(y_labels[i, j], fontsize="5", family="Arial", labelpad=0.5)
            ax_.set_xlabel(None)

    # fig.savefig("F:/research/WaterBudgetClosure/Decomposition/Performance_analysis/performance_analysis_E.tiff")
    

def create_ridgeplot_df(decomposition_months_list, months_text):
    for i in range(len(decomposition_months_list)):
        decomposition_month_array = np.array(decomposition_months_list[i])
        month = months_text[i]
        month_array = np.full_like(decomposition_month_array, fill_value=month, dtype="S3")
        month_int_array = np.full_like(decomposition_month_array, fill_value=i + 1, dtype=int)
        decomposition_month_df = pd.DataFrame(columns=["value", "month", "month_int"])
        decomposition_month_df.loc[:, "value"] = decomposition_month_array
        decomposition_month_df.loc[:, "month"] = month_array
        decomposition_month_df.loc[:, "month_int"] = month_int_array
        
        if i == 0:
            combined_df = decomposition_month_df
        else:
            combined_df = pd.concat([combined_df, decomposition_month_df], axis=0)
    
    return combined_df
    
def ridgeplot_decomposition(decomposition_months_df):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(decomposition_months_df, row="month", hue="month", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "value",
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
        
    g.map(label, "value")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    

def boxplot_decomposition(decomposition_months_df, months_color):
    sns.set_theme(palette=months_color,
                  style="darkgrid",
                  font="Arial", font_scale=0.8,
                  rc={"figure.dpi": 300})
    
    # boxplot
    g = sns.catplot(
        data=decomposition_months_df, x="value", y="month", hue="month",
        native_scale=True, zorder=1, orient="h", legend=False,
        kind="box", errorbar="sd", showfliers=False,
        medianprops={"color": "k", "linestyle": None, "alpha": 1, "linewidth": 1.5},
        notch=True
    )
    xlim = g.ax.get_xlim()
    ylim = g.ax.get_ylim()
    
    # stripplot
    sns.stripplot(decomposition_months_df, x="value", y="month",
                  color=".3", size=1,
                  alpha=0.1, zorder=1, 
                  ) # palette=months_color,
    
    # x = 0 line
    g.ax.vlines(x=0, ymin=ylim[0], ymax=ylim[1], color='red',
                linestyles="--", linewidths=1.5, zorder=3)
    
    # set
    g.ax.set_xlim(*xlim)
    g.ax.set_ylim(*ylim)
    g.ax.set_xlabel(None)
    g.ax.set_ylabel(None)
    g.figure.set_size_inches((4, 6))
    
    return g


def scatterplot_decomposition_ratio(leakage_ratio_list, inconsistency_ratio_list, date, vmin=-3, vmax=3, s=10, plot_inconsistency=True, plotcb=False):
    # get array
    leakage_ratio_array = np.array(leakage_ratio_list)
    leakage_ratio_array_mean = np.nanmean(leakage_ratio_array, axis=1)
    leakage_bool_remove_min = leakage_ratio_array_mean < vmin
    leakage_bool_remove_max = leakage_ratio_array_mean > vmax
    leakage_bool_remove = np.array([leakage_bool_remove_min[i] or leakage_bool_remove_max[i] for i in range(len(leakage_bool_remove_min))])
    leakage_ratio_array_mean_r = leakage_ratio_array_mean[~leakage_bool_remove]
    leakage_date = date[~leakage_bool_remove]
    
    inconsistency_ratio_array = np.array(inconsistency_ratio_list)
    inconsistency_ratio_array_mean = np.nanmean(inconsistency_ratio_array, axis=1)
    inconsistency_bool_remove_min = inconsistency_ratio_array_mean < vmin
    inconsistency_bool_remove_max = inconsistency_ratio_array_mean > vmax
    inconsistency_bool_remove = np.array([inconsistency_bool_remove_min[i] or inconsistency_bool_remove_max[i] for i in range(len(inconsistency_bool_remove_min))])
    inconsistency_ratio_array_mean_r = inconsistency_ratio_array_mean[~inconsistency_bool_remove]
    inconsistency_date = date[~inconsistency_bool_remove]
    
    # plot
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.labelsize"] = 3
    fig_scatter, ax = plt.subplots(dpi=300)
    scatter_handle = ax.scatter(leakage_date, leakage_ratio_array_mean_r,
                                c=leakage_ratio_array_mean_r,
                                cmap=plt.cm.get_cmap("seismic"), 
                                edgecolor="k",
                                linewidth=0.2,
                                marker="o",
                                s=s,
                                vmin=vmin, vmax=vmax)
    
    if plot_inconsistency:
        ax.scatter(inconsistency_date,
                   inconsistency_ratio_array_mean_r,
                   c=inconsistency_ratio_array_mean_r,
                   cmap=plt.cm.get_cmap("seismic"), 
                   edgecolor="k",
                   linewidth=0.2,
                   marker="^",
                   s=s,
                   vmin=vmin, vmax=vmax)
    
    ax.set_ylim([vmin, vmax])
    ax.set_xlim([date[0], date[-1]])
    
    # colorbar
    if plotcb:
        plt.colorbar(scatter_handle, orientation="vertical", location="right")
    
    return fig_scatter


def boxplot_decomposition_ratio(leakage_ratio_list, inconsistency_ratio_list, date, vmin=-3, vmax=3):
    # get array
    leakage_ratio_array = np.array(leakage_ratio_list)
    leakage_ratio_array_mean = np.nanmean(leakage_ratio_array, axis=1)
    leakage_bool_remove_min = leakage_ratio_array_mean < vmin
    leakage_bool_remove_max = leakage_ratio_array_mean > vmax
    leakage_bool_remove = np.array([leakage_bool_remove_min[i] or leakage_bool_remove_max[i] for i in range(len(leakage_bool_remove_min))])
    leakage_ratio_array_mean_r = leakage_ratio_array_mean[~leakage_bool_remove]
    leakage_date = date[~leakage_bool_remove]
    
    inconsistency_ratio_array = np.array(inconsistency_ratio_list)
    inconsistency_ratio_array_mean = np.nanmean(inconsistency_ratio_array, axis=1)
    inconsistency_bool_remove_min = inconsistency_ratio_array_mean < vmin
    inconsistency_bool_remove_max = inconsistency_ratio_array_mean > vmax
    inconsistency_bool_remove = np.array([inconsistency_bool_remove_min[i] or inconsistency_bool_remove_max[i] for i in range(len(inconsistency_bool_remove_min))])
    inconsistency_ratio_array_mean_r = inconsistency_ratio_array_mean[~inconsistency_bool_remove]
    inconsistency_date = date[~inconsistency_bool_remove]
    
    # boxplot
    sns.set_theme(font="Arial", font_scale=0.8,
                    rc={"figure.dpi": 300})
    leakage_df = pd.DataFrame(columns=["value", "type"])
    inconsistency_df = pd.DataFrame(columns=["value", "type"])
    type_leakage = np.full_like(leakage_ratio_array_mean_r, fill_value="leakage", dtype="S7")
    type_leakage = type_leakage.astype(str)
    type_inconsistency = np.full_like(inconsistency_ratio_array_mean_r, fill_value="inconsistency", dtype="S13")
    type_inconsistency = type_inconsistency.astype(str)
    leakage_df.loc[:, "value"] = leakage_ratio_array_mean_r
    leakage_df.loc[:, "type"] = type_leakage
    inconsistency_df.loc[:, "value"] = inconsistency_ratio_array_mean_r
    inconsistency_df.loc[:, "type"] = type_inconsistency
    
    combined_df = pd.concat([leakage_df, inconsistency_df], axis=0)
    
    g_boxplot = sns.violinplot(data=combined_df, y="value",  hue="type",
                                split=True,
                                inner="quart",
                                fill=False,
                                palette={"leakage": "g", "inconsistency": ".35"})
    g_boxplot.get_legend().remove()
    g_boxplot.set_xlabel(None)
    g_boxplot.set_ylabel(None)
    g_boxplot.figure.set_size_inches((3, 6))
    
    return g_boxplot
    
def plot_decomposition(dpc_base, decomposition_dict):
    # read
    basinArea_all = dpc_base.basin_shp.loc[:, "AREA_km2"]
    basinAreas = np.array([dpc_base.basin_shp.loc[key, "AREA_km2"] for key in decomposition_dict])
    basinLatCen = np.array([dpc_base.basin_shp.loc[key, "lat_cen"] for key in decomposition_dict])
    basinLonCen = np.array([dpc_base.basin_shp.loc[key, "lon_cen"] for key in decomposition_dict])
    
    # plot decomposition set
    radius_max = 90
    radius_min = 30
    radius_func = lambda basinArea: radius_min + (radius_max - radius_min) * (basinArea - basinArea_all.min()) / (basinArea_all.max()-basinArea_all.min())   
    radius = np.array([radius_func(area) for area in basinAreas])
    
    # get statics
    res_monthly_list = [decomposition_dict[key]["res"]["monthly"] for key in decomposition_dict]
    leakage_monthly_list = [decomposition_dict[key]["leakage"]["monthly"] for key in decomposition_dict]
    inconsistency_monthly_list = [decomposition_dict[key]["inconsistency"]["monthly"] for key in decomposition_dict]
    
    res_daily_list = [decomposition_dict[key]["res"]["daily"] for key in decomposition_dict]
    leakage_daily_list = [decomposition_dict[key]["leakage"]["daily"] for key in decomposition_dict]
    inconsistency_daily_list = [decomposition_dict[key]["inconsistency"]["daily"] for key in decomposition_dict]
    
    res_monthly_mean = np.array([res_.mean() for res_ in res_monthly_list])  # multi-year mean
    leakage_monthly_mean = np.array([leakage_.mean() for leakage_ in leakage_monthly_list])
    inconsistency_monthly_mean = np.array([inconsistency_.mean() for inconsistency_ in inconsistency_monthly_list])
    
    res_monthly_mean_perc = np.percentile(res_monthly_mean, [0, 10, 25, 50, 75, 90, 100])
    leakage_monthly_mean_perc = np.percentile(leakage_monthly_mean, [0, 10, 25, 50, 75, 90, 100])
    inconsistency_monthly_mean_perc = np.percentile(inconsistency_monthly_mean, [0, 10, 25, 50, 75, 90, 100])
    
    uppack_func = lambda l: [y for x in l for y in x]
    
    res_months_list = [[res_monthly_list[j].loc[res_monthly_list[j].index.month == i, :].values.flatten().tolist() for j in range(len(res_monthly_list))] for i in range(1, 13)]
    res_months_list = [uppack_func(res_months) for res_months in res_months_list]
    leakage_months_list = [[leakage_monthly_list[j].loc[leakage_monthly_list[j].index.month == i, :].values.flatten().tolist() for j in range(len(leakage_monthly_list))] for i in range(1, 13)]
    leakage_months_list = [uppack_func(leakage_months) for leakage_months in leakage_months_list]
    inconsistency_months_list = [[inconsistency_monthly_list[j].loc[inconsistency_monthly_list[j].index.month == i, :].values.flatten().tolist() for j in range(len(inconsistency_monthly_list))] for i in range(1, 13)]
    inconsistency_months_list = [uppack_func(inconsistency_months) for inconsistency_months in inconsistency_months_list]
    
    months_text = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months_color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]
    
    res_months_df = create_ridgeplot_df(res_months_list, months_text)
    leakage_months_df = create_ridgeplot_df(leakage_months_list, months_text)
    inconsistency_months_df = create_ridgeplot_df(inconsistency_months_list, months_text)
    
    # monthly ratio list
    res_monthly_ratio_list = [res_monthly_list[i] for i in range(len(res_monthly_list))]  # same shape with ratio, used for plot
    leakage_monthly_ratio_list = [leakage_monthly_list[i] / res_monthly_list[i] for i in range(len(leakage_monthly_list))]
    inconsistency_monthly_ratio_list = [inconsistency_monthly_list[i] / res_monthly_list[i] for i in range(len(inconsistency_monthly_list))]
    
    res_monthly_ratio_list = [[res_monthly_ratio_list[j].loc[res_monthly_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(res_monthly_ratio_list))] for date in res_monthly_ratio_list[0].index]
    leakage_months_ratio_list = [[leakage_monthly_ratio_list[j].loc[leakage_monthly_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(leakage_monthly_ratio_list))] for date in leakage_monthly_ratio_list[0].index]
    inconsistency_months_ratio_list = [[inconsistency_monthly_ratio_list[j].loc[inconsistency_monthly_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(inconsistency_monthly_ratio_list))] for date in inconsistency_monthly_ratio_list[0].index]
    
    res_months_ratio_list = [uppack_func(res_months_ratio) for res_months_ratio in res_monthly_ratio_list]
    leakage_months_ratio_list = [uppack_func(leakage_months_ratio) for leakage_months_ratio in leakage_months_ratio_list]
    inconsistency_months_ratio_list = [uppack_func(inconsistency_months_ratio) for inconsistency_months_ratio in inconsistency_months_ratio_list]
    
    date_months = leakage_monthly_ratio_list[0].index
    
    # daily ratio list
    res_daily_ratio_list = [res_daily_list[i] for i in range(len(res_daily_list))]
    leakage_daily_ratio_list = [leakage_daily_list[i] / res_daily_list[i] for i in range(len(leakage_daily_list))]
    inconsistency_daily_ratio_list = [inconsistency_daily_list[i] / res_daily_list[i] for i in range(len(inconsistency_daily_list))]
    
    res_days_ratio_list = [[res_daily_ratio_list[j].loc[res_daily_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(res_daily_ratio_list))] for date in res_daily_ratio_list[0].index]
    leakage_days_ratio_list = [[leakage_daily_ratio_list[j].loc[leakage_daily_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(leakage_daily_ratio_list))] for date in leakage_daily_ratio_list[0].index]
    inconsistency_days_ratio_list = [[inconsistency_daily_ratio_list[j].loc[inconsistency_daily_ratio_list[j].index == date, :].values.flatten().tolist() for j in range(len(inconsistency_daily_ratio_list))] for date in inconsistency_daily_ratio_list[0].index]
    
    res_days_ratio_list = [uppack_func(res_days_ratio) for res_days_ratio in res_days_ratio_list]
    leakage_days_ratio_list = [uppack_func(leakage_days_ratio) for leakage_days_ratio in leakage_days_ratio_list]
    inconsistency_days_ratio_list = [uppack_func(inconsistency_days_ratio) for inconsistency_days_ratio in inconsistency_days_ratio_list]
    
    date_days = leakage_daily_ratio_list[0].index
    
    # plot timeseries ratio
    fig_scatterplot_monthly = scatterplot_decomposition_ratio(leakage_months_ratio_list, inconsistency_months_ratio_list, date_months, s=15, vmin=-3, vmax=3)
    fig_scatterplot_monthly.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_monthly.tiff")
    
    fig_scatterplot_monthly_cb = scatterplot_decomposition_ratio(leakage_months_ratio_list, inconsistency_months_ratio_list, date_months, s=15, vmin=-3, vmax=3, plotcb=True)
    fig_scatterplot_monthly_cb.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_monthly_cb.svg")
    
    fig_scatterplot_daily = scatterplot_decomposition_ratio(leakage_days_ratio_list, inconsistency_days_ratio_list, date_days, vmin=-3, vmax=3, s=5, plot_inconsistency=False)
    fig_scatterplot_daily.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_daily.tiff")
    
    fig_scatterplot_daily_cb = scatterplot_decomposition_ratio(leakage_days_ratio_list, inconsistency_days_ratio_list, date_days, vmin=-3, vmax=3, s=5, plot_inconsistency=False, plotcb=True)
    fig_scatterplot_daily_cb.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_daily_cb.svg")
    
    # plot boxplot ratio
    g_boxplot_monthly = boxplot_decomposition_ratio(leakage_months_ratio_list, inconsistency_months_ratio_list, date_months, vmin=-3, vmax=3)
    g_boxplot_monthly.figure.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_monthly_boxplot.tiff")
    
    g_boxplot_daily = boxplot_decomposition_ratio(leakage_days_ratio_list, inconsistency_days_ratio_list, date_days, vmin=-3, vmax=3)
    g_boxplot_daily.figure.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Ratio_monthly_daily.tiff")
    
    # plot res map
    fig, ax = plot_base_map()
    cmap_res = plt.cm.RdBu
    handle_res = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=res_monthly_mean,
                            cmap=cmap_res,
                            vmin=res_monthly_mean_perc[1], vmax=res_monthly_mean_perc[-2],
                            edgecolor="k", linewidths=0.1)  # , vmax=1, vmin=-0.41
    cmap_res.set_under(cmap_res(653))
    cmap_res.set_over(cmap_res(0))
    
    # cb_res = plt.colorbar(handle_res, location="right", shrink=0.69, pad=0.02)
    # cb_res.ax.tick_params(labelsize=5, direction='in')
    # fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Res_mean_cb.svg")
    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Res_mean.tiff")
    
    
    # plot leakage map
    fig, ax = plot_base_map()
    cmap_leakage = plt.cm.RdBu
    handle_leakage = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=leakage_monthly_mean,
                                cmap=cmap_leakage,
                                vmin=leakage_monthly_mean_perc[1], vmax=leakage_monthly_mean_perc[-2],
                                edgecolor="k", linewidths=0.1)  # , vmax=1, vmin=-0.41
    cmap_leakage.set_under(cmap_leakage(653))
    cmap_leakage.set_over(cmap_leakage(0))
    
    # cb_leakage = plt.colorbar(handle_leakage, location="right", shrink=0.69, pad=0.02)
    # cb_leakage.ax.tick_params(labelsize=5, direction='in')
    # fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Leakage_mean_cb.svg")
    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Leakage_mean.tiff")
    
    # plot inconsistency map
    fig, ax = plot_base_map()
    cmap_inconsistency = plt.cm.RdBu
    handle_inconsistency = ax.scatter(basinLonCen, basinLatCen, marker=None, s=radius, c=inconsistency_monthly_mean,
                                      cmap=cmap_inconsistency,
                                      vmin=inconsistency_monthly_mean_perc[1], vmax=inconsistency_monthly_mean_perc[-2],
                                      edgecolor="k", linewidths=0.1)  # , vmax=1, vmin=-0.41
    cmap_inconsistency.set_under(cmap_inconsistency(653))
    cmap_inconsistency.set_over(cmap_inconsistency(0))
    
    # cb_inconsistency = plt.colorbar(handle_inconsistency, location="right", shrink=0.69, pad=0.02)
    # cb_inconsistency.ax.tick_params(labelsize=5, direction='in')
    # fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Inconsistency_mean_cb.svg")
    fig.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Inconsistency_mean.tiff")
    
    # plot res, leakage, inconsistency boxplot
    g_res_months = boxplot_decomposition(res_months_df, months_color)
    g_res_months.figure.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Res_mean_boxplot.tiff")
    
    g_leakage_months = boxplot_decomposition(leakage_months_df, months_color)
    g_leakage_months.figure.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Leakage_mean_boxplot.tiff")
    
    g_inconsistency_months = boxplot_decomposition(inconsistency_months_df, months_color)       
    g_inconsistency_months.figure.savefig("F:/research/WaterBudgetClosure/Decomposition/decomposition/Inconsistency_mean_boxplot.tiff")
    
    pass
    
    

if __name__ == "__main__":
    root, home = setHomePath(root="F:")
    
    # instance dpc_base
    with open(os.path.join("E:/data/hydrometeorology/CAMELS/dataPreprocess_CAMELS/WaterBalanceAnalysis", "dpc_base.pkl"), "rb") as f:
        dpc_base = pickle.load(f)
    
    # Decomposition_residual_basins
    # Decomposition_residual_basins(cpu=8, input_pet=True)
    
    # read
    # fitness_dict = read_fitness_func(home)
    decomposition_dict = read_decomposition_func(home)
    
    # plot fitness
    # plot_fitness(dpc_base, fitness_dict)
    
    # performance_analysis
    # performance_analysis(dpc_base, fitness_dict)
    
    # plot decomposition
    plot_decomposition(dpc_base, decomposition_dict)
    
    