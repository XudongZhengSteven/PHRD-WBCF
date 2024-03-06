# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime
import copy

def ExtractForcingDaymet(forcingDaymet, forcingDaymetGaugeAttributes, gauge_id, period, plot=False):
    """_summary_

    Args:
        forcingDaymet (_type_): _description_
        forcingDaymetGaugeAttributes (_type_): _description_
        gauge_id (_type_): _description_
        extract_dates (_type_): _description_, #!note should be set to avoid missing value
            e.g. extract_dates = pd.date_range("19800101", "20141231", freq="D")
        plot (bool, optional): _description_. Defaults to False.
    """
    # read as df
    forcingDaymet_gauge_id = np.array([s["gauge_id"] for s in forcingDaymetGaugeAttributes])
    gauge_index = np.where(forcingDaymet_gauge_id == gauge_id)
    forcingDaymet_gauge = copy.copy(forcingDaymet[gauge_index[0][0]])
    
    # create datetimeIndex
    dates = list(map(lambda i: datetime(*i), zip(forcingDaymet_gauge.loc[:, 'Year'], forcingDaymet_gauge.loc[:, 'Mnth'],
                                                 forcingDaymet_gauge.loc[:, 'Day'])))
    dates = pd.to_datetime(dates)
    forcingDaymet_gauge.index = dates
    date_period_index_bool = (dates >= datetime.strptime(period[0], "%Y%m%d")) & (dates <= datetime.strptime(period[1], "%Y%m%d"))

    # extract
    try:
        prcp = forcingDaymet_gauge.loc[date_period_index_bool, "prcp(mm/day)"]
        srad = forcingDaymet_gauge.loc[date_period_index_bool, "srad(W/m2)"]
        swe = forcingDaymet_gauge.loc[date_period_index_bool, "swe(mm)"]
        tmax = forcingDaymet_gauge.loc[date_period_index_bool, "tmax(C)"]
        tmin = forcingDaymet_gauge.loc[date_period_index_bool, "tmin(C)"]
        vp = forcingDaymet_gauge.loc[date_period_index_bool, "vp(Pa)"]
        
        forcingDaymet_gauge_set = {"prcp(mm/day)": prcp,
                                "srad(W/m2)": srad,
                                "swe(mm)": swe,
                                "tmax(C)": tmax,
                                "tmin(C)": tmin,
                                "vp(Pa)": vp}
    
        # plot
        fig_all = []
        if plot:
            for k in forcingDaymet_gauge_set:
                fig, ax = plt.subplots()
                forcingDaymet_gauge_set[k].plot(ax=ax, label=k)
                plt.legend()
                fig_all.append(fig)
                
    except ValueError(f"extract_period not suitable for gauge_id: {gauge_id}, return None"):
        forcingDaymet_gauge_set = None
        fig_all = None
            
    return forcingDaymet_gauge_set, fig_all
    
def formatForcing():
    pass
    
def Mete_param_Preparation():
    pass
    
    
    