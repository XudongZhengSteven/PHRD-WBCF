# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from typing import Any
import lumod
from lumod.models import HBV
import numpy as np
import pandas as pd


default_params = {
    "maxbas": 3,
    "tthres": 5.0,
    "dd": 2.0,
    "beta": 2.0,
    "fc": 500.0,
    "pwp": 0.8,
    "k0": 0.5,
    "k1": 0.1,
    "k2": 0.01,
    "kp": 0.05,
    "lthres": 50,
}


class Lumod_HBV:
    
    def __init__(self, area, lat, **params) -> None:
        self._area = area
        self._lat = lat
        self._params = default_params
        self._params.update(params)
        self.model = HBV(area=self._area, lat=self._lat, params=self._params)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    def warm_up(self, forcings, start, end, **kwargs):
        self.model.run(forcings, start=start, end=end, save_state=True, **kwargs)
    
    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        sim = self.model.run(forcings, start=start, end=end, save_state=save_state, **kwargs)
        return sim
    

if __name__ == "__main__":
    hbv = Lumod_HBV(1000, 30, fc=300)
    hbv()