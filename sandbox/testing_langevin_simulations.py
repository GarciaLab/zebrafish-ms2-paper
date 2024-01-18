#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:43:03 2024

@author: brandon
"""

from simulation import Params, simulate
import matplotlib.pyplot as plt
import numpy as np


p = Params()
p = Params()
p.transcription_rate_0 = 10.0
p.transcription_rate_1 = 0.0
p.translation_rate = 5.0
p.mrna_decay_rate = 0.23
p.protein_decay_rate = 0.23
p.noise_strength = 1.0

p.Tmax = 360
p.dt = 0.01
p.max_time_to_next_reaction = 1.0

p.k_on0 = 0.08
p.k_off0 = 0.0

p.k_on1 = 0.0
p.k_off1 = 0.2

p.KD = 800
p.n = 12.0

p.delay = 0

state, mrna, protein, tvec, p = simulate(p)

plt.figure()
plt.plot(tvec, state)