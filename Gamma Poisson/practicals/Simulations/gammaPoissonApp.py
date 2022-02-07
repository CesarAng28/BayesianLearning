#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:02:10 2022

@author: cesarangeles
"""
import numpy as np
from ConjugatePriorDistributions import GammaPoisson
import matplotlib.pyplot as plt


from scipy.stats import poisson
from scipy.stats import gamma
from typing import Optional

np.random.seed()
data = np.array([50, 65, 72, 63, 70])
data
gp = GammaPoisson(0)

gp.setGamma(1020, 15)

sample = gp.sampleGammaDistribution(10**5)

gp.setPoisson(np.mean(sample), np.var(sample))
postSample = gp.samplePoissonDistribution(10**5)


print(gp.BayesianProbability(80))

print(len(sample))

