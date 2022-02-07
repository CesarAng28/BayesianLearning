#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:05:51 2022

@author: cesarangeles
"""

import numpy as np 
from scipy.stats import poisson
from scipy.stats import gamma
from typing import Optional

"""
 Simulate mu* from its posterior Gamma distribution 

 Simulate x_p from the data generating mechanism

"""

class GammaPoisson:
    
    def __init__(self, n_sim: int = 0) -> None:
        """
        

        Parameters
        ----------
        n_sim : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None
            DESCRIPTION.
            
        Examples
        --------
        
        >>> import numpy as np
        >>> from ConjugatePriorDistributions import GammaPoisson
        >>> import matplotlib.pyplot as plt


        >>> np.random.seed(123)

        >>> gp = GammaPoisson(0)

        >>> gp.setGamma(1020, 15)

        """
        self.n_sim = n_sim
        self.iter_number = 0

        self.gamma_alpha = []
        self.gamma_beta = []
        
        self.poisson_intensity_mean = []
        self.poisson_intensity_var = []
        
        self.gammaSample = np.array([])
        
    def cleanArray(self, array: np.ndarray) -> np.ndarray :
        return np.delete(array, 0)
        
        
    def setGamma(self, alpha: float, beta: float) -> None:
        """
        

        Parameters
        ----------
        alpha : float
            DESCRIPTION.
        beta : float
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        self.gamma_alpha.append(alpha)
        self.gamma_beta.append(beta)
        print("Setting Gamma Parameter\nAlpha = {} Beta = {}".format(*self.getGammaParams()))
        
        
        
    def setPoisson(self, intensity_exp: float, variance: float) -> None:
        """
        

        Parameters
        ----------
        intensity_exp : float
            DESCRIPTION.
        variance : float
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        self.poisson_intensity_mean.append(intensity_exp)
        self.poisson_intensity_var.append(variance)
        
        
    def getGammaParams(self) -> tuple[float, float]:
        """
        

        Returns
        -------
        float
            DESCRIPTION.

        """

        return self.gamma_alpha[self.iter_number], self.gamma_beta[self.iter_number]
        
        
    def posteriorGammaStatistics(self, data: type(np.array([]))) -> float:
        
        alpha = self.gamma_alpha[self.iter_number] + np.sum(data)
        beta = self.gamma_beta[self.iter_number] + len(data)
        
        print("Posterior")
        return alpha, beta
        
        
    def posteriorPoissonStatistics(self) -> tuple:
        
        alpha, beta = self.getGammaParams()
        poissonExpectationUpdate = alpha/beta
        poissonVarianceUpdate = alpha/beta**2
        print("Updating Expected Value with™")
        return poissonExpectationUpdate, poissonVarianceUpdate
    
    def sampleGammaDistribution(self, n_samples: int) -> np.ndarray:
        print('Sampling Gamma distribution using values:')
        print('alpha: {}\nbeta: {}\n'.format(*self.getGammaParams()))
        print(n_samples)
        self.gammaSample = gamma.rvs(self.gamma_alpha[self.iter_number],
                          scale=1/self.gamma_beta[self.iter_number],
                          size = n_samples)
        
        return self.gammaSample
    
    
    
    
    def updateBelief(self, data: np.ndarray) -> tuple:
        
        alpha_pos, beta_pos = self.posteriorGammaStatistics(data)
        self.setGamma(alpha_pos, beta_pos)
        new_mean, new_var = self.posteriorPoissonStatistics()
        self.setPoisson(new_mean, new_var)
        self.iter_number =+1
        return alpha_pos, beta_pos, new_mean, new_var
    
    
    def samplePoissonDistribution(self, n_samples: int) -> np.ndarray:
        """
        

        Parameters
        ----------
        n_samples : int
            DESCRIPTION.

        Returns
        -------
        np.ndarray
            

        """
        iteration = self.iter_number
        poissonIntensity = self.poisson_intensity_mean[iteration]
        print("sampling x w poisson intensity {}".format(poissonIntensity))
        return poisson.rvs(poissonIntensity, size = n_samples)
    

    def BayesianPvalue(self, quantile : float = .95, sample_size : int = 1000) -> float:
        """
        

        Parameters
        ----------
        sample_size : int, optional
            DESCRIPTION. The default is 1000.
        certainty : float, optional
            DESCRIPTION. The default is 0.95.
        quantile : Optional[float], optional
            DESCRIPTION. The default is 0.95.

        Returns
        -------
        float
            DESCRIPTION.
            
        Examples
        --------
        >>> import numpy as np 
        >>> from scipy.stats import poisson
        >>> from scipy.stats import gamma
        >>> from typing import Optional
        >>>
        >>> data = poisson.rvs(0.6, size = 1000)
        >>>
        >>> 
            
        

        """
        sample = self.posteriorPredictiveDistribution(sample_size)
        certainty_condition = np.quantile(sample, quantile)
        return certainty_condition
    
    def BayesianProbability(self, condition : float, n_samples : int = 10000) -> float:
        """
        

        Parameters
        ----------
        n_samples : int, optional
            DESCRIPTION. The default is 10000.

        Returns
        -------
        float
            DESCRIPTION.

        """
        
        sample = self.samplePoissonDistribution(n_samples)
        print("mean {}".format(np.mean(sample)))
        print("var {}".format(np.var(sample)))
        probability = np.mean(sample>=condition)
        
        return probability
        
        
        
        
        
        
        
        
        
        
        
        
    
    
        
     