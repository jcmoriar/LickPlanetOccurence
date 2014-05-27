""" Determine the occurrence rate of planets from the Lick data"""
import thesisio as tio
import numpy as np
import matplotlib.pyplot as plt
import plotting as pl
from astroML.time_series import lomb_scargle
from scipy.signal import find_peaks_cwt
from math import *

class RVTimeSeries(object):
    """ Class to hold and anylize the time series data for one star. """

    def __init__(self, starName):
        """Read in data for star."""
        lickData = tio.LickData()
        data = lickData.GetVelsErrs(starName)
        self.obsTimes = np.array(data[0])
        self.velocities = np.array(data[1])
        self.velocityErrors = np.array(data[2])
        lickData.Close()
        self.freqs = None
        self.periodogram = None
         
    def CalculatePeriodogram(self):
        """Calculate the Lomb-Scargle periodogram."""
        if self.freqs is None:
            deltaFreq = 28
            nFreqs = (self.obsTimes[-1]-self.obsTimes[0])*deltaFreq*4
            nFreqs = 5000
            self.freqs = np.linspace(log(2), log(30*365), nFreqs)
            self.freqs = np.exp(self.freqs)
            self.freqs = np.pi*2/self.freqs
        self.periodogram = lomb_scargle(self.obsTimes, self.velocities,
                                        self.velocityErrors, self.freqs,
                                        generalized=True)

    def FindPeriodogramPeaks(self, nPeaks=10):
        """Find the nPeaks highest peaks in the periodogram"""

        self.peaks = []
        self.peakLocs = []
        remainingPeriodogram = self.periodogram.copy()
        remainingFreqs = self.freqs.copy()
        for i in range(nPeaks):
            maxInd = remainingPeriodogram.argmax()
            maxPower = remainingPeriodogram[maxInd]
            maxFreq = remainingFreqs[maxInd]
            remainingInd = np.where(np.abs(remainingFreqs-maxFreq)/maxFreq > 0.05)
            remainingPeriodogram = remainingPeriodogram[remainingInd]
            remainingFreqs = remainingFreqs[remainingInd]
            self.peaks.append(maxPower)
            self.peakLocs.append(maxFreq)
        #need to use a real peak finding algorithm




        


    def PlotPeriodogram(self, axis=None):
        """Plot the Lomb-Scargle periodogram"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        plt.hold(True)
        for peakLoc in self.peakLocs:
            axis.plot([np.pi*2/peakLoc, np.pi*2/peakLoc], [0, 1e10], color="black", linewidth=0.5)
        axis.plot(np.pi*2./self.freqs, self.periodogram)
        print max(np.pi*2./self.freqs), min(np.pi*2./self.freqs)
        axis.set_xlabel("Period (days)")
        axis.set_ylabel("Power")
        axis.set_xlim([2,1e4])
        axis.set_xscale("log")
        axis.set_ylim([0,max(self.periodogram)*1.1])

        
        
        
    

def main():
    star = RVTimeSeries("9826") 
    star.CalculatePeriodogram()
    star.FindPeriodogramPeaks()
    star.PlotPeriodogram()
