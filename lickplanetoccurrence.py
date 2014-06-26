""" Determine the occurrence rate of planets from the Lick data"""
import thesisio as tio
import numpy as np
import matplotlib.pyplot as plt
import plotting as pl
from astroML.time_series import lomb_scargle
from scipy.signal import find_peaks_cwt
from math import *
from mks import *
from scipy.optimize import leastsq

class RVTimeSeries(object):
    """ Class to hold and anylize the time series data for one star. """

    def __init__(self, starName):
        """Read in data for star."""
        lickData = tio.LickData()
        data = lickData.GetVelsErrs(starName)
        self.starName = starName
        self.obsTimes = np.array(data[0])
        self.velocities = np.array(data[1])
        self.velocityErrors = np.array(data[2])
        lickData.Close()
        self.freqs = None
        self.periodogram = None
        self.residuals = None
        self.peakLocs = None
        self.peaks = None
         
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
        #Find Peaks
        deriv = self.periodogram[1::]-self.periodogram[0:-1]
        peakInd = np.where(((deriv[0:-1] > 0) != (deriv[1::] > 0)) & (deriv[0:-1] > 0))
        peakInd = np.array(peakInd[0]) + 1
        self.peaks = self.periodogram[peakInd]
        self.peakLocs = self.freqs[peakInd]
        #Sort Peaks
        sortInd = self.peaks.argsort()
        self.peaks = self.peaks[sortInd[::-1]]
        self.peakLocs = self.peakLocs[sortInd[::-1]]

    def Plot(self, axis=None, phaseFold=None):
        """Plot the time series data"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        if phaseFold is None:
            plotTimes = self.obsTimes-min(self.obsTimes)
        else:
            plotTimes = np.remainder(self.obsTimes, np.zeros(len(self.obsTimes))+phaseFold)
        axis.errorbar(plotTimes, self.velocities,
                      yerr=self.velocityErrors, fmt="o")
        axis.set_xlabel("Time from first observation (days)")
        axis.set_ylabel("Velocity (m/s)")
        axis.set_title(self.starName)
         
    def PlotErrMag(self, axis=None):
        """Plot the magnitude of the errors over time"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.plot(self.obsTimes-min(self.obsTimes), self.velocityErrors, "o")
        axis.set_xlabel("Time from first observation (days)")
        axis.set_ylabel("Velocity Error (m/s)")
        axis.set_title(self.starName)


    def PlotPeriodogram(self, axis=None):
        """Plot the Lomb-Scargle periodogram"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        plt.hold(True)
        for peakLoc in self.peakLocs[0:10]:
            axis.plot([np.pi*2/peakLoc, np.pi*2/peakLoc], [0, 1e10], color="black", linewidth=0.5)
        axis.plot(np.pi*2./self.freqs, self.periodogram)
        axis.set_xlabel("Period (days)")
        axis.set_ylabel("Power")
        axis.set_xlim([2,1e4])
        axis.set_xscale("log")
        axis.set_ylim([0,max(self.periodogram)*1.1])


    def RandomErrors(self):
        """Return random errors for each data point.

        Description: Calculates the residuals of the best fit sinusoid
        of the velocities (minus signals from planets if they are known
        to exist). Random (normal) errors are then scaled by these 
        residuals and returned.

        """

        if self.residuals is None:
            self.CalculateResiduals()
        errors = np.random.randn(len(self.residuals))
        errors = errors*self.residuals
        return(errors)

    def CalculateResiduals(self):
        """Calcualate residuals to best fitting sinusoid to velocities."""
        if self.periodogram is None:
            self.CalculatePeriodogram()
        if self.peakLocs is None:
            self.FindPeriodogramPeaks()
        fitFunc = lambda p, x: p[0]*np.cos(self.peakLocs[0]*x+p[1]) + p[2]
        errFunc = lambda p, x, y: (fitFunc(p, x) - y)/self.velocityErrors/self.velocityErrors
        p0 = [100.0, 0, 0]
        bestFit, success = leastsq(errFunc, p0[:],
                                   args=(self.obsTimes, self.velocities))
        self.fit = fitFunc(bestFit, self.obsTimes)
        self.bestFit = bestFit
        self.residuals = self.velocities - self.fit

    def PlotResiduals(self, axis=None):
        """Plot time series of the best sinusoid residuals"""

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.plot(self.obsTimes, self.residuals, "o")
        axis.set_xlabel("Time from first observation (days)")
        axis.set_ylabel("Velocity Residuals (m/s)")
        axis.set_title(self.starName)

    def SimulatedData(self, period, semiAmplitude, phase, offset=0):
        """Create siulated velocities with noise.i
        
        Args:
            period: Period in days of signal
            semiAmplitude: Semi-amplitude of signal in m/s
            phase: Phase offset [0-1]

        """
        omega = 2 * PI / period
        phi = phase * PI
        vels = semiAmplitude*np.sin(omega*self.obsTimes+phi)+offset
        return(self.obsTimes, vels + self.RandomErrors())

    def PlotSimulatedData(self, velocities, period=None, semiAmplitude=None, 
                          phase=None, offset=0, axis=None, fold=False):
        """Plot a simulated dataset and optionally the true signal"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        if fold:
            plotTimes = np.remainder(self.obsTimes, np.zeros(len(self.obsTimes))+period)
        else:
            plotTimes = self.obsTimes-min(self.obsTimes)
        axis.plot(plotTimes, velocities, "o")
        if period is not None:
            omega = 2 * PI / period
            phi = phase * PI
            times = np.arange(0, self.obsTimes[-1]-min(self.obsTimes),1.0)
            if fold:
                times = np.remainder(times, np.zeros(len(times))+period)
            axis.plot(times, semiAmplitude*np.sin(omega*times+phi)+offset, "o")

    def MonteCarloSignalPower(self, period, semiAmplitude, nTrials=1000):
        """Calculate periodogram power for nTrials realizations of signal.

        Args:
            period: Period of signal in days.
            semiAmplitude: Semi-amplitude of signal in m/s.
            nTrials: Number of realizations of signal to test power of.
        Return: 
            list(periodogram powers)

        """
        powers = []
        phis = []
        for i in range(nTrials):
            phi = np.random.rand()
            phis.append(phi)
            signal = self.SimulatedData(period, semiAmplitude, phi)
            powers.append(LombScargleSingleFreq(self.obsTimes, signal, 2*PI/period))
        return(powers)

def LombScargleSingleFreq(x, y, f):
    """Calculate Lomb-Scargle periodogram power at a single frequency"""

def main(star):
    star = RVTimeSeries(star) 
    star.CalculatePeriodogram()
    star.FindPeriodogramPeaks()
    star.PlotPeriodogram()
    star.Plot()
    star.PlotErrMag()
    star.CalculateResiduals()
    star.PlotResiduals()
    p = 200
    v = 100
    ph = 0.2
    sim = star.SimulatedData(p, v, ph)
    star.PlotSimulatedData(sim, p, v, ph)



if __name__ == '__main__':
    main(sys.argv[1])
