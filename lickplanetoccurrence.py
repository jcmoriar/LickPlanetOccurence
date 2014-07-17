""" Determine the occurrence rate of planets from the Lick data"""
import thesisio as tio
import numpy as np
import matplotlib.pyplot as plt
from astroML.time_series import lomb_scargle
from math import *
from mks import *
from scipy.optimize import leastsq
from scipy.interpolate import griddata
from sklearn.linear_model import Lasso
from astroML.linear_model import NadarayaWatson
from sklearn.gaussian_process import GaussianProcess


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
        self.gridAmplitudes = None
        self.gridFractionDetectable = None
        self.gridPeriods = None

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
        self.periodogram = (self.periodogram / (1-self.periodogram.max())
                            * (len(self.velocities) - 3) / 2)

    def FindPeriodogramPeaks(self, nPeaks=10):
        """Find the nPeaks highest peaks in the periodogram"""
        #Find Peaks
        deriv = self.periodogram[1::]-self.periodogram[0:-1]
        peakInd = np.where(((deriv[0:-1] > 0) != (deriv[1::] > 0))
                           & (deriv[0:-1] > 0))
        peakInd = np.array(peakInd[0]) + 1
        self.peaks = self.periodogram[peakInd]
        self.peakLocs = self.freqs[peakInd]
        #Sort Peaks
        sortInd = self.peaks.argsort()
        self.peaks = self.peaks[sortInd[::-1]][0:nPeaks]
        self.peakLocs = self.peakLocs[sortInd[::-1]][0:nPeaks]

    def Plot(self, axis=None, phaseFold=None):
        """Plot the time series data"""
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        if phaseFold is None:
            plotTimes = self.obsTimes-min(self.obsTimes)
        else:
            plotTimes = np.remainder(self.obsTimes,
                                     np.zeros(len(self.obsTimes))+phaseFold)
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
            axis.plot([np.pi*2/peakLoc, np.pi*2/peakLoc], [0, 1e10],
                      color="black", linewidth=0.5)
        axis.plot(np.pi*2./self.freqs, self.periodogram)
        axis.set_xlabel("Period (days)")
        axis.set_ylabel("Power")
        axis.set_xlim([2, 1e4])
        axis.set_xscale("log")
        axis.set_ylim([0, max(self.periodogram)*1.1])

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
        errFunc = lambda p, x, y: ((fitFunc(p, x) - y) / self.velocityErrors
                                   / self.velocityErrors)
        p0 = [100.0, 0, 0]
        bestFit, success = leastsq(errFunc, p0[:],
                                   args=(self.obsTimes, self.velocities))
        self.bestFitSinusoid = fitFunc(bestFit, self.obsTimes)
        self.residuals = self.velocities - self.bestFitSinusoid

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
            plotTimes = np.remainder(self.obsTimes,
                                     np.zeros(len(self.obsTimes))+period)
        else:
            plotTimes = self.obsTimes-min(self.obsTimes)
        axis.plot(plotTimes, velocities, "o")
        if period is not None:
            omega = 2 * PI / period
            phi = phase * PI
            times = np.arange(0, self.obsTimes[-1]-min(self.obsTimes), 1.0)
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
            signal = self.SimulatedData(period, semiAmplitude, phi)[1]
            error = self.residuals
            error = 1
            powers.append(LombScargleSingleFreq(self.obsTimes, signal,
                                                2*PI/period, err=error))
        return(powers)

    def DetectabilityGrid(self, periods, nAmplitudes=100, nTrials=1000):
        """Calculate detectabilites for signals in a period/amplitude grid.

        Description:

        Args:
            periods: The periods for the grid.
            nAmplitudes: The number of amplitudes to test at each period.

        """
        amplitudes = []
        fractionDetectable = []
        periodsOut = []
        for i in range(len(periods)):
            print i
            maxSA = 100
            minSA = 0
            thisSA = np.random.random() * maxSA
            for j in range(nAmplitudes):
                powers = self.MonteCarloSignalPower(periods[i], thisSA,
                                                    nTrials=nTrials)
                frac = float(sum(powers > self.peaks[0])) / nTrials
                amplitudes.append(thisSA)
                fractionDetectable.append(frac)
                periodsOut.append(periods[i])
                if frac > 0.99999:
                    maxSA = thisSA
                elif frac < 0.00001:
                    minSA = thisSA
                thisSA = np.random.random() * (maxSA - minSA) + minSA
        self.gridAmplitudes = np.array(amplitudes)
        self.gridDetectability = np.array(fractionDetectable)
        self.gridPeriods = np.array(periodsOut)
        return(self.gridPeriods, self.gridAmplitudes, self.gridDetectability)

    def PlotDetectabilityGrid(self, axis=None, data=None):
        if data is not None:
            self.gridPeriods = data[0]
            self.gridAmplitudes = data[1]
            self.gridDetectability = data[2]
        if self.gridAmplitudes is None:
            self.DetectabilityGrid(np.linspace(1, 1000, 20))
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        # periods = np.linspace(np.log10(self.gridPeriods.min()),
        #                       np.log10(self.gridPeriods.max()), 100)
        periods = list(set(np.log10(self.gridPeriods)))
        periods.sort()
        periods = np.array(periods)
        semiAmp = np.linspace(0, 100, 100)
        pIn = np.hstack((self.gridPeriods,
                        [self.gridPeriods.min(), self.gridPeriods.min(),
                         self.gridPeriods.max(), self.gridPeriods.max()]))
        periodsIn = np.log10(pIn)
        amplitudeIn = np.hstack((self.gridAmplitudes, [0, 100, 0, 100]))
        detectabilityIn = np.hstack((self.gridDetectability, [0, 1, 0, 1]))
        detectableFraction = griddata((periodsIn, amplitudeIn),
                                      detectabilityIn,
                                      (periods[None, :], semiAmp[:, None]))
        detectableFraction = GridDetectibilites(periodsIn, amplitudeIn, detectabilityIn, semiAmp)
        axis.matshow(detectableFraction, origin="lower", extent=[min(periods),
                     max(periods), min(semiAmp), max(semiAmp)],
                     aspect=(periods.ptp()/semiAmp.ptp()))
        axis.set_ylabel("Semi-amplitude")
        axis.set_xlabel("Period (Days)")
        tickMax = int(floor(periods.max()))
        tickMin = int(ceil(periods.min()))
        tickLocs = range(tickMin, tickMax+1)
        labels = [str(10**x) for x in tickLocs]
        axis.set_xticks(tickLocs)
        axis.set_xticklabels(labels)


def GridDetectibilites(period, amps, detectability, newAmps):
    """Put the detectability info onto a regular grid.

    Args:
        period: Periods for each data point.
        amps: Semi-amplitudes for each datapoint.
        detectability: Detected fraction for each datapoint.
    Return:
        2-D numpy array of detectability gridded onto period by newAmps.

    """
    periods = list(set(period))
    periods.sort()
    grid = np.zeros((len(newAmps), len(periods)))
    for i, p in enumerate(periods):
        # Select data for just this period and add endpoints
        ind = np.where(period == p)
        ampsThisP = np.hstack((0, amps[ind], 100))
        detectThisP = np.hstack((0, detectability[ind], 1))
        # Fix the shapes of arrays so they can be used in fitter
        ampsThisP.shape = (len(ampsThisP), 1)
        detectThisP.shape = (len(detectThisP),)
        newAmps.shape = (len(newAmps), 1)
        # Fit with Gaussian Kernel Regression
        model = NadarayaWatson('gaussian', h=1)
        model.fit(ampsThisP, detectThisP)
        gridDetectability = model.predict(newAmps)
        #Make sure everything is between limits
        gridDetectability[np.where(gridDetectability < 1e-3)] = 0
        gridDetectability[np.where(gridDetectability > 0.999)] = 1
        grid[:, i] = gridDetectability
    return grid


def LombScargleSingleFreq(xIn, yIn, f, err=1):
    """Calculate Lomb-Scargle periodogram power at a single frequency weighted
    as in cumming 1999

    """
    power = lomb_scargle(xIn, yIn, err, [f, .005], generalized=True)[0]
    power = power / (1 - power) * (len(xIn) - 3) / 2
    return power


def main(data=None, starName="98697"):
    star = RVTimeSeries(starName)
    star.CalculatePeriodogram()
    star.FindPeriodogramPeaks()
    # star.Plot()
    if data is None:
        periods = np.power(10, np.linspace(np.log10(1), np.log10(10000), 300))
        data = star.DetectabilityGrid(periods)
    star.PlotDetectabilityGrid(data=data)
    # star.PlotErrMag()
    # star.CalculateResiduals()
    # star.PlotResiduals()
    # p = 200
    # v = 100
    # ph = 0.2
    # sim = star.SimulatedData(p, v, ph)
    # star.PlotSimulatedData(sim, p, v, ph)
    # fig = plt.figure()
    # axis = fig.add_subplot(111)
    # axis.plot(data[1], data[2], "ro")
    return(data)


if __name__ == '__main__':
    main(sys.argv[1])
