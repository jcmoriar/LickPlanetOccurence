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
from astroML.linear_model import BasisFunctionRegression


class RVTimeSeries(object):
    """ Class to hold and anylize the time series data for one star. """

    def __init__(self, starName):
        """Read in data for star."""
        lickData = tio.LickData()
        data = lickData.GetVelsErrs(starName)
        self.starName = starName
        self.obsTimes = np.array(data[0])
        # self.obsTimes = 
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

    def TestLombScargle(self):
        pGram = LombScargleLinear(self.obsTimes, self.velocities,
                                             self.freqs,
                                             err=self.velocityErrors)
        fig = plt.figure() 
        axis = fig.add_subplot(111)
        # axis.semilogx(2*PI/self.freqs, pGram)
def LombScargleSingleFreq(xIn, yIn, f, err=1):
    """Calculate Lomb-Scargle periodogram power at a single frequency"""
    fitFunc = lambda p, x: p[0] * np.cos(f*x) + p[1] * np.sin(f*x) + p[2]
    errFunc = lambda p, x, y: (fitFunc(p, x) - y) / err / err
    p0 = [100.0, 100.0, 0]
    bestFit, success = leastsq(errFunc, p0[:], args=(xIn, yIn))

def LombScargle(xIn, yIn, freqs, err=1):
    """Calculate Lomb-Scargle Periodogram with weighting from cumming 1999.

    Args:
        xIn (numpy.array): times
        yIn (numpy.array): observations
        freqs (numpy.array): frequencies at which to evaluate power
        err (numpy.array): observation errors
    return:
        Lomb-Scargle power at each frequency in freqs

    """
    weights = 1/err/err
    chi2s = np.zeros(len(freqs))
    fitFunc = lambda p, x, f: p[0] * np.cos(f*x) + p[1] * np.sin(f*x) + p[2]
    errFunc = lambda p, x, y, f: (fitFunc(p, x, f) - y) / err
    fit = [100.0, 100.0, 0]
    for i in range(len(freqs)):
        fitPars, success = leastsq(errFunc, fit[:], args=(xIn, yIn, freqs[i]))
        chi2s[i]= errFunc(fitPars, xIn, yIn, freqs[i]).sum()
    mean = np.average(yIn, weights=weights)
    chi2Line = sum(weights*(yIn-mean)**2)
    bestChi2 = min(chi2s)
    # return((len(xIn)-3)/2*(chi2Line-chi2s)/bestChi2)
    return chi2s

def LombScargleLinear(xIn, yIn, freqs, err=1):
    def linearSin(X, freq=1):
        rtn = (np.hstack((np.sin(X*freq), np.cos(X*freq), np.ones((len(X),1)))))
        print rtn.shape
        return(np.hstack((np.sin(X*freq), np.cos(X*freq), np.ones((len(X), 1)))))
    model = BasisFunctionRegression(linearSin, freq=1)
    mu = np.linspace(0, 1, 10)[:, np.newaxis]
    sigma = .1
    # model = BasisFunctionRegression("gaussian", mu=mu, sigma=sigma)
    xIn.shape = (len(xIn),1)
    print xIn.dtype
    # xIn = (xIn - min(xIn))/xIn.ptp()*6
    xIn = (xIn - min(xIn))/10000
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(xIn, linearSin(xIn, freqs[len(freqs)*.8]), "o")
    # xIn = np.linspace(min(xIn), max(xIn),500)[:,None]
    # axis.plot(xIn, linearSin(xIn, freqs[len(freqs)*.8]))
    xIn = np.random.random((53, 1)) # 100 points in 1 # dimension 
    chi2s = np.zeros(len(freqs))
    print xIn
    for i in range(len(freqs)):
        # model.kwargs["freq"]=freqs[0]
        model.fit(xIn, yIn, 1) 
        import pdb; pdb.set_trace()


def linearSin(X, freq=1):
    rtn = (np.hstack((np.sin(X*freq), np.cos(X*freq), np.ones((len(X), 1)))))
    print rtn.shape
    return(np.hstack((np.sin(X*freq), np.cos(X*freq), np.ones((len(X), 1)))))


def main(starName):
    # f = BasisFunctionRegression('gaussian', mu=range(10), sigma=1)
    # x = np.random.random((100,1))*4-2
    # y = f.basis_func(x, mu=np.array([0,1])[:,None], sigma=1)
    # y2 = linearSin(x)
    # fig = plt.figure()
    # axis = fig.add_subplot(111)
    # axis.plot(x,y2)
    import numpy as np
    from astroML.linear_model import BasisFunctionRegression
    X = np.random.random((100, 1))  # 100 points in 1 # dimension
    dy = 0.1
    y = np.random.normal(X[:, 0], dy)
    mu = np.linspace(0, 1, 10)[:, np.newaxis]
    # 10 x 1 array of mu 
    sigma = 0.1
    model = BasisFunctionRegression('gaussian', mu=mu, sigma=sigma)
    model.fit(X, y, dy)
    print "book: ", type(X), X.shape
    print "book: ", type(y), y.shape
    # print type(dy), dy.shape
    y_pred = model.predict(X)
    # import pdb; pdb.set_trace()
    star = RVTimeSeries(starName) 
    star.CalculatePeriodogram()
    star.FindPeriodogramPeaks()
    star.PlotPeriodogram()
    # star.Plot()
    # star.PlotErrMag()
    # star.CalculateResiduals()
    # star.PlotResiduals()
    # p = 200
    # v = 100
    # ph = 0.2
    # sim = star.SimulatedData(p, v, ph)
    # star.PlotSimulatedData(sim, p, v, ph)
    star.TestLombScargle()
    


if __name__ == '__main__':
    main(sys.argv[1])
