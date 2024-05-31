#!/usr/bin/env python
# encoding: utf-8
"""
airfoilprep.py

Created by Andrew Ning on 2012-04-16.
Copyright (c) NREL. All rights reserved.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from __future__ import print_function
from math import pi, sin, cos, radians, degrees, tan, ceil, floor, factorial, sqrt, atan
import numpy as np
import cmath, mpmath
from scipy.interpolate import RectBivariateSpline, bisplev
import sys, copy, csv, subprocess, warnings, os
from io import open
try:
    import pyXLIGHT
except:
    pass #warnings.warn('XFOIL not configured')

class Polar(object):
    """
    Defines section lift, drag, and pitching moment coefficients as a
    function of angle of attack at a particular Reynolds number.

    """

    def __init__(self, Re, alpha, cl, cd, cm):
        """Constructor

        Parameters
        ----------
        Re : float
            Reynolds number
        alpha : ndarray (deg)
            angle of attack
        cl : ndarray
            lift coefficient
        cd : ndarray
            drag coefficient
        cm : ndarray
            moment coefficient
        """

        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.array(cm)

    def blend(self, other, weight):
        """Blend this polar with another one with the specified weighting

        Parameters
        ----------
        other : Polar
            another Polar object to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        polar : Polar
            a blended Polar

        """

        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other.alpha)
        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other.alpha.min())
        max_alpha = min(self.alpha.max(), other.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]
        # alpha = np.array([a for a in alpha if a >= min_alpha and a <= max_alpha])

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other.alpha, other.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other.alpha, other.cd)
        cm1 = np.interp(alpha, self.alpha, self.cm)
        cm2 = np.interp(alpha, other.alpha, other.cm)

        # linearly blend
        Re = self.Re + weight*(other.Re-self.Re)
        cl = cl1 + weight*(cl2-cl1)
        cd = cd1 + weight*(cd2-cd1)
        cm = cm1 + weight*(cm2-cm1)

        return type(self)(Re, alpha, cl, cd, cm)



    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """Applies 3-D corrections for rotating sections from the 2-D data.

        Parameters
        ----------
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----
        The Du-Selig method :cite:`Du1998A-3-D-stall-del` is used to correct lift, and
        the Eggers method :cite:`Eggers-Jr2003An-assessment-o` is used to correct drag.


        """
        if all(elem == self.cd[0] for elem in self.cd) or tsr == 0.0:
            return type(self)(self.Re, self.alpha, self.cl, self.cd, self.cm)
        else:
            # rename and convert units for convenience
            alpha = np.radians(self.alpha)
            cl_2d = self.cl
            cd_2d = self.cd
            alpha_max_corr = radians(alpha_max_corr)
            alpha_linear_min = radians(alpha_linear_min)
            alpha_linear_max = radians(alpha_linear_max)

            # parameters in Du-Selig model
            a = 1
            b = 1
            d = 1
            if tsr==None:
                lam = 1
            else:
                lam = tsr/(1+tsr**2)**0.5  # modified tip speed ratio
            expon = d/lam/r_over_R

            # find linear region
            idx = np.logical_and(alpha >= alpha_linear_min,
                                alpha <= alpha_linear_max)
            p = np.polyfit(alpha[idx], cl_2d[idx], 1)
            m = p[0]
            alpha0 = -p[1]/m

            # correction factor
            fcl = 1.0/m*(1.6*chord_over_r/0.1267*(a-chord_over_r**expon)/(b+chord_over_r**expon)-1)

            # not sure where this adjustment comes from (besides AirfoilPrep spreadsheet of course)
            adj = ((pi/2-alpha)/(pi/2-alpha_max_corr))**2
            adj[alpha <= alpha_max_corr] = 1.0

            # Du-Selig correction for lift
            cl_linear = m*(alpha-alpha0)
            cl_3d = cl_2d + fcl*(cl_linear-cl_2d)*adj

            # Eggers 2003 correction for drag
            delta_cl = cl_3d-cl_2d

            delta_cd = delta_cl*(np.sin(alpha) - 0.12*np.cos(alpha))/(np.cos(alpha) + 0.12*np.sin(alpha))
            cd_3d = cd_2d + delta_cd

            return type(self)(self.Re, np.degrees(alpha), cl_3d, cd_3d, self.cm)



    def extrapolate(self, cdmax, AR=None, cdmin=0.001, nalpha=15):
        """Extrapolates force coefficients up to +/- 180 degrees using Viterna's method
        :cite:`Viterna1982Theoretical-and`.

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            aspect ratio = (rotor radius / chord_75% radius)
            if provided, cdmax is computed from AR
        cdmin: float, optional
            minimum drag coefficient.  used to prevent negative values that can sometimes occur
            with this extrapolation method
        nalpha: int, optional
            number of points to add in each segment of Viterna method

        Returns
        -------
        polar : Polar
            a new Polar object

        Notes
        -----
        If the current polar already supplies data beyond 90 degrees then
        this method cannot be used in its current form and will just return itself.

        If AR is provided, then the maximum drag coefficient is estimated as

        cdmax = 1.11 + 0.018*AR

        """
        if all(elem == self.cd[0] for elem in self.cd):
            alpha = np.linspace(-pi, pi, 8*nalpha)
            cd = np.ones(8*nalpha,)*self.cd[0]
            cl = np.ones(8*nalpha,)*self.cl[0]
            cm = np.ones(8*nalpha,)*self.cm[0]
            return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)
        else:
            if cdmin < 0:
                raise Exception('cdmin cannot be < 0')

            # lift coefficient adjustment to account for assymetry
            cl_adj = 0.7

            # estimate CD max
            if AR is not None:
                cdmax = 1.11 + 0.018*AR
            self.cdmax = max(max(self.cd), cdmax)

            # extract matching info from ends
            alpha_high = radians(self.alpha[-1])
            cl_high = self.cl[-1]
            cd_high = self.cd[-1]
            cm_high = self.cm[-1]

            alpha_low = radians(self.alpha[0])
            cl_low = self.cl[0]
            cd_low = self.cd[0]

            if alpha_high > pi/2:
                raise Exception('alpha[-1] > pi/2')
                return self
            if alpha_low < -pi/2:
                raise Exception('alpha[0] < -pi/2')
                return self

            # parameters used in model
            sa = sin(alpha_high)
            ca = cos(alpha_high)
            self.A = (cl_high - self.cdmax*sa*ca)*sa/ca**2
            self.B = (cd_high - self.cdmax*sa*sa)/ca

            # alpha_high <-> 90
            alpha1 = np.linspace(alpha_high, pi/2, nalpha)
            alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
            cl1, cd1 = self.__Viterna(alpha1, 1.0)

            # 90 <-> 180-alpha_high
            alpha2 = np.linspace(pi/2, pi-alpha_high, nalpha)
            alpha2 = alpha2[1:]
            cl2, cd2 = self.__Viterna(pi-alpha2, -cl_adj)

            # 180-alpha_high <-> 180
            alpha3 = np.linspace(pi-alpha_high, pi, nalpha)
            alpha3 = alpha3[1:]
            cl3, cd3 = self.__Viterna(pi-alpha3, 1.0)
            cl3 = (alpha3-pi)/alpha_high*cl_high*cl_adj  # override with linear variation

            if alpha_low <= -alpha_high:
                alpha4 = []
                cl4 = []
                cd4 = []
                alpha5max = alpha_low
            else:
                # -alpha_high <-> alpha_low
                # Note: this is done slightly differently than AirfoilPrep for better continuity
                alpha4 = np.linspace(-alpha_high, alpha_low, nalpha)
                alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
                cl4 = -cl_high*cl_adj + (alpha4+alpha_high)/(alpha_low+alpha_high)*(cl_low+cl_high*cl_adj)
                cd4 = cd_low + (alpha4-alpha_low)/(-alpha_high-alpha_low)*(cd_high-cd_low)
                alpha5max = -alpha_high

            # -90 <-> -alpha_high
            alpha5 = np.linspace(-pi/2, alpha5max, nalpha)
            alpha5 = alpha5[1:]
            cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

            # -180+alpha_high <-> -90
            alpha6 = np.linspace(-pi+alpha_high, -pi/2, nalpha)
            alpha6 = alpha6[1:]
            cl6, cd6 = self.__Viterna(alpha6+pi, cl_adj)

            # -180 <-> -180 + alpha_high
            alpha7 = np.linspace(-pi, -pi+alpha_high, nalpha)
            cl7, cd7 = self.__Viterna(alpha7+pi, 1.0)
            cl7 = (alpha7+pi)/alpha_high*cl_high*cl_adj  # linear variation

            alpha = np.concatenate((alpha7, alpha6, alpha5, alpha4, np.radians(self.alpha), alpha1, alpha2, alpha3))
            cl = np.concatenate((cl7, cl6, cl5, cl4, self.cl, cl1, cl2, cl3))
            cd = np.concatenate((cd7, cd6, cd5, cd4, self.cd, cd1, cd2, cd3))

            cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients


            # Setup alpha and cm to be used in extrapolation
            cm1_alpha = floor(self.alpha[0] / 10.0) * 10.0
            cm2_alpha = ceil(self.alpha[-1] / 10.0) * 10.0
            alpha_num = abs(int((-180.0-cm1_alpha)/10.0 - 1))
            alpha_cm1 = np.linspace(-180.0, cm1_alpha, alpha_num)
            alpha_cm2 = np.linspace(cm2_alpha, 180.0, int((180.0-cm2_alpha)/10.0 + 1))
            alpha_cm = np.concatenate((alpha_cm1, self.alpha, alpha_cm2))  # Specific alpha values are needed for cm function to work
            cm1 = np.zeros(len(alpha_cm1))
            cm2 = np.zeros(len(alpha_cm2))
            cm_ext = np.concatenate((cm1, self.cm, cm2))
            if np.count_nonzero(self.cm) > 0:
                cmCoef = self.__CMCoeff(cl_high, cd_high, cm_high)  # get cm coefficient
                cl_cm = np.interp(alpha_cm, np.degrees(alpha), cl)  # get cl for applicable alphas
                cd_cm = np.interp(alpha_cm, np.degrees(alpha), cd)  # get cd for applicable alphas
                alpha_low_deg = self.alpha[0]
                alpha_high_deg = self.alpha[-1]
                for i in list(range(len(alpha_cm))):
                    cm_new = self.__getCM(i, cmCoef, alpha_cm, cl_cm, cd_cm, alpha_low_deg, alpha_high_deg)
                    if cm_new is None:
                        pass  # For when it reaches the range of cm's that the user provides
                    else:
                        cm_ext[i] = cm_new
            try:
                cm = np.interp(np.degrees(alpha), alpha_cm, cm_ext)
            except:
                cm = np.zeros(len(cl))
            return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)




    def __Viterna(self, alpha, cl_adj):
        """private method to perform Viterna extrapolation"""

        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax/2*np.sin(2*alpha) + self.A*np.cos(alpha)**2/np.sin(alpha)
        cl = cl*cl_adj

        cd = self.cdmax*np.sin(alpha)**2 + self.B*np.cos(alpha)

        return cl, cd

    def __CMCoeff(self, cl_high, cd_high, cm_high):
        """private method to obtain CM0 and CMCoeff"""

        found_zero_lift = False

        for i in list(range(len(self.cm))):
            try:
                if abs(self.alpha[i]) < 20.0 and self.cl[i] <= 0 and self.cl[i+1] >= 0:
                    p = -self.cl[i] / (self.cl[i + 1] - self.cl[i])
                    cm0 = self.cm[i] + p * (self.cm[i+1] - self.cm[i])
                    found_zero_lift = True
                    break
            except:
                pass # if not found_zero_lift
        if not found_zero_lift:
            p = -self.cl[0] / (self.cl[1] - self.cl[0])
            cm0 = self.cm[0] + p * (self.cm[1] - self.cm[0])
        self.cm0 = cm0
        alpha_high = radians(self.alpha[-1])
        XM = (-cm_high + cm0) / (cl_high * cos(alpha_high) + cd_high * sin(alpha_high))
        cmCoef = (XM - 0.25) / tan((alpha_high - pi/2))
        return cmCoef

    def __getCM(self, i, cmCoef, alpha, cl_ext, cd_ext, alpha_low_deg, alpha_high_deg):
        """private method to extrapolate Cm"""

        cm_new = 0
        if alpha[i] >= alpha_low_deg and alpha[i] <= alpha_high_deg:
            return
        if alpha[i] > -165 and alpha[i] < 165:
            if abs(alpha[i]) < 0.01:
                cm_new = self.cm0
            else:
                if alpha[i] > 0:
                    x = cmCoef * tan(radians(alpha[i]) - pi/2) + 0.25
                    cm_new = self.cm0 - x * (cl_ext[i] * cos(radians(alpha[i])) + cd_ext[i] * sin(radians(alpha[i])))
                else:
                    x = cmCoef * tan(-radians(alpha[i]) - pi/2) + 0.25
                    cm_new = -(self.cm0 - x * (-cl_ext[i] * cos(-radians(alpha[i])) + cd_ext[i] * sin(-radians(alpha[i]))))
        else:
            if alpha[i] == 165:
                cm_new = -0.4
            elif alpha[i] == 170:
                cm_new = -0.5
            elif alpha[i] == 175:
                cm_new = -0.25
            elif alpha[i] == 180:
                cm_new = 0
            elif alpha[i] == -165:
                cm_new = 0.35
            elif alpha[i] == -170:
                cm_new = 0.4
            elif alpha[i] == -175:
                cm_new = 0.2
            elif alpha[i] == -180:
                cm_new = 0
            else:
                print("Angle encountered for which there is no CM table value "
                      "(near +/-180 deg). Program will stop.")
        return cm_new

    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """compute unsteady aero parameters used in AeroDyn input file

        Parameters
        ----------
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        aerodynParam : tuple of floats
            (control setting, stall angle, alpha for 0 cn, cn slope,
            cn at stall+, cn at stall-, alpha for min CD, min(CD))

        """

        alpha = np.radians(self.alpha)
        cl = self.cl
        cd = self.cd

        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        cn = cl*np.cos(alpha) + cd*np.sin(alpha)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)

        # checks for inppropriate data (like cylinders)
        if len(idx) < 10 or len(np.unique(cl)) < 10:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # linear fit
        p = np.polyfit(alpha[idx], cn[idx], 1)
        m = p[0]
        alpha0 = -p[1]/m

        # find cn at stall locations
        alphaUpper = np.radians(np.arange(40.0))
        alphaLower = np.radians(np.arange(5.0, -40.0, -1))
        cnUpper = np.interp(alphaUpper, alpha, cn)
        cnLower = np.interp(alphaLower, alpha, cn)
        cnLinearUpper = m*(alphaUpper - alpha0)
        cnLinearLower = m*(alphaLower - alpha0)
        deviation = 0.05  # threshold for cl in detecting stall

        alphaU = np.interp(deviation, cnLinearUpper-cnUpper, alphaUpper)
        alphaL = np.interp(deviation, cnLower-cnLinearLower, alphaLower)

        # compute cn at stall according to linear fit
        cnStallUpper = m*(alphaU-alpha0)
        cnStallLower = m*(alphaL-alpha0)

        # find min cd
        minIdx = cd.argmin()

        # return: control setting, stall angle, alpha for 0 cn, cn slope,
        #         cn at stall+, cn at stall-, alpha for min CD, min(CD)
        return (0.0, degrees(alphaU), degrees(alpha0), m,
                cnStallUpper, cnStallLower, alpha[minIdx], cd[minIdx])

    def plot(self):
        """plot cl/cd/cm polar

        Returns
        -------
        figs : list of figure handles

        """
        import matplotlib.pyplot as plt

        p = self

        figs = []

        # plot cl
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        plt.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('lift coefficient')
        ax.legend(loc='best')

        # plot cd
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('drag coefficient')
        ax.legend(loc='best')

        # plot cm
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('moment coefficient')
        ax.legend(loc='best')

        return figs

class Airfoil(object):
    """A collection of Polar objects at different Reynolds numbers

    """

    def __init__(self, polars):
        """Constructor

        Parameters
        ----------
        polars : list(Polar)
            list of Polar objects

        """

        # sort by Reynolds number
        self.polars = sorted(polars, key=lambda p: p.Re)

        # save type of polar we are using
        self.polar_type = polars[0].__class__

    @classmethod
    def initFromAerodynFile(cls, aerodynFile, polarType=Polar):
        """Construct Airfoil object from AeroDyn file

        Parameters
        ----------
        aerodynFile : str
            path/name of a properly formatted Aerodyn file

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        # open aerodyn file
        f = open(aerodynFile, 'r')

        # skip through header
        f.readline()
        description = f.readline().rstrip()  # remove newline
        f.readline()
        numTables = int(f.readline().split()[0])

        # loop through tables
        for i in list(range(numTables)):

            # read Reynolds number
            Re = float(f.readline().split()[0])*1e6

            # read Aerodyn parameters
            param = [0]*8
            for j in list(range(8)):
                param[j] = float(f.readline().split()[0])

            alpha = []
            cl = []
            cd = []
            cm = []

            # read polar information line by line
            while True:
                line = f.readline()
                if 'EOT' in line:
                    break
                data = [float(s) for s in line.split()]
                alpha.append(data[0])
                cl.append(data[1])
                cd.append(data[2])
                cm.append(data[3])

            polars.append(polarType(Re, alpha, cl, cd, cm))

        f.close()

        return cls(polars)

    @classmethod
    def initFromAirfoilShape(cls, Saf, afOptions=None, airfoilShapeMethod=None, airfoilAnalysisMethod=None, polarType=Polar):
        """Construct Airfoil object from airfoil shape parameters

        Parameters
        ----------
        Saf : array of str/int/floats
            Airfoil shape parameters that used to generate airfoil coordinates that are then analyzed.
            Method is specified in afOptions: CST, CoordinateFile, NACA

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []
        if airfoilShapeMethod is not None and afOptions is None:
            afOptions = dict(AirfoilParameterization=airfoilShapeMethod)
        elif airfoilShapeMethod is not None and afOptions is not None:
            afOptions['AirfoilParameterization'] = airfoilShapeMethod
        if airfoilAnalysisMethod is not None and afOptions is None:
            afOptions = dict(AnalysisMethod=airfoilAnalysisMethod)
        elif airfoilAnalysisMethod is not None and afOptions is not None:
            afOptions['AnalysisMethod'] = airfoilAnalysisMethod
        afanalysis = AirfoilAnalysis(Saf, afOptions)
        cl, cd, cm, alphas, failure = afanalysis.computeSpline()
        polars.append(polarType(afanalysis.afOptions['SplineOptions']['Re'], alphas, cl, cd, cm))

        # self.failure = failure #TODO
        # if failure:
        #     Saf = np.asarray([-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25])

        return cls(polars)

    @classmethod
    def initFromAirfoilAnalysis(cls, Saf, afanalysis, polarType=Polar):
        """Construct Airfoil object from airfoil shape parameters

        Parameters
        ----------
        Saf : array of str/int/floats
            Airfoil shape parameters that used to generate airfoil coordinates that are then analyzed.
            Method is specified in afOptions: CST, CoordinateFile, NACA

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []
        afanalysis.setAirfoilShape(Saf)
        cl, cd, cm, alphas, failure = afanalysis.computeSpline()
        polars.append(polarType(afanalysis.afOptions['SplineOptions']['Re'], alphas, cl, cd, cm))

        return cls(polars)

    def getPolar(self, Re):
        """Gets a Polar object for this airfoil at the specified Reynolds number.

        Parameters
        ----------
        Re : float
            Reynolds number

        Returns
        -------
        obj : Polar
            a Polar object

        Notes
        -----
        Interpolates as necessary. If Reynolds number is larger than or smaller than
        the stored Polars, it returns the Polar with the closest Reynolds number.

        """

        p = self.polars

        if Re <= p[0].Re:
            return copy.deepcopy(p[0])

        elif Re >= p[-1].Re:
            return copy.deepcopy(p[-1])

        else:
            Relist = [pp.Re for pp in p]
            i = np.searchsorted(Relist, Re)
            weight = (Re - Relist[i-1]) / (Relist[i] - Relist[i-1])
            return p[i-1].blend(p[i], weight)



    def blend(self, other, weight):
        """Blend this Airfoil with another one with the specified weighting.


        Parameters
        ----------
        other : Airfoil
            other airfoil to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        obj : Airfoil
            a blended Airfoil object

        Notes
        -----
        First finds the unique Reynolds numbers.  Evaluates both sets of polars
        at each of the Reynolds numbers, then blends at each Reynolds number.

        """

        # combine Reynolds numbers
        Relist1 = [p.Re for p in self.polars]
        Relist2 = [p.Re for p in other.polars]
        Relist = np.union1d(Relist1, Relist2)

        # blend polars
        n = len(Relist)
        polars = [0]*n
        for i in list(range(n)):
            p1 = self.getPolar(Relist[i])
            p2 = other.getPolar(Relist[i])
            polars[i] = p1.blend(p2, weight)


        return Airfoil(polars)


    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """apply 3-D rotational corrections to each polar in airfoil

        Parameters
        ----------
        r_over_R : float
            radial position / rotor radius
        chord_over_r : float
            local chord / local radius
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        airfoil : Airfoil
            airfoil with 3-D corrections

        See Also
        --------
        Polar.correction3D : apply 3-D corrections for a Polar

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.correction3D(r_over_R, chord_over_r, tsr, alpha_max_corr, alpha_linear_min, alpha_linear_max)

        return Airfoil(polars)


    def extrapolate(self, cdmax, AR=None, cdmin=0.001):
        """apply high alpha extensions to each polar in airfoil

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            blade aspect ratio (rotor radius / chord at 75% radius).  if included
            it is used to estimate cdmax
        cdmin: minimum drag coefficient

        Returns
        -------
        airfoil : Airfoil
            airfoil with +/-180 degree extensions

        See Also
        --------
        Polar.extrapolate : extrapolate a Polar to high angles of attack

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.extrapolate(cdmax, AR, cdmin)

        return Airfoil(polars)

    def interpToCommonAlpha(self, alpha=None):
        """Interpolates all polars to a common set of angles of attack

        Parameters
        ----------
        alpha : ndarray, optional
            common set of angles of attack to use.  If None a union of
            all angles of attack in the polars is used.

        """

        if alpha is None:
            # union of angle of attacks
            alpha = []
            for p in self.polars:
                alpha = np.union1d(alpha, p.alpha)

        # interpolate each polar to new alpha
        n = len(self.polars)
        polars = [0]*n
        if n == 1:
            polars[0] = self.polar_type(p.Re, alpha, p.cl, p.cd, p.cm)
            return Airfoil(polars)
        for idx, p in enumerate(self.polars):
            cl = np.interp(alpha, p.alpha, p.cl)
            cd = np.interp(alpha, p.alpha, p.cd)
            cm = np.interp(alpha, p.alpha, p.cm)
            polars[idx] = self.polar_type(p.Re, alpha, cl, cd, cm)

        return Airfoil(polars)

    def writeToAerodynFile(self, filename):
        """Write the airfoil section data to a file using AeroDyn input file style.

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        # aerodyn and wtperf require common set of angles of attack
        af = self.interpToCommonAlpha()

        f = open(filename, 'w')

        f.write('AeroDyn airfoil file.\n')
        f.write('Compatible with AeroDyn v13.0.\n')
        f.write('Generated by airfoilprep.py\n')
        f.write('{0:<10d}\t\t{1:40}'.format(len(af.polars), 'Number of airfoil tables in this file')+'\n')
        for p in af.polars:
           f.write('{0:<10f}\t{1:40}'.format(p.Re/1e6, 'Reynolds number in millions.')+'\n')
           param = p.unsteadyparam()
           f.write('{0:<10f}\t{1:40}'.format(param[0], 'Control setting')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[1], 'Stall angle (deg)')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[2], 'Angle of attack for zero Cn for linear Cn curve (deg)')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[3], 'Cn slope for zero lift for linear Cn curve (1/rad)')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[4], 'Cn at stall value for positive angle of attack for linear Cn curve')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[5], 'Cn at stall value for negative angle of attack for linear Cn curve')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[6], 'Angle of attack for minimum CD (deg)')+'\n')
           f.write('{0:<10f}\t{1:40}'.format(param[7], 'Minimum CD value')+'\n')
           for a, cl, cd, cm in list(zip(p.alpha, p.cl, p.cd, p.cm)):
               f.write('{:<10f}\t{:<10f}\t{:<10f}\t{:<10f}'.format(a, cl, cd, cm)+'\n')
           f.write('EOT\n')
        f.close()


    def createDataGrid(self):
        """interpolate airfoil data onto uniform alpha-Re grid.

        Returns
        -------
        alpha : ndarray (deg)
            a common set of angles of attack (union of all polars)
        Re : ndarray
            all Reynolds numbers defined in the polars
        cl : ndarray
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : ndarray
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """
        if len(self.polars) > 1:
            af = self.interpToCommonAlpha()
            polarList = af.polars
        else:
            polarList = self.polars
        # angle of attack is already same for each polar
        alpha = polarList[0].alpha

        # all Reynolds numbers
        Re = [p.Re for p in polarList]

        # fill in cl, cd grid
        cl = np.zeros((len(alpha), len(Re)))
        cd = np.zeros((len(alpha), len(Re)))
        cm = np.zeros((len(alpha), len(Re)))

        for (idx, p) in enumerate(polarList):
            cl[:, idx] = p.cl
            cd[:, idx] = p.cd
            cm[:, idx] = p.cm


        return alpha, Re, cl, cd, cm

    def plot(self, single_figure=True):
        """plot cl/cd/cm polars

        Parameters
        ----------
        single_figure : bool
            True  : plot all cl on the same figure (same for cd,cm)
            False : plot all cl/cd/cm on separate figures

        Returns
        -------
        figs : list of figure handles

        """

        import matplotlib.pyplot as plt

        figs = []

        # if in single figure mode (default)
        if single_figure:
            # generate figure handles
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            figs.append(fig1)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            figs.append(fig2)

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            figs.append(fig3)

            # loop through polars and plot
            for p in self.polars:
                # plot cl
                ax1.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax1.set_xlabel('angle of attack (deg)')
                ax1.set_ylabel('lift coefficient')
                ax1.legend(loc='best')

                # plot cd
                ax2.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax2.set_xlabel('angle of attack (deg)')
                ax2.set_ylabel('drag coefficient')
                ax2.legend(loc='best')

                # plot cm
                ax3.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax3.set_xlabel('angle of attack (deg)')
                ax3.set_ylabel('moment coefficient')
                ax3.legend(loc='best')

        # otherwise, multi figure mode -- plot all on separate figures
        else:
            for p in self.polars:
                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('lift coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('drag coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('moment coefficient')
                ax.legend(loc='best')
        return figs


class AirfoilAnalysis:
    """A helper class to generate 2D airfoil coordinates and perform the airfoil analysis

    """

    def __init__(self, Saf, afOptions=None, airfoilShapeMethod=None, numCoordinates=200, generatePreCompModel=True):
        """Constructor
        prepare airfoil shapes or generate precomputational model

        Parameters
        ----------
        Saf : str/int/float
            airfoil shape parameters that are used to generate airfoil coordinates to be ready for analysis
            method is specified in afOptions or airfoilShapeMethod as either 'CST', 'CoordinateFile', or 'NACA'
        afOptions : dictionary, optional
            a number of options to control how the airfoil shape is generated or analyzed, overrides defaultOptions
        airfoilShapeMethod : str, optional
            specifies the type of airfoil shape parameter: 'CST', 'CoordinateFile', or 'NACA'
        numCoordinates : int, optional
            sets the number of points returned from the airfoil coordinates
        generatePreCompModel : bool, optional
            only applicable if the airfoil parameterization is a precomputational method
            if True the surrogate model is automatically generated
            if False the surrogate model is not generated for use to compute only the airfoil coordinates

        Returns
        -------
        obj : AirfoilAnalysis
            AirfoilAnalysis object

        """
        # Default dictionary of airfoil parameterization and analysis options
        defaultOptions = dict(AnalysisMethod='XFOIL', AirfoilParameterization='CST',
                        CFDOptions=dict(iterations=50000, processors=0, configFile='compRANS.cfg', computeAirfoilsInParallel=False, CFDGradType='AutoDiff', DirectoryForComputation='AirfoilAnalysisFiles'),
                        GradientOptions=dict(ComputeGradient=False, ComputeAirfoilGradients=True, fd_step=1.e-6, cs_step=1.e-20),
                        SplineOptions=dict(AnalysisMethod='XFOIL', maxDirectAoA=180, alphas=np.linspace(-15, 15, 30), Re=1e6, cd_max=1.5,
                                           correction3D=False, r_over_R=0.5, chord_over_r=0.15, tsr=7.55),
                        PrecomputationalOptions=dict(AirfoilParameterization='Blended', numAirfoilFamilies=2, numAirfoilsToCompute=10, tcMax=0.42, tcMin=0.13, airfoilShapeMethod='CoordinateFile'))
        # Override any provided airfoil options
        if afOptions is not None:
            for k, v in afOptions.items:
                if v is not None:
                    if type(v) is dict:
                        for k2, v2 in afOptions[k].items:
                            if v2 is not None:
                                defaultOptions[k][k2] = afOptions[k][k2]
                    else:
                        defaultOptions[k] = afOptions[k]
        if airfoilShapeMethod is not None:
            defaultOptions['AirfoilParameterization'] = airfoilShapeMethod

        # Set configurations
        self.Saf = Saf
        self.afOptions = defaultOptions
        self.numCoordinates = numCoordinates
        self.af_parameterization = defaultOptions['AirfoilParameterization']
        self.af_analysis = defaultOptions['AnalysisMethod']
        self.af_spline_analysis = defaultOptions['AnalysisMethod']
        self.generatedPreComp = False
        if self.af_parameterization == 'CST':
            self.af_dof = len(self.Saf)
        else:
            self.af_dof = 1

        self.derivatives = self.afOptions['GradientOptions']['ComputeGradient']
        if defaultOptions['GradientOptions']['ComputeGradient']:
            self.derivatives_af = self.afOptions['GradientOptions']['ComputeAirfoilGradients']
        else:
            self.derivatives_af = False

        # Create folder for doing CFD calculations if it does not already exist
        self.basepath = defaultOptions['CFDOptions']['DirectoryForComputation']
        self.basepathCFD = self.basepath + os.path.sep + 'CFD'
        if not os.path.exists(self.basepath) and self.af_analysis == 'CFD':
            os.makedirs(self.basepath)
            if not os.path.exists(self.basepath + os.path.sep + 'CFD'):
                os.makedirs(self.basepath + os.path.sep + 'CFD')

        # Check if airfoil analysis methods installed
        if self.af_analysis == 'XFOIL':
            try:
                import pyXLIGHT
            except:
                raise ValueError('XFOIL not installed correctly. See https://github.com/Ry10/pyxlight for more details.')
        elif self.af_analysis == 'CFD':
            try:
                import SU2
            except:
                raise ValueError('SU2 CFD not installed correctly. See https://github.com/su2code/SU2/wiki/Installation for more details.')

        # If not precomputational generate coordinates, else generate surrogate model
        if self.af_parameterization == 'Precomputational':
            self.n, self.thick_max, self.thick_min = self.afOptions['PrecomputationalOptions']['numAirfoilsToCompute'], self.afOptions['PrecomputationalOptions']['tcMax'], self.afOptions['PrecomputationalOptions']['tcMin']
            self.af_precomputational_parameterization = self.afOptions['PrecomputationalOptions']['AirfoilParameterization']
            self.__generatePreCompCoords()
            if generatePreCompModel:
                import time
                print("Generating precomputational model")
                time0 = time.time()
                self.__generatePreCompModel()
                self.generatedPreComp = True
                print("Precomputational model generation complete in " + str(time.time() - time0), " seconds.")

        self.x, self.y, self.xl, self.xu, self.yl, self.yu = self.__setCoordinates()
        self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.__setCoordinates(complexNum=True)
        self.tc = max(self.y) - min(self.y)
        # Set if CFD simulations are simultaneously run or not
        if self.afOptions is not None:
            if self.afOptions['AnalysisMethod'] == 'CFD':
                self.parallel = self.afOptions['CFDOptions']['computeAirfoilsInParallel']
            else:
                self.parallel = False
        else:
            self.parallel = False

        self.cl_storage = []
        self.cd_storage = []
        self.alpha_storage = []
        self.dcl_storage = []
        self.dcd_storage = []
        self.dalpha_storage = []
        self.dcl_dSaf_storage = []
        self.dcd_dSaf_storage = []
        self.dalpha_dSaf_storage = []
        self.cl_spline, self.cd_spline, self.cl_splines_grad, self.cd_splines_grad = None, None, None, None

    ########################################
    #      AIRFOIL COORDINATE METHODS      #
    ########################################
    def getCoordinates(self, Saf=None, form=None, complexNum=False, xl=None, xu=None, airfoilShapeMethod=None):
        """obtain airfoil coordinates

        Parameters
        ----------
        form : str, optional
            Determines the form that the airfoil coordinates are returned
            'split' - returns the top and bottom surfaces separately
            'all' - returns combined coordinates from leading edge clockwise and the top and bottom surfaces separately
            default - returns combined coordinates from leading edge clockwise
        complex: bool, optional
            True - the airfoil coordinates are returned as complex numbers (useful for complex step gradients)
            False - the airfoil coordinates are returned as real numbers
        xl: ndarray, optional
            lower surface x-coordinates, if provided then the corresponding y-coordinates are calculated
        xu: ndarray, optional
            upper surface x-coordinates, if provided then the corresponding y-coordinates are calculated

        Returns
        -------
        array of float : x, y
            airfoil coordinates in different forms

        """

        if Saf is not None:
            Saf_old = self.Saf
            self.Saf = Saf
            af_param_old = self.af_parameterization
            if airfoilShapeMethod is not None:
                self.af_parameterization = airfoilShapeMethod
            self.x, self.y, self.xl, self.xu, self.yl, self.yu = self.__setCoordinates()
            self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.x, self.y, self.xl, self.xu, self.yl, self.yu # Complex numbers not necessary for precomputational model
            self.af_parameterization = af_param_old
            self.Saf = Saf_old
        if xl is None or xu is None:
            if form == 'all':
                if complexNum:
                    return self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c
                else:
                    return self.x, self.y, self.xl, self.xu, self.yl, self.yu
            elif form == 'split':
                if complexNum:
                    return self.xl_c, self.xu_c, self.yl_c, self.yu_c
                else:
                    return self.xl, self.xu, self.yl, self.yu
            else:
                if complexNum:
                    return self.x_c, self.y_c
                else:
                    return self.x, self.y
        else:
            if self.af_parameterization != 'CST':
                raise ValueError('Not currently working for non-CST parameterization')
            else:
                yl, yu = self.__cstYgivenX(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu, complexNum)
            return xl, xu, yl, yu

    def saveCoordinateFile(self, afFile, x=None, y=None, complexNum=False):
        """ saves coordinates to file

        Parameters
        ----------
        afFile : str
            file name for coordinates to be saved
        x, y : array of floats, optional
            coordinates to be saved as combined coordinates from leading edge clockwise

        Returns
        -------
        None

        """

        afFile = open(afFile, 'w')
        afFile.write('Airfoil\n')
        if x is None or y is None:
            for i in list(range(len(self.x))):
                if complexNum:
                    afFile.write('{:<20f}\t{:<20f}'.format(complex(np.real(self.x_c[i]), np.imag(self.x_c[i])), complex(np.real(self.y_c[i]), np.imag(self.y_c[i]))) + '\n')
                else:
                    afFile.write('{:<20f}\t{:<20f}'.format(self.x[i], self.y[i]) + '\n')
        else:
            for i in list(range(len(x))):
                if complex:
                    afFile.write('{:<20f}\t{:<20f}'.format(complex(np.real(x[i]), np.imag(x[i])), complex(np.real(y[i]), np.imag(y[i]))) + '\n')
                else:
                    afFile.write('{:<20f}\t{:<20f}'.format(x[i], y[i]) + '\n')
        afFile.close()

    def readCoordinateFile(self, afFile, complexNum=False):
        """ reads in airfoil coordinates file

        Parameters
        ----------
        afFile : str
            file name with airfoil coordinates as combined coordinates from leading edge clockwise
        complex: bool, optional
            True - the airfoil coordinates are complex numbers
            False - the airfoil coordinates are real numbers

        Returns
        -------
        x, y : array of floats
            airfoil coordinates as combined coordinates from leading edge clockwise

        """
        try:
            afFile = open(afFile, 'r')
        except:
            pass
        x, y = [], []
        for row in afFile:
            try:
                row = row.split()
                if complexNum:
                    x.append(complex(row[0]))
                    y.append(complex(row[1]))
                else:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
            except:
                pass
        afFile.close()
        x = np.asarray(x)
        y = np.asarray(y)
        self.x = x
        self.y = y
        return x, y

    def plotAirfoilShape(self):
        """ plots airfoil coordinates

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(self.x, self.y, 'x')
        plt.axis('equal')
        plt.show()

    def getCoordinateDerivatives(self, xl=None, xu=None):
        """ provides geometric gradients with respect to airfoil surface

        Parameters
        ----------
        xl : array of floats
            lower x-coordinates
        xu : array of floats
            lower x-coordinates

        Returns
        -------
        dy_dSaf: 2D array of floats
            the derivative of y-coordinates with respect to airfoil shape parameters

        """
        if self.af_parameterization != 'CST':
            raise ValueError('Not currently working for non-CST parameterization')
        else:
            if xl is None or xu is None:
                dy_dSaf = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, self.xl, self.xu)
            else:
                dy_dSaf = self.__cstYDerivatives(self.wl, self.wu, self.numCoordinates, self.dz, xl, xu)
        return dy_dSaf

    def coordToCST(self, afFile=None, x=None, y=None):
        """ given the coordinates obtain the Kulfan parameters, only works for 8 Kulfan parameters
        Parameters
        ----------
        afFile : str, optional
            airfoil file name
        x : array, optional
            x-coordinates
        y : array, optional
            y-coordinates

        Returns
        -------
        None

        """
        # afFile takes precendent, then provided x and y coordinates, then class coordinates

        from pyOpt import Optimization, SLSQP

        if afFile is not None:
            x1, y1 = self.readCoordinateFile(afFile)
        elif x is not None and y is not None:
            x1, y1 = x, y
        else:
            raise ValueError('Please provide either airfoil coordinate file or x and y coordinates.')

        self.dz = 0
        N = len(x1)
        self.N1 = 0.5
        self.N2 = 1.0
        def objfunc(x):

            wl = [x[0], x[1], x[2], x[3]]
            wu = [x[4], x[5], x[6], x[7]]

            x2, y2 = self.__cstCoordinatesReal(wl, wu, N, self.dz)

            f = 0
            for i in list(range(N-1)):
                f += abs(y1[i]*100 - y2[i]*100)**2

            g = []

            fail = 0
            return f, g, fail

        opt_prob = Optimization('Coordinate to CST Parameterization Conversion', objfunc)
        opt_prob.addVar('x1','c', lower=-2.0,upper=2.0, value=-1.0)
        opt_prob.addVar('x2','c', lower=-2.0,upper=2.0, value=-1.0)
        opt_prob.addVar('x3','c', lower=-2.0,upper=2.0, value=-1.0)
        opt_prob.addVar('x4','c', lower=-2.0,upper=2.0, value=-1.0)
        opt_prob.addVar('x5','c', lower=-2.0, upper=2.0, value=1.0)
        opt_prob.addVar('x6','c', lower=-2.0, upper=2.0, value=1.0)
        opt_prob.addVar('x7','c', lower=-2.0, upper=2.0, value=1.0)
        opt_prob.addVar('x8','c', lower=-2.0, upper=2.0, value=1.0)
        opt_prob.addObj('f')

        # Instantiate Optimizer (SLSQP) & Solve Problem
        slsqp = SLSQP()
        slsqp.setOption('IPRINT',-1)
        slsqp(opt_prob, sens_type='FD')
        sol = opt_prob.solution(0)
        CST = np.zeros(8)
        for i in list(range(len(CST))):
            CST[i] = sol._variables[i].value
        return CST

    def setNumCoordinates(self, numCoordinates):
        """ set the number of coordinates
        Parameters
        ----------
        numCoordinates : int
            Sets the number of points returned from the airfoil coordinates

        Returns
        -------
        None

        """
        self.numCoordinates = numCoordinates
        self.x, self.y, self.xl, self.xu, self.yl, self.yu = self.__setCoordinates()
        self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.__setCoordinates(complexNum=True)

    def setAirfoilShape(self, Saf):
        """ set the number of coordinates
        Parameters
        ----------
        numCoordinates : int
            Sets the number of points returned from the airfoil coordinates

        Returns
        -------
        None

        """
        self.Saf = Saf
        self.x, self.y, self.xl, self.xu, self.yl, self.yu = self.__setCoordinates()
        self.x_c, self.y_c, self.xl_c, self.xu_c, self.yl_c, self.yu_c = self.__setCoordinates(complexNum=True)


    def __setCoordinates(self, complexNum=False):
        """ return the coordinates
        Parameters
        ----------
        complex : bool, optional
            True - the airfoil coordinates are returned as complex numbers
            False - the airfoil coordinates are returned as real numbers

        Returns
        -------
        x : array of floats
            x-coordinates
        y : array of floats
            y-coordinates
        xl : array of floats
            lower surface x-coordinates
        xu : array of floats
            upper surface y-coordinates
        yl : array of floats
            lower surface y-coordinates
        yu : array of floats
            upper surface y-coordinates
        """
        if self.af_parameterization == 'CST':
            self.N1, self.N2, self.dz = 0.5, 1.0, 0.0
            if complexNum:
                x, y, xl, xu, yl, yu = self.__cstCoordinatesComplex()
            else:
                x, y, xl, xu, yl, yu = self.__cstCoordinates()
        elif self.af_parameterization == 'CoordinateFile':
            x, y = self.readCoordinateFile(self.Saf, complexNum)
            xl, xu, yl, yu = [], [], [], []
        elif self.af_parameterization == 'NACA':
            x, y = self.__NACA(self.Saf)
            xl, xu, yl, yu = [], [], [], []
        elif self.af_parameterization == 'Precomputational':
            if complexNum:
                x, y, xl, xu, yl, yu = self.__getPreCompCoordinates(self.Saf, form='all', complexNum=complexNum)
            else:
                x, y, xl, xu, yl, yu = self.__getPreCompCoordinates(self.Saf, form='all')
        else:
            raise ValueError('Only airfoil parameterizations of CST, CoordinateFile, NACA, or Precomputational are currently supported.')
        return x, y, xl, xu, yl, yu

    def __generatePreCompCoords(self):
        """generate coordinates for precomp model

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.xx0, self.yy0, self.thicknesses0 = self.__generatePreCompCoordinates(0)
        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            self.xx1, self.yy1, self.thicknesses1 = self.__generatePreCompCoordinates(1)


    def __getPreCompCoordinates(self, Saf, form=None, complexNum=False):
        """obtain precomputational model coordinates

        Parameters
        ----------
        Saf : ndarray of size 2
            precomputational airfoil shape parameter (thickness-to-chord ratio, blended family factor)
        form : str, optional
            Determines the form that the airfoil coordinates are returned
            'split' - returns the top and bottom surfaces separately
            'all' - returns combined coordinates from leading edge clockwise and the top and bottom surfaces separately
            default - returns combined coordinates from leading edge clockwise

        Returns
        -------
        x, y - airfoil coordinates in various forms

        """
        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            thickness_to_chord = Saf[0]
            blended_factor = Saf[1]

            for i in list(range(len(self.thicknesses0))):
                if thickness_to_chord >= self.thicknesses0[i] and thickness_to_chord < self.thicknesses0[i+1]:
                    x0 = self.xx0[i]
                    y0 = self.yy0[i]
                    yy0 = self.__convertTCCoordinates(thickness_to_chord, y0)

            for i in list(range(len(self.thicknesses1))):
                if thickness_to_chord >= self.thicknesses1[i] and thickness_to_chord < self.thicknesses1[i+1]:
                    x1 = self.xx1[i]
                    y1 = self.yy1[i]
                    yy1 = self.__convertTCCoordinates(thickness_to_chord, y1)

            xx = np.zeros(len(x1))
            yy = np.zeros(len(x1))
            if len(x1) == len(x0):
                for i in list(range(len(x0))):
                    xx[i] = x1[i]
                    yy[i] = yy0[i] + blended_factor*(yy1[i] - yy0[i])
            else:
                print("Uneven blended airfoils")
        else:
            try:
                thickness_to_chord = Saf[0]
            except:
                thickness_to_chord = Saf
            for i in list(range(len(self.thicknesses0))):
                if thickness_to_chord >= self.thicknesses0[i] and thickness_to_chord < self.thicknesses0[i+1]:
                    x0 = self.xx0[i]
                    y0 = self.yy0[i]
                    yy0 = self.__convertTCCoordinates(thickness_to_chord, y0)
            xx = x0
            yy = yy0

        try:
            zerind = np.where(xx == 0)  # Used to separate upper and lower surfaces
            zerind = zerind[0][0]
        except:
            zerind = len(xx)/2

        xl = np.zeros(zerind)
        xu = np.zeros(len(xx)-zerind)
        yl = np.zeros(zerind)
        yu = np.zeros(len(xx)-zerind)

        for z in list(range(len(xu))):
            xu[z] = xx[z]        # Lower surface x-coordinates
            yu[z] = yy[z]
        for z in list(range(len(xl))):
            xl[z] = xx[z + zerind]   # Upper surface x-coordinates
            yl[z] = yy[z + zerind]

        # Get in ascending order if not already
        if xl[int(len(xl)/4)] > 0.5:
            xl = xl[::-1]
            yl = yl[::-1]
        if xu[int(len(xu)/4)] > 0.5:
            xu = xu[::-1]
            yu = yu[::-1]

        if xu[0] != 0.0:
            xu[0] = 0.0
        if xl[0] != 0.0:
            xl[0] = 0.0
        if xu[-1] != 1.0:
            xu[-1] = 1.0
        if xl[-1] != 1.0:
            xl[-1] = 1.0

        if yu[0] != 0.0:
            yu[0] = 0.0
        if yl[0] != 0.0:
            yl[0] = 0.0
        if yu[-1] != 0.0:
            yu[-1] = 0.0
        if yl[-1] != 0.0:
            yl[-1] = 0.0

        # Get in right order for precomp
        xl = xl[::-1]
        yl = yl[::-1]

        if complexNum:
            xx, yy, xl, xu, yl, yu = xx.astype(complex), yy.astype(complex), xl.astype(complex), xu.astype(complex), yl.astype(complex), yu.astype(complex)
        if form == 'all':
            return xx, yy, xl, xu, yl, yu
        elif form == 'split':
            return xl, xu, yl, yu
        else:
            return xx, yy

    def __cstToKulfan(self, CST):
        """ splits the Kulfan parameters into lower and top components
        Parameters
        ----------
        CST : array
            Array of all the Kulfan parameters, typically the bottom surface is first

        Returns
        -------
        wl : array
            lower surface Kulfan parameters
        wu : array
            upper surface Kulfan parameters
        """
        n = int(len(CST) / 2)
        wu = np.zeros(n)
        wl = np.zeros(n)
        for j in list(range(n)):
            wl[j] = CST[j]
            wu[j] = CST[j + n]
        w1 = np.average(wl)
        w2 = np.average(wu)
        # if the Kulfan parameters are given top first then switch
        if w1 < w2:
            self.switchedCST = False
        else:
            higher = wl
            lower = wu
            wl = lower
            wu = higher
            self.switchedCST = True
        return wl, wu

    def __NACA(self, naca_num):
        """
        Parameters
        ----------
        naca_num : int
            The 4 or 5 digit NACA number

        Returns
        -------
        x : array
            x-coordinates
        y : array
            y-coordinates
        """
        x, y = self.__naca(naca_num, int(self.numCoordinates/2))
        return x, y

    def __ClassShape(self, w, x, N1, N2, dz):
        """
        Parameters
        ----------
        w : array
            Kulfan parameters
        x : array
            x-coordinates
        N1 : float
            Determines type of CST method
        N2 : float
            Determines type of CST method
        dz : float
            trailing edge thickness
        complex : bool, optional
            True - the airfoil coordinates are returned as complex numbers
            False - the airfoil coordinates are returned as real numbers
        Returns
        -------
        y : array
            y-coordinates
        """
        # Class function; taking input of N1 and N2
        C = np.zeros(len(x))
        for i in list(range(len(x))):
            C[i] = x[i]**N1*((1-x[i])**N2)

        # Shape function; using Bernstein Polynomials
        n = len(w) - 1  # Order of Bernstein polynomials

        K = np.zeros(n+1)
        for i in list(range(0, n+1)):
            K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

        S = np.zeros(len(x))
        for i in list(range(len(x))):
            S[i] = 0
            for j in list(range(0, n+1)):
                S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

        # Calculate y output
        y = np.zeros(len(x))
        for i in list(range(len(y))):
            y[i] = C[i] * S[i] + x[i] * dz

        return y

    def __ClassShapeComplex(self, w, x, N1, N2, dz):
        """
        Parameters
        ----------
        w : array
            Kulfan parameters
        x : array
            x-coordinates
        N1 : float
            Determines type of CST method
        N2 : float
            Determines type of CST method
        dz : float
            trailing edge thickness

        Returns
        -------
        y : array
            y-coordinates
        """
        # Class function; taking input of N1 and N2
        C = np.zeros(len(x), dtype=complex)
        for i in list(range(len(x))):
            C[i] = x[i]**N1*((1-x[i])**N2)

        # Shape function; using Bernstein Polynomials
        n = len(w) - 1  # Order of Bernstein polynomials

        K = np.zeros(n+1, dtype=complex)
        for i in list(range(0, n+1)):
            K[i] = mpmath.factorial(n)/(mpmath.factorial(i)*(mpmath.factorial((n)-(i))))

        S = np.zeros(len(x), dtype=complex)
        for i in list(range(len(x))):
            S[i] = 0
            for j in list(range(0, n+1)):
                S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

        # Calculate y output
        y = np.zeros(len(x), dtype=complex)
        for i in list(range(len(y))):
            y[i] = C[i] * S[i] + x[i] * dz

        return y

    def __cstYgivenX(self, wl, wu, N, dz, xl, xu, complexNum=False):
        """
        Parameters
        ----------
        wl : array
            lower surface Kulfan parameters
        wu : array
            upper surface Kulfan parameters
        N : int
            num of coordinates
        dz : float
            trailing edge thickness
        xl : array
            lower surface x-coordinates
        xu : array
            upper surface y-coordinates
        complex : bool, optional
            True - the airfoil coordinates are returned as complex numbers
            False - the airfoil coordinates are returned as real numbers

        Returns
        -------
        yl : array
            lower y-coordinates
        yu : array
            upper y-coordinates

        """
        if complexNum:
            yl = self.__ClassShapeComplex(wl, xl, self.N1, self.N2, -self.dz) # Call ClassShape function to determine lower surface y-coordinates
            yu = self.__ClassShapeComplex(wu, xu, self.N1, self.N2, self.dz)  # Call ClassShape function to determine upper surface y-coordinates
        else:
            yl = self.__ClassShape(wl, xl, self.N1, self.N2, -self.dz) # Call ClassShape function to determine lower surface y-coordinates
            yu = self.__ClassShape(wu, xu, self.N1, self.N2, self.dz)  # Call ClassShape function to determine upper surface y-coordinates
        return yl, yu

    def __cstYDerivatives(self, wl, wu, N, dz, xl, xu):
        """
        Parameters
        ----------
        wl : array
            lower surface Kulfan parameters
        wu : array
            upper surface Kulfan parameters
        N : int
            num of coordinates
        dz : float
            trailing edge thickness
        xl : array
            lower surface x-coordinates
        xu : array
            upper surface y-coordinates

        Returns
        -------
        dy_dSaf : array
            geometric gradients with respect to airfoil shape parameters

        """
        dyl = self.__ClassShapeDerivative(wl, xl, self.N1, self.N2, -self.dz) # Call ClassShape function to determine lower surface y-coordinates
        dyu = self.__ClassShapeDerivative(wu, xu, self.N1, self.N2, self.dz)  # Call ClassShape function to determine upper surface y-coordinates
        dyl_dzeros = np.zeros((len(wl), N-len(xl)))
        dyu_dzeros = np.zeros((len(wu), N-len(xu)))
        # if provided top Kulfan first then switch
        if self.switchedCST:
            dyl_dw = np.hstack((dyl, dyl_dzeros))
            dyu_dw = np.hstack((dyu_dzeros, dyu))
            dy_dSaf = np.vstack((dyl_dw, dyu_dw))
        else:
            dyl_dw = np.hstack((dyl, dyl_dzeros))
            dyu_dw = np.hstack((dyu_dzeros, dyu))
            dy_dSaf = np.vstack((dyu_dw, dyl_dw))
        return dy_dSaf

    def __cstCoordinates(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        x : array of floats
            x-coordinates
        y : array of floats
            y-coordinates
        xl : array of floats
            lower surface x-coordinates
        xu : array of floats
            upper surface y-coordinates
        yl : array of floats
            lower surface y-coordinates
        yu : array of floats
            upper surface y-coordinates
        """

        self.wl, self.wu = self.__cstToKulfan(self.Saf)
        self.dz = 0.0
        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
        x, y, xl, xu, yl, yu = self.__cstCoordinatesReal(wl, wu, N, dz, form='all')
        return x, y, xl, xu, yl, yu

    def __cstCoordinatesReal(self, wl, wu, N, dz, form=None):
        """
        Parameters
        ----------
        wl : array
            lower surface Kulfan parameters
        wu : array
            upper surface Kulfan parameters
        N : int
            num of coordinates
        dz : float
            trailing edge thickness

        Returns
        -------
        x : array
            x-coordinates
        y : array
            y-coordinates

        """
        x = np.ones((N, 1))
        zeta = np.zeros((N, 1))
        for z in list(range(0, N)):
            zeta[z] = 2 * pi / N * z
            if z == N - 1:
                zeta[z] = 2.0 * pi
            x[z] = 0.5*(cos(zeta[z])+1.0)

        try:
            zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
            zerind = zerind[0][0]
        except:
            zerind = N/2

        xl = np.zeros(zerind)
        xu = np.zeros(N-zerind)

        for z in list(range(len(xl))):
            xl[z] = x[z]        # Lower surface x-coordinates
        for z in list(range(len(xu))):
            xu[z] = x[z + zerind]   # Upper surface x-coordinates

        yl = self.__ClassShape(wl, xl, self.N1, self.N2, -self.dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShape(wu, xu, self.N1, self.N2, self.dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x))
        for k in list(range(len(x))):
            x1[k] = x[k][0]
        x = x1

        if form is None:
            return x, y
        else:
            return x, y, xl, xu, yl, yu

    def __cstCoordinatesComplexFull(self, wl, wu, N, dz):
        """
        Parameters
        ----------
        wl : array
            lower surface Kulfan parameters
        wu : array
            upper surface Kulfan parameters
        N : int
            num of coordinates
        dz : float
            trailing edge thickness

        Returns
        -------
        x : array
            x-coordinates
        y : array
            y-coordinates

        """
        # Populate x coordinates
        x = np.ones((N, 1), dtype=complex)
        zeta = np.zeros((N, 1)) #, dtype=complex)
        for z in list(range(0, N)):
            zeta[z] = 2.0 * pi / N * z
            if z == N - 1:
                zeta[z] = 2.0 * pi
            x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

        try:
            zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
            zerind = zerind[0][0]
        except:
            zerind = N/2

        xl = np.zeros(zerind, dtype=complex)
        xu = np.zeros(N-zerind, dtype=complex)

        for z in list(range(len(xl))):
            xl[z] = x[z][0]        # Lower surface x-coordinates
        for z in list(range(len(xu))):
            xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

        yl = self.__ClassShapeComplex(wl, xl, self.N1, self.N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShapeComplex(wu, xu, self.N1, self.N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x), dtype=complex)
        for k in list(range(len(x))):
            x1[k] = x[k][0]
        x = x1
        return x, y

    def __cstCoordinatesComplex(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        x : array of floats
            x-coordinates
        y : array of floats
            y-coordinates
        xl : array of floats
            lower surface x-coordinates
        xu : array of floats
            upper surface y-coordinates
        yl : array of floats
            lower surface y-coordinates
        yu : array of floats
            upper surface y-coordinates
        """

        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
        # Populate x coordinates
        x = np.ones((N, 1), dtype=complex)
        zeta = np.zeros((N, 1)) #, dtype=complex)
        for z in list(range(0, N)):
            zeta[z] = 2.0 * pi / N * z
            if z == N - 1:
                zeta[z] = 2.0 * pi
            x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1

        try:
            zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
            zerind = zerind[0][0]
        except:
            zerind = N/2

        xl = np.zeros(zerind, dtype=complex)
        xu = np.zeros(N-zerind, dtype=complex)

        for z in list(range(len(xl))):
            xl[z] = x[z][0]        # Lower surface x-coordinates
        for z in list(range(len(xu))):
            xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

        yl = self.__ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
        yu = self.__ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

        y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
        y = y[::-1]
        # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
        # coord = [x, y]
        x1 = np.zeros(len(x), dtype=complex)
        for k in list(range(len(x))):
            x1[k] = x[k][0]
        x = x1
        return x, y, xl, xu, yl, yu


    def __ClassShapeDerivative(self, w, x, N1, N2, dz):
        """
        Parameters
        ----------
        w : array
            Kulfan parameters
        x : array
            x-coordinates
        N1 : float
            Determines type of CST method
        N2 : float
            Determines type of CST method
        dz : float
            trailing edge thickness

        Returns
        -------
        dy_dw : array
            Geometric y-coordinate gradients with respect to Kulfan parameters
        """
        n = len(w) - 1
        dy_dw = np.zeros((n+1, len(x)))
        for i in list(range(len(x))):
            for j in list(range(0, n+1)):
                dy_dw[j][i] = x[i]**N1*((1-x[i])**N2) * factorial(n)/(factorial(j)*(factorial((n)-(j)))) * x[i]**(j) * ((1-x[i])**(n-(j)))
        return dy_dw

    def __normalVector(self, w, x, N1, N2, dz):
        """
        Parameters
        ----------
        w : array
            Kulfan parameters
        x : array
            x-coordinates
        N1 : float
            Determines type of CST method
        N2 : float
            Determines type of CST method
        dz : float
            trailing edge thickness

        Returns
        -------
        dy_total : array
            y-coordinate gradients with respect to normal direction

        """
        dy_dw = self.__ClassShapeDerivative(w, x, N1, N2, dz)
        y = self.__ClassShape(w, x, N1, N2, dz)
        dy_total = np.zeros_like(dy_dw)
        n = len(w) - 1
        for i in list(range(len(y))):
            if i == 0 or i == len(y) - 1:
                norm_y = 0
            else:
                # normal vector of forward line adjacent point
                dnormx3 = y[i+1] - y[i]
                dnormy3 = -(x[i+1] - x[i])

                # normal vector of backward line with adjacent point
                dnormx4 = y[i] - y[i-1]
                dnormy4 = -(x[i] - x[i-1])

                dnormx = dnormx3 + dnormx4
                dnormy = dnormy3 + dnormy4

                norm_y = dnormy / np.sqrt(dnormy**2 + dnormx**2)
            for j in list(range(0, n+1)):
                dy_total[j][i] = dy_dw[j][i] * norm_y
        return dy_total



    ########################################
    #       PRECOMPUTATIONAL METHOD        #
    ########################################
    def evaluate(self, alpha, Re, analysisMethod='XFOIL', derivatives=None):
        derivatives = 'flow, shape, all'
        if self.Saf is not None and abs(np.degrees(alpha)) < self.afOptions['SplineOptions']['maxDirectAoA']:
            if alpha in self.alpha_storage and self.afOptions['AirfoilParameterization'] != 'Precomputational':
                index = self.alpha_storage.index(alpha)
                cl = self.cl_storage[index]
                cd = self.cd_storage[index]
                if self.afOptions['GradientOptions']['ComputeGradient'] and alpha in self.dalpha_storage:
                    index = self.dalpha_storage.index(alpha)
                    dcl_dalpha = self.dcl_storage[index]
                    dcd_dalpha = self.dcd_storage[index]
                else:
                    dcl_dalpha, dcd_dalpha = 0.0, 0.0
                if self.afOptions['GradientOptions']['ComputeAirfoilGradients'] and alpha in self.dalpha_dSaf_storage:
                    index = self.dalpha_dSaf_storage.index(alpha)
                    dcl_dSaf = self.dcl_dSaf_storage[index]
                    dcd_dSaf = self.dcd_dSaf_storage[index]
                else:
                    dcl_dSaf = np.zeros(self.af_dof)
                    dcd_dSaf = np.zeros(self.af_dof)
                dcl_dRe = 0.0
                dcd_dRe = 0.0
            else:
                if self.preCompModel is not None:
                    cl, cd = self.preCompModel.evaluatePreCompModel(alpha, self.Saf)
                    dcl_dalpha, dcl_dSaf, dcd_dalpha, dcd_dSaf = self.preCompModel.derivativesPreCompModel(alpha, self.Saf)
                    dcl_dRe, dcd_dRe = 0.0, 0.0
                else:
                    # if self.afOptions['DirectSpline']:
                    #     cl, cd = self.evaluate_spline(alpha, Re)
                    #     dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = self.derivatives_spline(alpha, Re)
                    #     dcl_dSaf, dcd_dSaf = self.splineFreeFormGrad_spline(alpha, Re)
                    #
                    # else:
                    if self.derivatives:
                        cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf, lexitflag = self.computeDirect(alpha, Re)
                    else:
                        cl, cd = self.computeDirect(alpha, Re)
                        dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf, lexitflag = 0.0, 0.0, 0.0, 0.0, np.zeros(8), np.zeros(8), False
                        if lexitflag or abs(cl) > 2.5 or cd < 0.000001 or cd > 1.5 or not np.isfinite(cd) or not np.isfinite(cl):
                            cl, cd = self.evaluate(alpha, Re)
                            dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe = self.derivatives(alpha, Re)
                self.cl_storage.append(cl)
                self.cd_storage.append(cd)
                self.alpha_storage.append(alpha)
                if self.afOptions['GradientOptions']['ComputeGradient']:
                    self.dcl_storage.append(dcl_dalpha)
                    self.dcd_storage.append(dcd_dalpha)
                    self.dalpha_storage.append(alpha)
                    if self.afOptions['GradientOptions']['ComputeAirfoilGradients']:
                        self.dcl_dSaf_storage.append(dcl_dSaf)
                        self.dcd_dSaf_storage.append(dcd_dSaf)
                        self.dalpha_dSaf_storage.append(alpha)
        else:
            self.__generateSplines()
            cl, cd = self.spline_evaluate(alpha, Re)
            dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe = self.spline_derivatives(alpha, Re)
            dcl_dSaf, dcd_dSaf = self.afShapeDerivatives(alpha, Re)

        return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dSaf, dcd_dSaf

    def afShapeDerivatives(self, alpha, Re):
        dcl_dSaf, dcd_dSaf = np.zeros(self.af_dof), np.zeros(self.af_dof)
        if self.derivatives_af:
            if self.af_parameterization == 'Precomputational':
                dcl_dalpha, dcl_dSaf, dcd_dalpha, dcd_dSaf = self.__derivativesPreCompModel(alpha, self.Saf, bem=True)
            else:
                self.__generateSplines()
                fd_step = self.afOptions['GradientOptions']['fd_step']
                cl_cur = self.cl_spline.ev(alpha, Re)
                cd_cur = self.cd_spline.ev(alpha, Re)
                for i in list(range(self.af_dof)):
                    cl_new_fd = self.cl_splines_grad[i].ev(alpha, Re)
                    cd_new_fd = self.cd_splines_grad[i].ev(alpha, Re)
                    dcl_dSaf[i] = (cl_new_fd - cl_cur) / fd_step
                    dcd_dSaf[i] = (cd_new_fd - cd_cur) / fd_step

        return dcl_dSaf, dcd_dSaf

    def spline_evaluate(self, alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.

        """

        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        return cl, cd

    def spline_derivatives(self, alpha, Re):

        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
            dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)

        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def __generateSplines(self, type='all'):
        if self.cl_spline is None:
            af = Airfoil.initFromAirfoilAnalysis(self.Saf, self)
            if self.afOptions['SplineOptions']['correction3D']:
                af = af.correction3D(self.afOptions['SplineOptions']['r_over_R'], self.afOptions['SplineOptions']['chord_over_r'], self.afOptions['SplineOptions']['tsr'])
            af_extrap = af.extrapolate(self.afOptions['SplineOptions']['cd_max'])
            alpha, Re, cl, cd, cm = af_extrap.createDataGrid()
            alpha = np.radians(alpha)
            self.one_Re = False

            if not all(np.diff(alpha)):
                to_delete = np.zeros(0)
                diff = np.diff(alpha)
                for j in list(range(len(alpha)-1)):
                    if not diff[j] > 0.0:
                        to_delete = np.append(to_delete, j)
                alpha = np.delete(alpha, to_delete)
                cl = np.delete(cl, to_delete)
                cd = np.delete(cd, to_delete)

            # special case if zero or one Reynolds number (need at least two for bivariate spline)
            if len(Re) < 2:
                Re = [1e1, 1e15]
                cl = np.c_[cl, cl]
                cd = np.c_[cd, cd]
                self.one_Re = True

            kx = min(len(alpha)-1, 3)
            ky = min(len(Re)-1, 3)



            # a small amount of smoothing is used to prevent spurious multiple solutions
            self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
            self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)

        if type == 'all':
            if self.cl_splines_grad is None:
                if self.afOptions['GradientOptions']['ComputeAirfoilGradients'] and self.afOptions['GradientOptions']['ComputeGradient'] and self.afOptions['AirfoilParameterization'] != 'Precomputational':
                    self.cl_splines_new = [0]*self.af_dof
                    self.cd_splines_new = [0]*self.af_dof
                    for i in list(range(self.af_dof)):
                        Saf_new = copy.deepcopy(self.Saf)
                        Saf_new[i] += self.afOptions['GradientOptions']['fd_step']
                        af = Airfoil.initFromAirfoilShape(Saf_new, self.afOptions)
                        if self.afOptions['SplineOptions']['correction3D']:
                            af = af.correction3D(self.afOptions['SplineOptions']['r_over_R'], self.afOptions['SplineOptions']['chord_over_r'], self.afOptions['SplineOptions']['tsr'])
                        af_extrap = af.extrapolate(self.afOptions['SplineOptions']['cd_max'])
                        alphas_new, Re_new, cl_new, cd_new, cm_new = af_extrap.createDataGrid()
                        alphas_new = np.radians(alphas_new)

                        if not all(np.diff(alphas_new)):
                            to_delete = np.zeros(0)
                            diff = np.diff(alphas_new)
                            for j in list(range(len(alphas_new)-1)):
                                if not diff[j] > 0.0:
                                    to_delete = np.append(to_delete, j)
                            alphas_new = np.delete(alphas_new, to_delete)
                            cl_new = np.delete(cl_new, to_delete)
                            cd_new = np.delete(cd_new, to_delete)

                        # special case if zero or one Reynolds number (need at least two for bivariate spline)
                        if len(Re_new) < 2:
                            Re2 = [1e1, 1e15]
                            cl_new = np.c_[cl_new, cl_new]
                            cd_new = np.c_[cd_new, cd_new]
                        kx = min(len(alphas_new)-1, 3)
                        ky = min(len(Re2)-1, 3)
                        # a small amount of smoothing is used to prevent spurious multiple solutions
                        self.cl_splines_new[i] = RectBivariateSpline(alphas_new, Re2, cl_new, kx=kx, ky=ky)#, s=0.1)#, s=0.1)
                        self.cd_splines_new[i] = RectBivariateSpline(alphas_new, Re2, cd_new, kx=kx, ky=ky)#, s=0.001) #, s=0.001)

    def __evaluatePreCompModel(self, alpha, Saf, bem=False):
        """ obtain lift and drag coefficients from surrogate model
        Parameters
        ----------
        alpha : float
            angle of attack (rad)
        Saf : array of length 2
            first element is thickness-to-chord ratio
            second element is blended airfoil family factor
        bem : bool, optional
            True - used for the BEM (Blade Element Momentum) method specific for CCBlade
            False - used elsewhere than the BEM method

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        """
        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            tc = Saf[0]
            bf = Saf[1]

            if bem:
                cl0 = self.cl_total_spline0_bem.ev(alpha, tc)
                cd0 = self.cd_total_spline0_bem.ev(alpha, tc)
                cl1 = self.cl_total_spline1_bem.ev(alpha, tc)
                cd1 = self.cd_total_spline1_bem.ev(alpha, tc)
            else:
                cl0 = self.cl_total_spline0.ev(alpha, tc)
                cd0 = self.cd_total_spline0.ev(alpha, tc)
                cl1 = self.cl_total_spline1.ev(alpha, tc)
                cd1 = self.cd_total_spline1.ev(alpha, tc)
            cl = cl0 + bf*(cl1-cl0)
            cd = cd0 + bf*(cd1-cd0)

            # Store for use with derivativesPreCompModel
            self.cl1, self.cl0, self.cd1, self.cd0 = cl1, cl0, cd1, cd0
        else:
            try:
                tc = Saf[0]
            except:
                tc = Saf
            if bem:
                cl0 = self.cl_total_spline0_bem.ev(alpha, tc)
                cd0 = self.cd_total_spline0_bem.ev(alpha, tc)
            else:
                cl0 = self.cl_total_spline0.ev(alpha, tc)
                cd0 = self.cd_total_spline0.ev(alpha, tc)
            self.cl0, self.cd0 = cl0, cd0
            cl, cd = cl0, cd0
        return cl, cd

    def __derivativesPreCompModel(self, alpha, Saf, bem=False):
        """ generate lift and drag coefficient gradients with respect to airfoil shape parameters
        Parameters
        ----------
        alpha : float
            angle of attack (rad)
        Saf : array of length 2
            first element is thickness-to-chord ratio
            second element is blended airfoil family factor
        bem : bool, optional
            True - used for the BEM (Blade Element Momentum) method specific for CCBlade
            False - used elsewhere than the BEM method

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        """
        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            tc = Saf[0]
            bf = Saf[1]

            # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
            if bem:
                tck_cl0 = self.cl_total_spline0_bem.tck[:3] + self.cl_total_spline0_bem.degrees  # concatenate lists
                tck_cd0 = self.cd_total_spline0_bem.tck[:3] + self.cd_total_spline0_bem.degrees
                tck_cl1 = self.cl_total_spline1_bem.tck[:3] + self.cl_total_spline1_bem.degrees  # concatenate lists
                tck_cd1 = self.cd_total_spline1_bem.tck[:3] + self.cd_total_spline1_bem.degrees
            else:
                tck_cl0 = self.cl_total_spline0.tck[:3] + self.cl_total_spline0.degrees  # concatenate lists
                tck_cd0 = self.cd_total_spline0.tck[:3] + self.cd_total_spline0.degrees
                tck_cl1 = self.cl_total_spline1.tck[:3] + self.cl_total_spline1.degrees  # concatenate lists
                tck_cd1 = self.cd_total_spline1.tck[:3] + self.cd_total_spline1.degrees
        else:
            try:
                tc = Saf[0]
            except:
                tc = Saf
            if bem:
                tck_cl0 = self.cl_total_spline0_bem.tck[:3] + self.cl_total_spline0_bem.degrees  # concatenate lists
                tck_cd0 = self.cd_total_spline0_bem.tck[:3] + self.cd_total_spline0_bem.degrees
            else:
                tck_cl0 = self.cl_total_spline0.tck[:3] + self.cl_total_spline0.degrees  # concatenate lists
                tck_cd0 = self.cd_total_spline0.tck[:3] + self.cd_total_spline0.degrees

        dcl_dalpha0 = bisplev(alpha, tc, tck_cl0, dx=1, dy=0)
        dcd_dalpha0 = bisplev(alpha, tc, tck_cd0, dx=1, dy=0)

        dcl_dt_c0 = bisplev(alpha, tc, tck_cl0, dx=0, dy=1)
        dcd_dt_c0 = bisplev(alpha, tc, tck_cd0, dx=0, dy=1)

        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            dcl_dalpha1 = bisplev(alpha, tc, tck_cl1, dx=1, dy=0)
            dcd_dalpha1 = bisplev(alpha, tc, tck_cd1, dx=1, dy=0)

            dcl_dt_c1 = bisplev(alpha, tc, tck_cl1, dx=0, dy=1)
            dcd_dt_c1 = bisplev(alpha, tc, tck_cd1, dx=0, dy=1)

            dcl_dalpha = dcl_dalpha0 + bf*(dcl_dalpha1-dcl_dalpha0)
            dcd_dalpha = dcd_dalpha0 + bf*(dcd_dalpha1-dcd_dalpha0)

            dcl_dtc = dcl_dt_c0 + bf*(dcl_dt_c1-dcl_dt_c0)
            dcd_dtc = dcd_dt_c0 + bf*(dcd_dt_c1-dcd_dt_c0)

            dcl_dweight = self.cl1-self.cl0
            dcd_dweight = self.cd1-self.cd0
        else:
            dcl_dalpha = dcl_dalpha0
            dcd_dalpha = dcd_dalpha0
            dcl_dtc = dcl_dt_c0
            dcd_dtc = dcd_dt_c0

        if self.afOptions['PrecomputationalOptions']['AirfoilParameterization'] == 'Blended' and self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            dcl_dSaf = np.asarray([dcl_dtc, dcl_dweight])
            dcd_dSaf = np.asarray([dcd_dtc, dcd_dweight])
        else:
            dcl_dSaf = np.asarray([dcl_dtc])
            dcd_dSaf = np.asarray([dcd_dtc])

        return dcl_dalpha, dcl_dSaf, dcd_dalpha, dcd_dSaf

    def plotPreCompModel(self, splineNum=0, bem=False):
        """ plot the surrogate model
        Parameters
        ----------
        splineNum : int
            Either 0 or 1 corresponding to the first or second airfoil family
        bem : bool, optional
            True - used for the BEM (Blade Element Momentum) method specific for CCBlade
            False - used elsewhere than the BEM method

        Returns
        -------
        None

        """
        import matplotlib.pylab as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        n = 200
        thick = np.linspace(self.thick_min, self.thick_max, n)
        alpha = np.linspace(-np.pi, np.pi, n)
        CL = np.zeros((n, n))
        CD = np.zeros((n, n))
        [X, Y] = np.meshgrid(alpha, thick)
        for i in list(range(n)):
            for j in list(range(n)):
                if splineNum == 0:
                    if not bem:
                        CL[i, j] = self.cl_total_spline0.ev(X[i, j], Y[i, j])
                        CD[i, j] = self.cd_total_spline0.ev(X[i, j], Y[i, j])
                    else:
                        CL[i, j] = self.cl_total_spline0_bem.ev(X[i, j], Y[i, j])
                        CD[i, j] = self.cd_total_spline0_bem.ev(X[i, j], Y[i, j])
                else:
                    if not bem:
                        CL[i, j] = self.cl_total_spline1.ev(X[i, j], Y[i, j])
                        CD[i, j] = self.cd_total_spline1.ev(X[i, j], Y[i, j])
                    else:
                        CL[i, j] = self.cl_total_spline1_bem.ev(X[i, j], Y[i, j])
                        CD[i, j] = self.cd_total_spline1_bem.ev(X[i, j], Y[i, j])
        font_size = 14
        fig4 = plt.figure()
        ax4 = fig4.gca(projection='3d')
        surf = ax4.plot_surface(np.degrees(X), Y, CD, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        plt.xlim(xmin=-180, xmax=180)
        plt.xticks(np.arange(-180, 180+1, 60.0))
        plt.yticks(np.arange(0.15, 0.46, 0.10))
        ax4.set_zlabel(r'$c_d$')
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel('t/c (\%)')
        fig4.colorbar(surf)
        #plt.savefig('cd_fin_surface.pdf')
        #plt.savefig('cd_fin_surface.png')

        fig5 = plt.figure()
        ax5 = fig5.gca(projection='3d')
        surf2 = ax5.plot_surface(np.degrees(X), Y, CL, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        fig5.colorbar(surf2)
        plt.xlim(xmin=-180, xmax=180)
        plt.xticks(np.arange(-180, 180+1, 60.0))
        plt.yticks(np.arange(0.15, 0.46, 0.10))
        ax5.set_zlabel(r'$c_l$')
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel('t/c (\%)')
        #plt.savefig('cl_fin_surface.pdf')
        #plt.savefig('cl_fin_surface.png')

        plt.show()

    def __generatePreCompModel(self):
        """ Generates the precomputational models
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.cl_total_spline0, self.cd_total_spline0, self.xx0, self.yy0, self.thicknesses0 = self.__generatePreCompSpline(0)

        if self.afOptions['AnalysisMethod'] == self.afOptions['SplineOptions']['AnalysisMethod']:
            self.cl_total_spline0_bem, self.cd_total_spline0_bem = copy.deepcopy(self.cl_total_spline0), copy.deepcopy(self.cd_total_spline0)
        else:
            self.cl_total_spline0_bem, self.cd_total_spline0_bem, _, _, _ = self.__generatePreCompSpline(0, bem=True)

        if self.afOptions['PrecomputationalOptions']['numAirfoilFamilies'] > 1:
            self.cl_total_spline1, self.cd_total_spline1, self.xx1, self.yy1, self.thicknesses1 = self.__generatePreCompSpline(1)
            if self.afOptions['AnalysisMethod'] == self.afOptions['SplineOptions']['AnalysisMethod']:
                self.cl_total_spline1_bem, self.cd_total_spline1_bem = copy.deepcopy(self.cl_total_spline1), copy.deepcopy(self.cd_total_spline1)
            else:
                self.cl_total_spline1_bem, self.cd_total_spline1_bem, _, _, _ = self.__generatePreCompSpline(1, bem=True)
    def __generatePreCompCoordinates(self, splineNum):
        """ Generate the precomputational model airfoil coordinates
        Parameters
        ----------
        splineNum : int
            Either 0 or 1 corresponding to the first or second airfoil family

        Returns
        -------
        xx : array
            Series of x-coordinates for the airfoil family
        yy : array
            Series of y-coordinates for the airfoil family
        thicknesses: array
            Series of thickness-to-chord ratios for the airfoil family
        """
        n = self.n
        try:
            airfoilsSpecified = self.afOptions['PrecomputationalOptions']['BaseAirfoilsCoordinates'+str(splineNum)]
        except:
            raise ValueError('Please specify base airfoil coordinates in afOptions')

        xs, ys, airfoil_thicknesses = [], [], []

        for i in list(range(len(airfoilsSpecified))):
            x, y = self.getCoordinates(airfoilsSpecified[i], airfoilShapeMethod=self.afOptions['PrecomputationalOptions']['airfoilShapeMethod']) #readCoordinateFile(airfoilsSpecified[i])]
            airfoil_thickness = max(y) - min(y)
            if airfoil_thickness > self.thick_max or airfoil_thickness < self.thick_min:
                raise ValueError('Provided Airfoil %i of family %i surpassed bounds at thickness-to-chord ratio of %f. Please change PrecomputationalOption: tcMin or tcMax' % (i+1, splineNum, airfoil_thickness))
            xs.append(x), ys.append(y), airfoil_thicknesses.append(airfoil_thickness)
        self.airfoilsSpecified = copy.deepcopy(airfoil_thicknesses)
        yx = list(zip(airfoil_thicknesses,xs))
        yx.sort()
        x_sorted = [x for y, x in yx]
        yx = list(zip(airfoil_thicknesses,ys))
        yx.sort()
        y_sorted = [x for y, x in yx]
        airfoil_thicknesses.sort()
        # Calculate thicknesses just past min and max because gradient at edges are zero
        thicknesses = [self.thick_min-1e-3] + airfoil_thicknesses + [self.thick_max+1e-3]
        yy = [self.__convertTCCoordinates(self.thick_min-1e-3, y_sorted[0])] + y_sorted + [self.__convertTCCoordinates(self.thick_max+1e-3, y_sorted[-1])]
        xx = [x_sorted[0]] + x_sorted + [x_sorted[-1]]
        airfoils_to_add = n - len(thicknesses)
        if airfoils_to_add > 0:
            to_insert = np.linspace(self.thick_min, self.thick_max, 2+airfoils_to_add)
            for j in list(range(len(to_insert)-2)):
                alreadyFound = False
                for k in list(range(len(thicknesses))):
                    if to_insert[j+1] >= thicknesses[k] and to_insert[j+1] <= thicknesses[k+1] and not alreadyFound:
                        thicknesses.insert(k+1, to_insert[j+1])
                        yy.insert(k+1, self.__convertTCCoordinates(to_insert[j+1], yy[k]))
                        xx.insert(k+1, xx[k])
                        alreadyFound = True
        return xx, yy, thicknesses

    def __generatePreCompSpline(self, splineNum, bem=False):
        """ generate the spline for the precomputational method
        Parameters
        ----------
        splineNum : int
            Either 0 or 1 corresponding to the first or second airfoil family
        bem : bool, optional
            True - used for the BEM (Blade Element Momentum) method specific for CCBlade
            False - used elsewhere than the BEM method

        Returns
        -------
        None

        """
        xx, yy, thicknesses = self.__generatePreCompCoordinates(splineNum)
        cls, cds, cms, alphass, failures = [], [], [], [], []
        from airfoilprep import Airfoil, Polar
        from akima import Akima
        alphas_set = np.linspace(np.radians(-180), np.radians(180), 360)
        clGrid = np.zeros((len(alphas_set), len(thicknesses)))
        cdGrid = np.zeros((len(alphas_set), len(thicknesses)))
        if self.afOptions['AnalysisMethod'] == 'Files' and self.n > len(self.airfoilsSpecified):
            computeCorrection = True
            compute = True
        elif self.afOptions['AnalysisMethod'] != 'Files':
            computeCorrection = False
            compute = True
        else:
            computeCorrection = False
            compute = False

        if compute:
            for i in list(range(len(thicknesses))):
                self.x, self.y = xx[i], yy[i]
                if self.afOptions['AnalysisMethod'] == 'Data' and not bem:
                    import precomp_data
                    if splineNum == 0:
                        cl, cd, alphas, failure = precomp_data.cl0_cfd[i], precomp_data.cd0_cfd[i], precomp_data.alphas_cfd, False
                    else:
                        cl, cd, alphas, failure = precomp_data.cl1_cfd[i], precomp_data.cd1_cfd[i], precomp_data.alphas_cfd, False
                    cm = np.zeros_like(cl)
                else:
                    cl, cd, cm, alphas, failure = self.computeSpline(bem)

                p1 = Polar(self.afOptions['SplineOptions']['Re'], alphas, cl, cd, cm)
                af = Airfoil([p1])
                if self.afOptions['SplineOptions']['correction3D']:
                    af = af.correction3D(self.afOptions['SplineOptions']['r_over_R'], self.afOptions['SplineOptions']['chord_over_r'], self.afOptions['SplineOptions']['tsr'])
                af_extrap = af.extrapolate(self.afOptions['SplineOptions']['cd_max'])
                alpha_ext, Re_ext, cl_ext, cd_ext, cm_ext = af_extrap.createDataGrid()
                if not all(np.diff(alpha_ext)):
                    to_delete = np.zeros(0)
                    diff = np.diff(alpha_ext)
                    for z in list(range(len(alpha_ext)-1)):
                        if not diff[z] > 0.0:
                            to_delete = np.append(to_delete, z)
                    alpha_ext = np.delete(alpha_ext, to_delete)
                    cl_ext = np.delete(cl_ext, to_delete)
                    cd_ext = np.delete(cd_ext, to_delete)
                cls.append(cl_ext), cds.append(cd_ext), cms.append(cm_ext), alphass.append(alpha_ext), failures.append(failure)

        # Do XFOIL correction on file inputs if there is not enough data
        cl_correction = np.zeros(len(alphas_set))
        cd_correction = np.zeros(len(alphas_set))
        cls_files, cds_files, cms_files, alphass_files, failures_files = [], [], [], [], []
        if computeCorrection:
            for i in list(range(len(self.airfoilsSpecified))):
                aerodynFile = self.afOptions['PrecomputationalOptions']['BaseAirfoilsData'+str(splineNum)][i]
                af = Airfoil.initFromAerodynFile(aerodynFile)
                alpha_ext, Re_ext, cl_ext, cd_ext, cm_ext = af.createDataGrid()
                failure = False
                index = thicknesses.index(self.airfoilsSpecified[i])
                if not all(np.diff(alpha_ext)):
                    to_delete = np.zeros(0)
                    diff = np.diff(alpha_ext)
                    for z in list(range(len(alpha_ext)-1)):
                        if not diff[z] > 0.0:
                            to_delete = np.append(to_delete, z)
                    alpha_ext = np.delete(alpha_ext, to_delete)
                    cl_ext = np.delete(cl_ext, to_delete)
                    cd_ext = np.delete(cd_ext, to_delete)
                cls_files.append(cl_ext), cds_files.append(cd_ext), cms_files.append(cm_ext), alphass_files.append(alpha_ext), failures_files.append(failure)

                cl_spline_xfoil = Akima(np.radians(alphass[index]), cls[index], delta_x=0)
                cl_set_xfoil, _ = cl_spline_xfoil.interp(alphas_set)
                cd_spline_xfoil = Akima(np.radians(alphass[index]), cds[index], delta_x=0)
                cd_set_xfoil, _, = cd_spline_xfoil.interp(alphas_set)
                cl_spline_files = Akima(np.radians(alpha_ext), cl_ext, delta_x=0)
                cl_set_files, _, = cl_spline_files.interp(alphas_set)
                cd_spline_files = Akima(np.radians(alpha_ext), cd_ext, delta_x=0)
                cd_set_files, _, = cd_spline_files.interp(alphas_set)
                for k in list(range(len(alphas_set))):
                    cl_correction[k] += cl_set_files[k] - cl_set_xfoil[k]
                    cd_correction[k] += cd_set_files[k] - cd_set_xfoil[k]
            cl_correction /= float(len(self.airfoilsSpecified))
            cd_correction /= float(len(self.airfoilsSpecified))

        for i in list(range(len(thicknesses))):
            if not all(np.diff(alphass[i])):
                to_delete = np.zeros(0)
                diff = np.diff(alphass[i])
                for z in list(range(len(alphass[i])-1)):
                    if not diff[z] > 0.0:
                        to_delete = np.append(to_delete, z)
                alphass[i] = np.delete(alphass[i], to_delete)
                cls[i] = np.delete(cls[i], to_delete)
                cds[i] = np.delete(cds[i], to_delete)
            cl_spline = Akima(np.radians(alphass[i]), cls[i])
            cd_spline = Akima(np.radians(alphass[i]), cds[i])

            cl_set, _, _, _ = cl_spline.interp(alphas_set)
            cd_set, _, _, _ = cd_spline.interp(alphas_set)
            # if computeCorrection:
            #     for w in list(range(len(cl_set))):
            #         cl_set[w] += cl_correction[w]
            #         cd_set[w] += cd_correction[w]
            #         if cd_set[w] < 0.0001:
            #             cd_set[w] = 0.001
            if thicknesses[i] in self.airfoilsSpecified and self.afOptions['AnalysisMethod'] == 'Files':
                index = self.airfoilsSpecified.index(thicknesses[i])
                cl_spline = Akima(np.radians(alphass_files[index]), cls_files[index])
                cd_spline = Akima(np.radians(alphass_files[index]), cds_files[index])

                cl_set, _, _, _ = cl_spline.interp(alphas_set)
                cd_set, _, _, _ = cd_spline.interp(alphas_set)

            for j in list(range(len(alphas_set))):
                clGrid[j][i] = cl_set[j]
                cdGrid[j][i] = cd_set[j]
        kx = min(len(alphas_set)-1, 3)
        ky = min(len(thicknesses)-1, 3)

        cl_total_spline = RectBivariateSpline(alphas_set, thicknesses, clGrid, kx=kx, ky=ky)#, s=0.001)
        cd_total_spline = RectBivariateSpline(alphas_set, thicknesses, cdGrid, kx=kx, ky=ky)#, s=0.0005)

        return cl_total_spline, cd_total_spline, xx, yy, thicknesses

    def __convertTCCoordinates(self, tc, y):
        """ given the y-coordinates, return the new y-coordinates at the new thickness-to-chord ratio
        Parameters
        ----------
        tc : float
            thickness-to-chord ratio
        y : array
            y-coordinates

        Returns
        -------
        yy : array
            new y-coordinates at the specified thickness-to-chord ratio
        """
        yy = np.zeros(len(y))
        base_tc = max(y) - min(y)
        for i in list(range(len(y))):
            yy[i] = y[i] * tc / base_tc
        return yy

    ##########################################
    #        AIRFOIL ANALYSIS METHODS        #
    ##########################################

    def computeSpline(self, alphas=None, Re=None, bem=False):
        """
        Parameters
        ----------
        bem : bool, optional
            True - used for the BEM (Blade Element Momentum) method specific for CCBlade
            False - used elsewhere than the BEM method

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        cm : array
            pitching moment coefficients
        alphas : array
            angles of attack (deg)
        failure : array
            if any failures during analysis
        """
        if bem:
            airfoilAnalysis = self.af_spline_analysis
        else:
            airfoilAnalysis = self.af_analysis

        if self.af_parameterization == 'Precomputational' and self.generatedPreComp:
            alphas = self.afOptions['SplineOptions']['alphas']
            cl, cd = np.zeros(len(alphas)), np.zeros(len(alphas))
            for i in list(range(len(alphas))):
                cl[i], cd[i] = self.evaluatePreCompModel(np.radians(alphas[i]), self.Saf)
            cm, alphas, failure = np.zeros_like(cl), alphas, False
        else:
            if airfoilAnalysis == 'CFD':
                cl, cd, cm, alphas, failure = self.__cfdSpline()
            else:
                cl, cd, cm, alphas, failure = self.__xfoilSpline()
            # error checking if CFD diverges
            if np.any(abs(cl) > 100):
                cl, cd, cm, alphas, failure  = self.__xfoilSpline()
        return cl, cd, cm, alphas, failure

    def computeDirect(self, alpha, Re):
        """
        Parameters
        ----------
        alpha : float
            angle of attack (rad)
        Re : float
            Reynolds number

        Returns
        -------
        cl : array
            lift coefficient
        cd : array
            drag coefficient

        """
        if self.af_parameterization == 'Precomputational':
            cl, cd = self.evaluatePreCompModel(alpha, self.Saf)
            dcl_dalpha, dcl_dSaf, dcd_dalpha, dcd_dSaf = self.derivativesPreCompModel(alpha, self.Saf)
            dcl_dRe, dcd_dRe = 0.0, 0.0  # Reynolds number gradients currently not provided in precomputational method
        else:
            if self.af_analysis == 'CFD':
                cl, cd, dcl_dalpha, dcd_dalpha, dcl_dSaf, dcd_dSaf = self.__cfdSerial(alpha, Re, GenerateMESH=True)
                dcl_dRe, dcd_dRe = 0.0, 0.0  # Reynolds number gradients currently not provided in CFD
            else:
                cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf = self.__xfoilDirect(alpha, Re)


        if self.afOptions['GradientOptions']['ComputeGradient']:
            return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf
        else:
            return cl, cd

    def evaluateDirectParallel(self, alphas, Res, afs):
        """
        Parameters
        ----------
        alphas : array
            angles of attack (rad)
        Res : array
            Reynolds numbers
        afs : array
            Airfoil objects

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        dcl_dalpha : array
            lift gradients with respect to alpha
        dcl_dRe : array
            lift gradients with respect to Re
        dcd_dalpha : array
            drag gradients with respect to alpha
        dcd_dRe : array
            drag gradients with respect to Re
        dcl_dSaf : array
            lift gradients with respect to airfoil shape parameters
        dcd_dSaf : array
            drag gradients with respect to airfoil shape parameters

        """
        indices_to_compute = []
        n = len(alphas)
        afOptions = afs[-1].afOptions
        cl = np.zeros(n)
        cd = np.zeros(n)
        dcl_dalpha = [0]*n
        dcd_dalpha = [0]*n
        dcl_dSaf = [0]*n
        dcd_dSaf = [0]*n
        dcl_dRe = [0]*n
        dcd_dRe = [0]*n
        computeAlphaGradient = self.afOptions['GradientOptions']['ComputeGradient']
        computeSafGradient = self.afOptions['GradientOptions']['ComputeAirfoilGradients']
        for i in list(range(len(alphas))):
            alpha = alphas[i]
            Re = Res[i]
            af = afs[i]
            if af.Saf is not None and abs(np.degrees(alpha)) < af.afOptions['SplineOptions']['maxDirectAoA']:
                found = False
                for b in list(range(len(af.alpha_storage))):
                    if abs(alpha - af.alpha_storage[b]) < 1e-10:
                    # index = af.alpha_storage.index(alpha)
                        cl[i] = af.cl_storage[b]
                        cd[i] = af.cd_storage[b]
                        found = True
                        if computeAlphaGradient:
                            # index = af.dalpha_storage.index(alpha)
                            dcl_dalpha[i] = af.dcl_storage[b]
                            dcd_dalpha[i] = af.dcd_storage[b]
                        if computeSafGradient:# and alpha in af.dalpha_dSaf_storage:
                            # index = af.dalpha_dSaf_storage.index(alpha)
                            dcl_dSaf[i] = af.dcl_dSaf_storage[b]
                            dcd_dSaf[i] = af.dcd_dSaf_storage[b]
                        dcl_dRe[i] = 0.0
                        dcd_dRe[i] = 0.0
                if not found:
                    indices_to_compute.append(i)
            else:
                cl[i] = af.cl_spline.ev(alpha, Re)
                cd[i] = af.cd_spline.ev(alpha, Re)
                tck_cl = af.cl_spline.tck[:3] + af.cl_spline.degrees  # concatenate lists
                tck_cd = af.cd_spline.tck[:3] + af.cd_spline.degrees

                dcl_dalpha[i] = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
                dcd_dalpha[i] = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

                if af.one_Re:
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    dcl_dRe[i] = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                    dcd_dRe[i] = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
                if computeSafGradient and af.Saf is not None:
                    dcl_dSaf[i], dcd_dSaf[i] = af.splineFreeFormGrad(alpha, Re)
                else:
                    dcl_dSaf[i], dcd_dSaf[i] = np.zeros(8), np.zeros(8)
        if indices_to_compute:
            alphas_to_compute = [alphas[i] for i in indices_to_compute]
            Res_to_compute = [Res[i] for i in indices_to_compute]
            Safs_to_compute = [afs[i].Saf for i in indices_to_compute]
            if afOptions['GradientOptions']['ComputeGradient']:
                cls, cds, dcls_dalpha, dcls_dRe, dcds_dalpha, dcds_dRe, dcls_dSaf, dcds_dSaf = self.__cfdDirectParallel(alphas_to_compute, Res_to_compute, Safs_to_compute, afOptions)
                for j in list(range(len(indices_to_compute))):
                    dcl_dalpha[indices_to_compute[j]] = dcls_dalpha[j]
                    dcl_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcd_dalpha[indices_to_compute[j]] = dcds_dalpha[j]
                    dcd_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcl_dSaf[indices_to_compute[j]] = dcls_dSaf[j]
                    dcd_dSaf[indices_to_compute[j]] = dcds_dSaf[j]
            else:
                cls, cds = self.__cfdDirectParallel(alphas_to_compute, Res_to_compute, Safs_to_compute, afOptions)
            for j in list(range(len(indices_to_compute))):
                cl[indices_to_compute[j]] = cls[j]
                cd[indices_to_compute[j]] = cds[j]

        for i in list(range(len(alphas))):
            if afs[i].Saf is not None and abs(np.degrees(alphas[i])) < afs[i].afOptions['SplineOptions']['maxDirectAoA']:
                afs[i].alpha_storage.append(alphas[i])
                afs[i].cl_storage.append(cl[i])
                afs[i].cd_storage.append(cd[i])
                afs[i].dcl_storage.append(dcl_dalpha[i])
                afs[i].dcd_storage.append(dcd_dalpha[i])
                afs[i].dcl_dSaf_storage.append(dcl_dSaf[i])
                afs[i].dcd_dSaf_storage.append(dcd_dSaf[i])

        if computeSafGradient:
            if computeAlphaGradient:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dSaf, dcd_dSaf
            else:
                return cl, cd
        elif computeAlphaGradient:
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        else:
            return cl, cd

    ##########################################
    #         XFOIL ANALYSIS METHODS         #
    ##########################################
    def __xfoilSpline(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        cm : array
            pitching moment coefficients
        alphas : array
            angles of attack (deg)
        failure : array
            if any failures during analysis

        """
        airfoilShapeFile = 'airfoil_shape.dat'
        self.saveCoordinateFile(airfoilShapeFile)
        alphas, Re = self.afOptions['SplineOptions']['alphas'], self.afOptions['SplineOptions']['Re']
        airfoil = pyXLIGHT.xfoilAnalysis(airfoilShapeFile, x=self.x, y=self.y)
        airfoil.re, airfoil.mach, airfoil.iter = Re, 0.0, 100
        cl, cd, cm, to_delete = np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(0)
        failure = False
        for j in list(range(len(alphas))):
            cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(alphas[j])
            if lexitflag:
                cl[j], cd[j] = -10.0, 0.0

        # Make sure none of the values are too far outliers
        cl_diff = np.diff(np.asarray(cl))
        cd_diff = np.diff(np.asarray(cd))
        for zz in list(range(len(cl_diff))):
            if abs(cd_diff[zz]) > 0.02 or abs(cl_diff[zz]) > 0.5:
                to_delete = np.append(to_delete, zz)

        # error handling in case of XFOIL failure
        for k in list(range(len(cl))):
            if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 1.0 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                to_delete = np.append(to_delete, k)

        cl, cd, cm = np.delete(cl, to_delete), np.delete(cd, to_delete), np.delete(cm, to_delete)

        if not cl.size or len(cl) < 3 or max(cl) < 0.0:
            print("XFOIL Failure! Using default backup airfoil.") # for CST = [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]
            cl = [-1.11249573, -1.10745928, -1.10242437, -1.10521061, -1.03248528, -0.9272929, -0.81920516, -0.70843745, -0.58962942, -0.45297636, -0.34881162, -0.26194, -0.17375163, -0.09322158, -0.01072867,  0.07232111,
                  0.15326737,  0.22932199, 0.29657574,  0.36818004,  0.45169576,  0.55640456 , 0.68532189,  0.81592085, 0.93355555,  1.04754944,  1.06513144,  1.07821432 , 1.09664777,  1.11425611]
            cd = [ 0.03966997,  0.03289554,  0.02783541,  0.02418726,  0.02120267,  0.01849611,  0.01623273,  0.01424686,  0.0124225 ,  0.01083306,  0.00973778,  0.00908278, 0.00867001,  0.00838171,  0.00823596,  0.00820681,
                   0.00828496 , 0.00842328,  0.00867177,  0.00921659,  0.01004469,  0.01129231,  0.01306175 , 0.01509252, 0.01731396,  0.01986422,  0.02234169 , 0.02555122,  0.02999641 , 0.03574208]
            cm = np.zeros(len(cl))
            alphas = np.linspace(-15, 15, len(cl))
            failure = True
        else:
            alphas = np.delete(alphas, to_delete)
        return cl, cd, cm, alphas, failure

    def __xfoilDirect(self, alpha, Re):
        """
        Parameters
        ----------
        alpha : float
            angles of attack (rad)
        Re : float
            Reynolds numbers

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        dcl_dalpha : array
            lift gradients with respect to alpha
        dcd_dalpha : array
            drag gradients with respect to alpha
        dcl_dSaf : array
            lift gradients with respect to airfoil shape parameters
        dcd_dSaf : array
            drag gradients with respect to airfoil shape parameters
        lexitflag : array
            if analysis failed or not
        """

        airfoil_shape_file = None
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=self.x, y=self.y)
        airfoil.re = Re
        airfoil.mach = 0.00
        airfoil.iter = 100
        dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf = 0.0, 0.0, 0.0, 0.0, np.zeros(self.af_dof), np.zeros(self.af_dof)
        if self.afOptions['GradientOptions']['ComputeGradient']:
            cs_step = 1e-20
            angle = complex(np.degrees(alpha), cs_step)
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            dcl_dalpha, dcd_dalpha = 180.0/np.pi*np.imag(copy.deepcopy(cl))/ cs_step, 180.0/np.pi*np.imag(copy.deepcopy(cd)) / cs_step
            airfoil.re = complex(Re, cs_step)
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(np.degrees(alpha))
            dcl_dRe, dcd_dRe = np.imag(copy.deepcopy(cl))/ cs_step, np.imag(copy.deepcopy(cd)) / cs_step
            cl, cd = np.real(np.asscalar(cl)), np.real(np.asscalar(cd))

            # if complex-step does not converge, do finite difference instead
            if abs(dcl_dalpha) > 20.0 or abs(dcd_dalpha) > 20.0:
                fd_step = self.afOptions['GradientOptions']['fd_step']
                cl, cd, cm, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
                cl, cd = np.asscalar(cl), np.asscalar(cd)
                angle2 = np.degrees(alpha + fd_step)
                cl2, cd2, cm2, lexitflag = airfoil.solveAlpha(angle2)
                cl2, cd2 = np.asscalar(cl2), np.asscalar(cd2)
                dcl_dalpha, dcd_dalpha = (cl2-cl)/ fd_step, (cd2-cd)/ fd_step
                airfoil.re = Re + fd_step * 1e6 # relative step size
                cl3, cd3, cm3, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
                cl3, cd3 = np.asscalar(cl3), np.asscalar(cd3)
                dcl_dRe, dcd_dRe = (cl3-cl)/ fd_step, (cd3-cd)/ fd_step
            if self.afOptions['GradientOptions']['ComputeAirfoilGradients']:
                if self.af_parameterization == 'CST':
                    dcl_dSaf, dcd_dSaf = self.__xfoilGradients(alpha, Re)
                if np.any(abs(dcl_dSaf)) > 100.0 or np.any(abs(dcd_dSaf) > 100.0):
                    print("Error in complex step splines")
        else:
            cl, cd, cm, lexitflag = airfoil.solveAlpha(np.degrees(alpha))
            cl, cd = np.asscalar(cl), np.asscalar(cd)
        return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dRe, dcd_dRe, dcl_dSaf, dcd_dSaf

    def __xfoilGradients(self, alpha, Re):
        """
        Parameters
        ----------
        alpha : float
            angles of attack (rad)
        Re : float
            Reynolds numbers

        Returns
        -------
        dcl_dSaf : array
            lift gradients with respect to airfoil shape parameters
        dcd_dSaf : array
            drag gradients with respect to airfoil shape parameters

        """
        alpha = np.degrees(alpha)
        wl, wu, N, dz = self.wl, self.wu, self.numCoordinates, self.dz
        nn = len(wl)+len(wu)
        step_size = 1e-20
        cs_step = complex(0, step_size)
        dcl_dSaf, dcd_dSaf = np.zeros(nn), np.zeros(nn)
        lexitflag = np.zeros(nn)
        for i in list(range(len(wl))):
            wl_complex = copy.deepcopy(wl.astype(complex))
            wu_complex = copy.deepcopy(wu.astype(complex))
            wl_complex[i] += cs_step
            cl_complex, cd_complex, lexitflag[i] = self.__xfoilSolveComplex(alpha, wl_complex, wu_complex, N, dz)
            dcl_dSaf[i], dcd_dSaf[i] = np.imag(cl_complex)/step_size, np.imag(cd_complex)/step_size
            wl_complex = copy.deepcopy(wl.astype(complex))
            wu_complex = copy.deepcopy(wu.astype(complex))
            wu_complex[i] += cs_step
            cl_complex, cd_complex, lexitflag[i+nn/2] = self.__xfoilSolveComplex(alpha, wl_complex, wu_complex, N, dz)
            dcl_dSaf[i+nn/2], dcd_dSaf[i+nn/2] = np.imag(cl_complex)/step_size, np.imag(cd_complex)/step_size
            if lexitflag[i] or lexitflag[i+nn/2] or abs(dcl_dSaf[i+nn/2]) > 100.0 or abs(dcd_dSaf[i+nn/2]) > 100.0 or abs(dcl_dSaf[i]) > 100.0 or abs(dcd_dSaf[i]) > 100.0:
                fd_step = 1.e-6 #self.afOptions['GradientOptions']['fd_step']
                wl_fd1 = np.real(copy.deepcopy(wl))
                wl_fd2 = np.real(copy.deepcopy(wl))
                wl_fd1[i] -= 0.0#fd_step
                wl_fd2[i] += fd_step
                cl_fd1, cd_fd1, flag1 = self.__xfoilSolveReal(alpha, wl_fd1, wu, N, dz)
                cl_fd2, cd_fd2, flag2 = self.__xfoilSolveReal(alpha, wl_fd2, wu, N, dz)
                lexitflag[i] = np.logical_or(flag1, flag2)
                dcl_dSaf[i] = (cl_fd2 - cl_fd1)/fd_step #(2.*fd_step)
                dcd_dSaf[i] = (cd_fd2 - cd_fd1)/fd_step #(2.*fd_step)
                wu_fd1 = np.real(copy.deepcopy(wu))
                wu_fd2 = np.real(copy.deepcopy(wu))
                wu_fd1[i] -= 0.0#fd_step
                wu_fd2[i] += fd_step
                cl_fd1, cd_fd1, flag1 = self.__xfoilSolveReal(alpha, wl, wu_fd1, N, dz)
                cl_fd2, cd_fd2, flag2 = self.__xfoilSolveReal(alpha, wl, wu_fd2, N, dz)
                lexitflag[i+nn/2] = np.logical_or(flag1, flag2)
                dcl_dSaf[i+nn/2] = (cl_fd2 - cl_fd1)/fd_step #(2.*fd_step)
                dcd_dSaf[i+nn/2] = (cd_fd2 - cd_fd1)/fd_step #(2.*fd_step)
                if lexitflag[i] or lexitflag[i+nn/2] or abs(dcl_dSaf[i+nn/2]) > 100.0 or abs(dcd_dSaf[i+nn/2]) > 100.0 or abs(dcl_dSaf[i]) > 100.0 or abs(dcd_dSaf[i]) > 100.0:
                    ## GENERATE SPLINE
                    cl_cur = self.cl_spline.ev(np.radians(alpha), self.Re)
                    cd_cur = self.cd_spline.ev(np.radians(alpha), self.Re)
                    cl_new_fd = self.cl_splines_new[i].ev(np.radians(alpha), self.Re)
                    cd_new_fd = self.cd_splines_new[i].ev(np.radians(alpha), self.Re)
                    dcl_dSaf[i] = (cl_new_fd - cl_cur) / fd_step
                    dcd_dSaf[i] = (cd_new_fd - cd_cur) / fd_step
                    cl_new_fd = self.cl_splines_new[i+nn/2].ev(np.radians(alpha), self.Re)
                    cd_new_fd = self.cd_splines_new[i+nn/2].ev(np.radians(alpha), self.Re)
                    dcl_dSaf[i+nn/2] = (cl_new_fd - cl_cur) / fd_step
                    dcd_dSaf[i+nn/2] = (cd_new_fd - cd_cur) / fd_step
                #print "derivative CST fail", alpha
        for i in list(range(nn)):
            if lexitflag[i]:
                from akima import Akima
                af1 = Airfoil.initFromCST(self.Saf, self.afOptions)
                af_extrap11 = af1.extrapolate(1.5)
                alphas_cur, Re_cur, cl_cur, cd_cur, cm_cur = af_extrap11.createDataGrid()
                cl_spline_cur = Akima(alphas_cur, cl_cur)
                cd_spline_cur = Akima(alphas_cur, cd_cur)
                cl_fd_cur, _, _, _ = cl_spline_cur.interp(alpha)
                cd_fd_cur, _, _, _ = cd_spline_cur.interp(alpha)
                Saf_new = copy.deepcopy(self.Saf)
                Saf_new[i] += fd_step
                af = Airfoil.initFromCST(Saf_new, self.afOptions)
                af_extrap1 = af.extrapolate(1.5)
                alphas_new, Re_new, cl_new, cd_new, cm_new = af_extrap1.createDataGrid()
                cl_spline = Akima(alphas_new, cl_new)
                cd_spline = Akima(alphas_new, cd_new)
                cl_fd_new, _, _, _ = cl_spline.interp(alpha)
                cd_fd_new, _, _, _ = cd_spline.interp(alpha)
                dcl_dSaf[i] = (cl_fd_new - cl_fd_cur) / fd_step
                dcd_dSaf[i] = (cd_fd_new - cd_fd_cur) / fd_step
        if self.switchedCST:
            dcl_dSaf = np.roll(dcl_dSaf, nn/2)
            dcd_dSaf = np.roll(dcd_dSaf, nn/2)
        return dcl_dSaf, dcd_dSaf

    def __xfoilSolveComplex(self, alpha, wl_complex, wu_complex, N, dz):
        """
        Parameters
        ----------
        wl_complex : array
            lower Kulfan parameters
        wu_complex : array
            upper Kulfan parameters
        N : int
            number of airfoil coordinates
        dz : float
            trailing edge thickness

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient
        lexitflag: bool
            failed or not
        """
        airfoil_shape_file = None

        x, y = self.__cstCoordinatesComplexFull(wl_complex, wu_complex, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.afOptions['SplineOptions']['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl_complex, cd_complex, cm_complex, lexitflag = airfoil.solveAlphaComplex(alpha)
        return copy.deepcopy(cl_complex), copy.deepcopy(cd_complex), copy.deepcopy(lexitflag)

    def __xfoilSolveReal(self, alpha, wl, wu, N, dz):
        """
        Parameters
        ----------
        wl : array
            lower Kulfan parameters
        wu : array
            upper Kulfan parameters
        N : int
            number of airfoil coordinates
        dz : float
            trailing edge thickness

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient
        lexitflat: bool
            failed or not
        """
        airfoil_shape_file = None
        x, y = self.__cstCoordinatesReal(wl, wu, N, dz)
        airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file, x=x, y=y)
        airfoil.re = self.afOptions['SplineOptions']['Re']
        airfoil.mach = 0.0
        airfoil.iter = 100
        cl, cd, cm, lexitflag = airfoil.solveAlpha(alpha)
        return np.asscalar(cl), np.asscalar(cd), copy.deepcopy(lexitflag)


    ##########################################
    #          CFD ANALYSIS METHODS          #
    ##########################################

    def __cfdSpline(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        cm : ndarray
            pitching moment coefficients
        alphas : ndarray
            angles of attack (rad)
        lexitflag: ndarray
            failed or not

        """
        alphas = self.afOptions['SplineOptions']['alphas']
        Re = self.afOptions['SplineOptions']['Re']
        cl, cd, cm, failure = np.zeros(len(alphas)), np.zeros(len(alphas)), np.zeros(len(alphas)), False
        if self.afOptions['CFDOptions']['computeAirfoilsInParallel']:
            cl, cd = self.__cfdParallelSpline(np.radians(alphas), Re, self.afOptions)
        else:
            for j in list(range(len(alphas))):
                if j == 0:
                    mesh = True
                else:
                    mesh = False
                cl[j], cd[j], dcl_dalpha, dcd_dalpha, dcl_dSaf, dcd_dSaf, lexitflag = self.__cfdSerial(np.radians(alphas[j]), Re, GenerateMESH=mesh)
        return cl, cd, cm, alphas, failure



    def __cfdSerial(self, alpha, Re, GenerateMESH=True, airfoilNum=0):
        """
        Parameters
        ----------
        alpha : float
            angles of attack (rad)
        Re : float
            Reynolds numbers
        GenerateMESH : bool, optional
            whether or not to generate the CFD mesh
        airfoilNum : int, optional
            the airfoil number

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        dcl_dalpha : array
            lift gradients with respect to alpha
        dcd_dalpha : array
            drag gradients with respect to alpha
        dcl_dSaf : array
            lift gradients with respect to airfoil shape parameters
        dcd_dSaf : array
            drag gradients with respect to airfoil shape parameters
        lexitflag : array
            if analysis failed or not

        """
        config, state = self.__cfdDirectSerial(alpha, Re, GenerateMESH, airfoilNum)
        cl, cd = state.FUNCTIONS['LIFT'], state.FUNCTIONS['DRAG']

        if self.afOptions['GradientOptions']['ComputeGradient'] and self.afOptions['GradientOptions']['ComputeAirfoilGradients']:
            konfig = copy.deepcopy(config)
            ztate = copy.deepcopy(state)
            dcd_dSaf, dcd_dalpha = self.__cfdGradientsSerial(konfig, ztate, 'DRAG')
            dcl_dSaf, dcl_dalpha = self.__cfdGradientsSerial(konfig, ztate, 'LIFT')
            if dcl_dalpha == 0.0 and dcd_dalpha == 0.0:
                # Inviscid does not provide alpha gradients so do finite-difference instead
                fd_step = self.afOptions['GradientOptions']['fd_step']
                config, state = self.__cfdDirectSerial(alpha+fd_step, Re, GenerateMESH, airfoilNum)
                cl2, cd2 = state.FUNCTIONS['LIFT'], state.FUNCTIONS['DRAG']
                dcl_dalpha, dcd_dalpha = (cl2-cl)/fd_step, (cd2-cd)/fd_step
        else:
            dcl_dalpha, dcd_dalpha, dcl_dSaf, dcd_dSaf, lexitflag = 0.0, 0.0, np.zeros(8), np.zeros(8), False
        return cl, cd, dcl_dalpha, dcd_dalpha, dcl_dSaf, dcd_dSaf

    def __generateMesh(self, meshFileName, config, state):
        """
        Parameters
        ----------
        meshFileName : str
            the name of mesh file
        config : dictionary
            CFD config
        state : dictionary
            CFD state

        Returns
        -------
        return_code : int
            0 means success, anything else means error

        """
        airfoilFile = basepath + os.path.sep + 'airfoil_shape.dat'
        self.saveCoordinateFile(airfoilFile)
        konfig = copy.deepcopy(config)
        konfig.VISUALIZE_DEFORMATION = 'NO'
        konfig.MESH_OUT_FILENAME = meshFileName
        konfig.DV_KIND = 'AIRFOIL'
        tempname = basepath + os.path.sep + 'config_DEF.cfg'
        konfig.dump(tempname)
        SU2_RUN = os.environ['SU2_RUN']
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = 'SU2_DEF ' + tempname
        the_Command = base_Command % the_Command
        sys.stdout.flush()
        cfd_mesh_output = open(basepath + os.path.sep + 'mesh_deformation_direct.txt', 'w')
        proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout= cfd_mesh_output    ,
                         stderr= cfd_mesh_output,
                         stdin=subprocess.PIPE)
        #proc.stderr.close()
        proc.stdin.write(airfoilFile+'\n')
        proc.stdin.write('Selig\n')
        proc.stdin.write('1.0\n')
        proc.stdin.write('No\n')
        proc.stdin.write('clockwise\n')
        proc.stdin.close()
        return_code = proc.wait()
        return return_code

    def __cfdDirectSerial(self, alpha, Re, GenerateMESH=True, airfoilNum=0, basepath=None):
        """
        Parameters
        ----------
        alpha : float
            angles of attack (rad)
        Re : float
            Reynolds numbers
        GenerateMESH : bool, optional
            whether or not to generate the CFD mesh
        airfoilNum : int, optional
            the airfoil number

        Returns
        -------
        config : dictionary
            CFD config
        state : dictionary
            CFD state

        """
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2
        afOptions = self.afOptions

        config_filename = basepath + os.path.sep + afOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = afOptions['CFDOptions']['processors']
        config.EXT_ITER    = afOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'CFD' + os.path.sep + 'mesh_AIRFOIL_serial.su2'
        config.MESH_FILENAME = basepath + os.path.sep + config.MESH_FILENAME

        if GenerateMESH:
            return_code = self.__generateMesh(meshFileName, config, state, basepath)
            restart = False
        else:
            restart = True
        restart = False
        config.MESH_FILENAME = meshFileName #'mesh_out.su2' # basepath + os.path.sep + 'mesh_AIRFOIL.su2'
        state.FILES.MESH = config.MESH_FILENAME
        config.AoA = np.degrees(alpha)
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        x_vel = Uinf * cos(alpha)
        y_vel = Uinf * sin(alpha)
        config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
        config.MACH_NUMBER = 0.15
        config.REYNOLDS_NUMBER = afOptions['SplineOptions']['Re']

        config.RESTART_SOL = 'NO'
        config.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'solution_flow_airfoil.dat'
        config.SOLUTION_ADJ_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'solution_adj_airfoil.dat'
        config.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'restart_flow_airfoil.dat'
        config.RESTART_ADJ_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'restart_adj_airfoil.dat'
        config.SURFACE_ADJ_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'surface_adjoint_airfoil'
        config.SURFACE_FLOW_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'surface_flow_airfoil'
        config.VOLUME_FLOW_FILENAME = basepath + os.path.sep + 'CFD' + os.path.sep + 'flow_airfoil'

        konfig = copy.deepcopy(config)
        konfig['MATH_PROBLEM']  = 'DIRECT'
        konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'
        tempname = basepath + os.path.sep + 'CFD' + os.path.sep + 'config_CFD_airfoil.cfg'
        konfig.dump(tempname)
        SU2_RUN = os.environ['SU2_RUN']
        sys.path.append( SU2_RUN )

        processes = konfig['NUMBER_PART']
        the_Command = 'SU2_CFD ' + tempname
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = base_Command % the_Command
        if konfig['NUMBER_PART'] > 0:
            mpi_Command = 'mpirun -n %i %s'
            the_Command = mpi_Command % (processes,the_Command)
        else:
            mpi_Command = ''
        if processes > 0:
            if not mpi_Command:
                raise RuntimeError('could not find an mpi interface')
        cfd_direct_output = open(basepath + os.path.sep + 'CFD' + os.path.sep + 'cfd_direct_output.txt', 'w')
        print(the_Command)
        sys.stdout.flush()
        proc = subprocess.Popen( the_Command, shell=True    ,
                     stdout=cfd_direct_output,
                     stderr=subprocess.PIPE,
                     stdin=subprocess.PIPE)
        proc.stderr.close()
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise ValueError('Error in CFD Direct. Error code: %c' % (return_code))

        konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
        plot_format      = konfig['OUTPUT_FORMAT']
        plot_extension   = SU2.io.get_extension(plot_format)
        history_filename = konfig['CONV_FILENAME'] + plot_extension
        special_cases    = SU2.io.get_specialCases(konfig)

        final_avg = config.get('ITER_AVERAGE_OBJ',0)
        aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg )
        config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
        info = SU2.io.State()
        info.FUNCTIONS.update( aerodynamics )
        state.update(info)

        return config, state


    def __cfdParallelSpline(self, alphas, Re):
        """
        Parameters
        ----------
        alphas : ndarray
            angles of attack (rad)
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : drag
            drag coefficient

        """
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')
        config_filename = basepath + os.path.sep + afOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = int(afOptions['CFDOptions']['processors']/ float(len(alphas)))
        remainder =  afOptions['CFDOptions']['processors'] % len(alphas) - 1
        if remainder <= 0 or config.NUMBER_PART == 0:
            remainder = 0
        config.MESH_FILENAME = basepath + os.path.sep + config.MESH_FILENAME
        config.EXT_ITER    = afOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        meshFileName = basepath + os.path.sep + 'mesh_AIRFOIL_spline_parallel.su2'
        config.CONSOLE = 'QUIET'

        return_code = self.__generateMesh(meshFileName, config, state, basepath)
        if return_code != 0:
            print("Error in mesh deformation.")

        config.MESH_FILENAME = meshFileName
        state.FILES.MESH = config.MESH_FILENAME
        Uinf = 10.0
        Ma = Uinf / 340.29  # Speed of sound at sea level
        config.MACH_NUMBER = 0.15 #Ma
        config.REYNOLDS_NUMBER = Re
        config.RESTART_SOL = 'NO'
        config.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_AIRFOIL_parallel.dat'

        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        alphas = np.degrees(alphas)
        procTotal = []
        konfigTotal = []
        for i in list(range(len(alphas))):
            x_vel = Uinf * cos(np.radians(alphas[i]))
            y_vel = Uinf * sin(np.radians(alphas[i]))
            config.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
            config.AoA = alphas[i]
            config.CONV_FILENAME = basepath + os.path.sep + 'history_'+str(int(alphas[i]))
            state = SU2.io.State(state)
            konfig = copy.deepcopy(config)
            konfig['MATH_PROBLEM']  = 'DIRECT'
            konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'
            tempname = basepath + os.path.sep + 'config_CFD'+str(int(alphas[i]))+'.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            mpi_Command = 'mpirun -n %i %s'
            if i <= remainder:
                processes = konfig['NUMBER_PART'] + 1
            else:
                processes = konfig['NUMBER_PART']
            the_Command = 'SU2_CFD ' + tempname
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError('could not find an mpi interface')
            the_Command = mpi_Command % (processes,the_Command)
            print(the_Command)
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=open(basepath + os.path.sep + 'cfd_spline'+str(i+1)+'.txt', 'w'),
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()
            procTotal.append(copy.deepcopy(proc))
            konfigTotal.append(copy.deepcopy(konfig))

        for i in list(range(len(alphas))):
            while procTotal[i].poll() is None:
                pass
            konfig = konfigTotal[i]
            konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            final_avg = config.get('ITER_AVERAGE_OBJ',0)
            aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg)
            config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
            info = SU2.io.State()
            info.FUNCTIONS.update( aerodynamics )
            state.update(info)

            cl[i], cd[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']
        return cl, cd

    def __cfdDirectParallel(self, alphas, Res, Safs):
        """
        Parameters
        ----------
        alphas : ndarray
            angles of attack (rad)
        Res : ndarray
            Reynolds nmber
        Safs : ndarray
            airfoil shape parameters

        Returns
        -------
        cl : array
            lift coefficients
        cd : array
            drag coefficients
        dcl_dalpha : array
            lift gradients with respect to alpha
        dcd_dalpha : array
            drag gradients with respect to alpha
        dcl_dSaf : array
            lift gradients with respect to airfoil shape parameters
        dcd_dSaf : array
            drag gradients with respect to airfoil shape parameters

        """
        # Import SU2
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AirfoilAnalysisFiles')

        config_filename = basepath + os.path.sep + afOptions['CFDOptions']['configFile']
        config = SU2.io.Config(config_filename)
        state  = SU2.io.State()
        config.NUMBER_PART = int(afOptions['CFDOptions']['processors'] / float(len(alphas)))
        remainder = afOptions['CFDOptions']['processors'] % len(alphas)
        config.EXT_ITER    = afOptions['CFDOptions']['iterations']
        config.WRT_CSV_SOL = 'YES'
        config.MESH_FILENAME = basepath + os.path.sep + config.MESH_FILENAME

        config.CONSOLE = 'QUIET'

        cl = np.zeros(len(alphas))
        cd = np.zeros(len(alphas))
        alphas = np.degrees(alphas)
        procTotal = []
        konfigTotal = []
        konfigDirectTotal = []
        ztateTotal = []
        Re = afOptions['SplineOptions']['Re']
        cores_count = 0
        host_count = 0
        for i in list(range(len(alphas))):
            meshFileName = basepath + os.path.sep + 'mesh_airfoil'+str(i+1)+'.su2'
            # Create airfoil coordinate file for SU2
            afanalysis = AirfoilAnalysis(Safs[i], afOptions)
            x, y = afanalysis.x, afanalysis.y
            airfoilFile = basepath + os.path.sep + 'airfoil'+str(i+1)+'_coordinates.dat'
            coord_file = open(airfoilFile, 'w')
            coord_file.write('Airfoil Parallel')
            for j in list(range(len(x))):
                coord_file.write('{:<10f}\t{:<10f}'.format(x[j], y[j]))
            coord_file.close()

            konfig = copy.deepcopy(config)
            ztate = copy.deepcopy(state)
            konfig.MESH_OUT_FILENAME = meshFileName
            konfig.DV_KIND = 'AIRFOIL'
            tempname = basepath + os.path.sep + 'config_DEF_direct.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            base_Command = os.path.join(SU2_RUN,'%s')
            the_Command = 'SU2_DEF ' + tempname
            the_Command = base_Command % the_Command
            sys.stdout.flush()
            proc = subprocess.Popen( the_Command, shell=True    ,
                             stdout=open(basepath + os.path.sep + 'mesh_deformation_airfoil'+str(i+1)+'.txt', 'w')      ,
                             stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.write(airfoilFile+'\n')
            proc.stdin.write('Selig\n')
            proc.stdin.write('1.0\n')
            proc.stdin.write('Yes\n')
            proc.stdin.write('clockwise\n')
            proc.stdin.close()
            return_code = proc.wait()

            restart = False

            konfig.MESH_FILENAME = meshFileName
            ztate.FILES.MESH = config.MESH_FILENAME
            Uinf = 10.0
            Ma = Uinf / 340.29  # Speed of sound at sea level
            konfig.MACH_NUMBER = 0.15 # Ma
            konfig.REYNOLDS_NUMBER = Re

            konfig.RESTART_SOL = 'NO'
            konfig.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'.dat'
            konfig.SOLUTION_ADJ_FILENAME = basepath + os.path.sep + 'solution_adj_airfoil'+str(i+1)+'.dat'
            konfig.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'restart_flow_airfoil'+str(i+1)+'.dat'
            konfig.RESTART_ADJ_FILENAME = basepath + os.path.sep + 'restart_adj_airfoil'+str(i+1)+'.dat'
            konfig.SURFACE_ADJ_FILENAME = basepath + os.path.sep + 'surface_adjoint_airfoil' + str(i+1)
            konfig.SURFACE_FLOW_FILENAME = basepath + os.path.sep + 'surface_flow_airfoil' + str(i+1)
            konfig.VOLUME_FLOW_FILENAME = basepath + os.path.sep + 'flow_airfoil' + str(i+1)
            konfig.VOLUME_ADJ_FILENAME = basepath + os.path.sep + 'adjoint_airfoil' + str(i+1)
            konfig.GRAD_OBJFUNC_FILENAME = basepath + os.path.sep + 'of_grad_airfoil' + str(i+1) + '.dat'


            x_vel = Uinf * cos(np.radians(alphas[i]))
            y_vel = Uinf * sin(np.radians(alphas[i]))
            konfig.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
            konfig.AoA = alphas[i]
            konfig.CONV_FILENAME = basepath + os.path.sep + 'history_airfoil'+str(i+1)
            #state = SU2.io.State(state)

            konfig_direct = copy.deepcopy(konfig)
            # setup direct problem
            konfig_direct['MATH_PROBLEM']  = 'DIRECT'
            konfig_direct['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'

            # Run Solution
            tempname = basepath + os.path.sep + 'config_CFD_airfoil'+str(i+1)+'.cfg'
            konfig_direct.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            slurm_job = os.environ.has_key('SLURM_JOBID')
            if slurm_job:
                hosts, all_nodes = self.__slurm()
                node_sum = sum(all_nodes)
                node_len = len(all_nodes)
                num = len(alphas)
                processes_1 = node_sum / num
                remainder1 = node_sum % num

                if node_len >= num:
                    node = hosts[i]
                    processes = all_nodes[i]
                else:
                    if processes_1 < all_nodes[host_count]:
                        node = hosts[host_count]
                        processes = all_nodes[host_count]
                        host_count += 1
                        cores_count += processes
                    else:
                        node = hosts[host_count]
                        processes = all_nodes[host_count] - processes_1
                        cores_count += processes
                mpi_Command = 'mpirun -n %i --map-by node -host %s %s'
            else:
                mpi_Command = 'mpirun -n %i %s'
                if i >= len(alphas) - remainder:
                    processes = konfig['NUMBER_PART'] + 1
                else:
                    processes = konfig['NUMBER_PART']

            the_Command = 'SU2_CFD ' + tempname
            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError('could not find an mpi interface')
            if slurm_job:
                the_Command = mpi_Command % (processes, node, the_Command)
            else:
                the_Command = mpi_Command % (processes,the_Command)
            sys.stdout.flush()
            cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'.txt', 'w')
            print(the_Command)
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()
            procTotal.append(copy.deepcopy(proc))
            konfigDirectTotal.append(copy.deepcopy(konfig_direct))
            konfigTotal.append(copy.deepcopy(konfig))
            ztateTotal.append(copy.deepcopy(state))

        for i in list(range(len(alphas))):
            while procTotal[i].poll() is None:
                pass
            konfig = konfigDirectTotal[i]
            #konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
            #oldstdout = sys.stdout
            #sys.stdout = oldstdout
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            final_avg = config.get('ITER_AVERAGE_OBJ',0)
            aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg )
            config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
            info = SU2.io.State()
            info.FUNCTIONS.update( aerodynamics )
            ztateTotal[i].update(info)
            SU2.io.restart2solution(konfig, ztateTotal[i])
            cl[i], cd[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']

        if afOptions['GradientOptions']['ComputeGradient']:
            if afOptions['CFDOptions']['CFDGradType'] != 'FiniteDiff':
                dcd_dSafs, dcd_dalphas = self.__cfdGradientsParallel(basepath, konfigTotal, ztateTotal, 'DRAG')
                dcl_dSafs, dcl_dalphas = self.__cfdGradientsParallel(basepath, konfigTotal, ztateTotal, 'LIFT')
                dcl_dRes, dcd_dRes = np.zeros(len(dcl_dalphas)), np.zeros(len(dcl_dalphas))
            else:
                cl_new = np.zeros(len(alphas))
                cd_new = np.zeros(len(alphas))
                alphas_new = np.zeros(len(alphas))
                fd_step = self.afOptions['GradientOptions']['fd_step']
                for k in list(range(len(alphas))):
                    alphas_new[k] = alphas[k] + fd_step
                procTotal = []
                konfigTotal = []
                konfigDirectTotal = []
                ztateTotal = []
                Re = afOptions['SplineOptions']['Re']
                for i in list(range(len(alphas))):
                    meshFileName = basepath + os.path.sep + 'mesh_airfoil'+str(i+1)+'.su2'
                    # Create airfoil coordinate file for SU2
                    afanalysis = AirfoilAnalysis(Safs[i], afOptions)
                    x, y = afanalysis.x, afanalysis.y
                    airfoilFile = basepath + os.path.sep + 'airfoil'+str(i+1)+'_coordinates.dat'
                    coord_file = open(airfoilFile, 'w')
                    coord_file.write('Airfoil Parallel')
                    for j in list(range(len(x))):
                        coord_file.write('{:<10f}\t{:<10f}'.format(x[j], y[j]))
                    coord_file.close()

                    konfig = copy.deepcopy(config)
                    ztate = copy.deepcopy(state)
                    konfig.MESH_OUT_FILENAME = meshFileName
                    konfig.DV_KIND = 'AIRFOIL'
                    tempname = basepath + os.path.sep + 'config_DEF_direct.cfg'
                    konfig.dump(tempname)
                    SU2_RUN = os.environ['SU2_RUN']
                    base_Command = os.path.join(SU2_RUN,'%s')
                    the_Command = 'SU2_DEF ' + tempname
                    the_Command = base_Command % the_Command
                    sys.stdout.flush()
                    proc = subprocess.Popen( the_Command, shell=True    ,
                                     stdout=open(basepath + os.path.sep + 'mesh_deformation_airfoil'+str(i+1)+'.txt', 'w')      ,
                                     stderr=subprocess.PIPE,
                                     stdin=subprocess.PIPE)
                    proc.stderr.close()
                    proc.stdin.write(airfoilFile+'\n')
                    proc.stdin.write('Selig\n')
                    proc.stdin.write('1.0\n')
                    proc.stdin.write('Yes\n')
                    proc.stdin.write('clockwise\n')
                    proc.stdin.close()
                    return_code = proc.wait()

                    restart = False

                    konfig.MESH_FILENAME = meshFileName
                    ztate.FILES.MESH = config.MESH_FILENAME
                    Uinf = 10.0
                    Ma = Uinf / 340.29  # Speed of sound at sea level
                    konfig.MACH_NUMBER = 0.15 # Ma
                    konfig.REYNOLDS_NUMBER = Re

                    if restart:
                        konfig.RESTART_SOL = 'YES'
                        konfig.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'.dat'
                        konfig.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'_SOLVED.dat'
                    else:
                        konfig.RESTART_SOL = 'NO'
                        konfig.SOLUTION_FLOW_FILENAME = basepath + os.path.sep + 'solution_flow_airfoil'+str(i+1)+'.dat'
                        konfig.SOLUTION_ADJ_FILENAME = basepath + os.path.sep + 'solution_adj_airfoil'+str(i+1)+'.dat'
                        konfig.RESTART_FLOW_FILENAME = basepath + os.path.sep + 'restart_flow_airfoil'+str(i+1)+'.dat'
                        konfig.RESTART_ADJ_FILENAME = basepath + os.path.sep + 'restart_adj_airfoil'+str(i+1)+'.dat'
                        konfig.SURFACE_ADJ_FILENAME = basepath + os.path.sep + 'surface_adjoint_airfoil' + str(i+1)
                        konfig.SURFACE_FLOW_FILENAME = basepath + os.path.sep + 'surface_flow_airfoil' + str(i+1)


                    x_vel = Uinf * cos(np.radians(alphas[i]))
                    y_vel = Uinf * sin(np.radians(alphas[i]))
                    konfig.FREESTREAM_VELOCITY = '( ' + str(x_vel) + ', ' + str(y_vel) + ', 0.00 )'
                    konfig.AoA = alphas[i]
                    konfig.CONV_FILENAME = basepath + os.path.sep + 'history_airfoil'+str(i+1)
                    #state = SU2.io.State(state)

                    konfig_direct = copy.deepcopy(konfig)
                    # setup direct problem
                    konfig_direct['MATH_PROBLEM']  = 'DIRECT'
                    konfig_direct['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_direct'

                    # Run Solution
                    tempname = basepath + os.path.sep + 'config_CFD_airfoil'+str(i+1)+'.cfg'
                    konfig_direct.dump(tempname)
                    SU2_RUN = os.environ['SU2_RUN']
                    sys.path.append( SU2_RUN )

                    mpi_Command = 'mpirun -n %i %s'
                    if i >= len(alphas) - remainder:
                        processes = konfig['NUMBER_PART'] + 1
                    else:
                        processes = konfig['NUMBER_PART']

                    the_Command = 'SU2_CFD ' + tempname
                    the_Command = base_Command % the_Command
                    if processes > 0:
                        if not mpi_Command:
                            raise RuntimeError('could not find an mpi interface')
                    the_Command = mpi_Command % (processes,the_Command)
                    sys.stdout.flush()
                    cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'.txt', 'w')
                    print(the_Command, alphas[i])
                    proc = subprocess.Popen( the_Command, shell=True    ,
                                 stdout=cfd_output      ,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE)
                    proc.stderr.close()
                    proc.stdin.close()
                    procTotal.append(copy.deepcopy(proc))
                    konfigDirectTotal.append(copy.deepcopy(konfig_direct))
                    konfigTotal.append(copy.deepcopy(konfig))
                    ztateTotal.append(copy.deepcopy(state))

                for i in list(range(len(alphas))):
                    while procTotal[i].poll() is None:
                        pass
                    konfig = konfigDirectTotal[i]
                    konfig['SOLUTION_FLOW_FILENAME'] = konfig['RESTART_FLOW_FILENAME']
                    #oldstdout = sys.stdout
                    #sys.stdout = oldstdout
                    plot_format      = konfig['OUTPUT_FORMAT']
                    plot_extension   = SU2.io.get_extension(plot_format)
                    history_filename = konfig['CONV_FILENAME'] + plot_extension
                    special_cases    = SU2.io.get_specialCases(konfig)

                    final_avg = config.get('ITER_AVERAGE_OBJ',0)
                    aerodynamics = SU2.io.read_aerodynamics( history_filename , special_cases, final_avg )
                    config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM']  })
                    info = SU2.io.State()
                    info.FUNCTIONS.update( aerodynamics )
                    ztateTotal[i].update(info)

                    cl_new[i], cd_new[i] = info.FUNCTIONS['LIFT'], info.FUNCTIONS['DRAG']
                dcl_dalphas, dcd_dalphas = np.zeros(len(cl_new)), np.zeros(len(cd_new))
                dcl_dRes, dcd_dRes = np.zeros(len(cl_new)), np.zeros(len(cd_new))
                dcl_dSafs, dcd_dSafs = [], []
                for w in list(range(len(cl_new))):
                    dcl_dalphas[w] = (cl_new[w] - cl[w]) / fd_step
                    dcd_dalphas[w] = (cd_new[w] - cd[w]) / fd_step
                    dcl_dSafs.append(np.zeros(8))
                    dcd_dSafs.append(np.zeros(8)) ## Haven't implemented FD for Saf yet
            return cl, cd, dcl_dalphas, dcl_dRes, dcd_dalphas, dcd_dRes, dcl_dSafs, dcd_dSafs

        return cl, cd

    def __cfdGradientsSerial(self, konfig, ztate, objective):
        """ get gradients of one CFD simulation
        Parameters
        ----------
        konfig : object
            the temporary configurations for the CFD simulation
        ztateTotal : object
            the temporary state of the CFD simulation
        objective : str
            either 'LIFT' or 'DRAG' depending on desired gradients

        Returns
        -------
        df_dSaf : array
            the objective gradients with respect to airfoil shape parameters
        df_dalpha : array
            the objective gradients with respect to angle of attack

        """
        sys.path.append(os.environ['SU2_RUN'])
        import SU2
        konfig.RESTART_SOL = 'NO'
        mesh_data = SU2.mesh.tools.read(konfig.MESH_FILENAME)
        points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')

        SU2.io.restart2solution(konfig, ztate)
        konfig.OBJECTIVE_FUNCTION = objective

        # setup problem
        if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
            konfig['MATH_PROBLEM']  = 'DISCRETE_ADJOINT'
        else:
            konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
        konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_adjoint'

        # Run Solution
        tempname = basepath + os.path.sep + 'CFD' + os.path.sep + 'config_CFD_airfoil_'+objective+'.cfg'
        konfig.dump(tempname)

        SU2_RUN = os.environ['SU2_RUN']
        sys.path.append( SU2_RUN )
        processes = konfig['NUMBER_PART']
        if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
            the_Command = 'SU2_CFD_AD ' + tempname
        else:
            the_Command = 'SU2_CFD ' + tempname
        base_Command = os.path.join(SU2_RUN,'%s')
        the_Command = base_Command % the_Command
        if konfig['NUMBER_PART'] > 0:
            mpi_Command = 'mpirun -n %i %s'
            the_Command = the_Command = mpi_Command % (processes,the_Command)
        else:
            mpi_Command = ''
        if processes > 0:
            if not mpi_Command:
                raise RuntimeError('could not find an mpi interface')
        print(the_Command)
        sys.stdout.flush()
        cfd_output = open(basepath + os.path.sep + 'CFD' + os.path.sep + 'cfd_output_airfoil_'+ objective +'.txt', 'w')
        proc = subprocess.Popen( the_Command, shell=True    ,
                     stdout=cfd_output      ,
                     stderr=cfd_output,
                     stdin=subprocess.PIPE)
        # proc.stderr.close()
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise ValueError('Error in CFD Gradient ' + objective + '. Error code: %c. Check cfd_output_airfoil_'+ objective +'.txt for more details.' % (return_code))

        konfig['SOLUTION_ADJ_FILENAME'] = konfig['RESTART_ADJ_FILENAME']

        # filenames
        plot_format      = konfig['OUTPUT_FORMAT']
        plot_extension   = SU2.io.get_extension(plot_format)
        history_filename = konfig['CONV_FILENAME'] + plot_extension
        special_cases    = SU2.io.get_specialCases(konfig)

        # get history
        history = SU2.io.read_history( history_filename )

        # update super config
        # config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM'] ,
        #                 'OBJECTIVE_FUNCTION'  : konfig['OBJECTIVE_FUNCTION']   })

        # files out
        objective    = konfig['OBJECTIVE_FUNCTION']
        adj_title    = 'ADJOINT_' + objective
        suffix       = SU2.io.get_adjointSuffix(objective)
        restart_name = konfig['RESTART_FLOW_FILENAME']
        restart_name = SU2.io.add_suffix(restart_name,suffix)

        # info out
        info = SU2.io.State()
        info.FILES[adj_title] = restart_name
        info.HISTORY[adj_title] = history
        ztate.update(info)
        if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
            SU2.io.restart2solution(konfig,ztate)
            konfig.DEFINITION_DV['KIND'] = [0]*8
            konfig.DEFINITION_DV['MARKER'] = [0]*8
            konfig.DEFINITION_DV['PARAM'] = [0]*8
            konfig.DEFINITION_DV['FFDTAG'] = [0]*8
            konfig.DEFINITION_DV['SCALE'] = [0]*8
            konfig.DEFINITION_DV['SIZE'] = [0]*8
            ww = [0, 0, 0, 0, 1, 1, 1, 1]
            www = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
            for w in list(range(8)):
                konfig.DEFINITION_DV['KIND'][w] = 'CST'
                konfig.DEFINITION_DV['MARKER'][w] = ['airfoil']
                konfig.DEFINITION_DV['PARAM'][w] = [ww[w], www[w]]
                konfig.DEFINITION_DV['FFDTAG'][w] = []
                konfig.DEFINITION_DV['SCALE'][w] = 1.0
                konfig.DEFINITION_DV['SIZE'][w] = 1

            # Gradient Projection
            step = 1e-3
            info = SU2.run.projection(konfig,step)
            ztate.update(info)
            if objective == 'DRAG':
                df_dSaf = np.asarray(ztate.GRADIENTS.DRAG)
            else:
                df_dSaf = np.asarray(ztate.GRADIENTS.LIFT)
        else:
            surface_adjoint = konfig.SURFACE_ADJ_FILENAME + '.csv'
            ztate.update(info)
            df_dx, xl, xu, dy = self.su2Gradient(points_sorted, surface_adjoint)
            dy_dSaf = self.__cstYDerivatives(self.wl, self.wu, len(df_dx), self.dz, xl, xu)
            dSaf_dx_ = np.matrix(dy_dSaf)
            df_dx_ = np.matrix(df_dx)
            df_dSaf = np.asarray(dSaf_dx_ * df_dx_.T).reshape(8)
            SU2.io.restart2solution(konfig, ztate)
        try:
            if objective == 'DRAG':
                df_dalpha = ztate.HISTORY.ADJOINT_DRAG.Sens_AoA[-1]
            else:
                df_dalpha = ztate.HISTORY.ADJOINT_LIFT.Sens_AoA[-1]
        except:
            df_dalpha = 0.0
        return df_dSaf, df_dalpha

    def __cfdGradientsParallel(self, konfigTotal, ztateTotal, objective):
        """ obtain the gradients through parallel CFD simulations
        Parameters
        ----------
        konfigTotal : array
            stores all the temporary configurations for each CFD simulation
        ztateTotal : array
            stores the temporary state of each CFD simulation
        objective : str
            either 'LIFT' or 'DRAG' depending on desired gradients
        Returns
        -------
        df_dSafs : array
            the objective gradients with respect to airfoil shape parameters
        df_dalphas : array
            the objective gradients with respect to angles of attack

        """
        sys.path.append(os.environ['SU2_RUN'])
        import SU2
        procTotal = []
        konfigFTotal = []
        df_dalphas = []
        df_dSafs = []
        SU2_RUN = os.environ['SU2_RUN']
        base_Command = os.path.join(SU2_RUN,'%s')
        host_count = 0
        cores_count = 0
        remainder = self.afOptions['CFDOptions']['processors'] % len(konfigTotal)

        for i in list(range(len(konfigTotal))):
            konfig = copy.deepcopy(konfigTotal[i])
            ztate = ztateTotal[i]
            konfig.RESTART_SOL = 'NO'
            konfig.EXT_ITER    = self.afOptions['CFDOptions']['iterations'] + 4000
            konfig.RESIDUAL_MINVAL = -5.8
            mesh_data = SU2.mesh.tools.read(konfig.MESH_FILENAME)
            points_sorted, loop_sorted = SU2.mesh.tools.sort_airfoil(mesh_data, marker_name='airfoil')


            konfig.OBJECTIVE_FUNCTION = objective

            # setup problem
            if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
                konfig['MATH_PROBLEM']  = 'DISCRETE_ADJOINT'
            else:
                konfig['MATH_PROBLEM']  = 'CONTINUOUS_ADJOINT'
            konfig['CONV_FILENAME'] = konfig['CONV_FILENAME'] + '_adjoint'

            # Run Solution
            tempname = basepath + os.path.sep + 'config_CFD_airfoil'+str(i+1)+'_'+objective+'.cfg'
            konfig.dump(tempname)
            SU2_RUN = os.environ['SU2_RUN']
            sys.path.append( SU2_RUN )
            slurm_job = os.environ.has_key('SLURM_JOBID')
            if slurm_job:
                host_names = os.environ.get('SLURM_JOB_NODELIST')
                host_names_array = host_names.split()
                nodes_per_host = os.environ.get('SLURM_TASKS_PER_NODE')
                nodes_per_host_array = host_names.split(',')
                node_sum = sum(nodes_per_host_array)
                node_len = len(nodes_per_host_array)
                num = len(konfigTotal)
                processes_1 = node_sum / num
                remainder1 = node_sum % num

                if node_len >= num:
                    node = host_names_array[i]
                    processes = nodes_per_host_array[i]
                else:
                    if processes_1 < nodes_per_host_array[host_count]:
                        node = host_names_array[host_count]
                        processes = nodes_per_host_array[host_count]
                        host_count += 1
                        cores_count += processes
                    else:
                        node = host_names_array[host_count]
                        processes = nodes_per_host_array[host_count] - processes_1
                        cores_count += processes
                    mpi_Command = 'mpirun -n %i --map-by node -host %s %s'
            else:
                mpi_Command = 'mpirun -n %i %s'
            if i >= len(konfigTotal) - remainder:
                processes = konfig['NUMBER_PART'] + 1
            else:
                processes = konfig['NUMBER_PART']

            if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
                the_Command = 'SU2_CFD_AD ' + tempname
            else:
                the_Command = 'SU2_CFD ' + tempname

            the_Command = base_Command % the_Command
            if processes > 0:
                if not mpi_Command:
                    raise RuntimeError('could not find an mpi interface')
            if slurm_job:
                the_Command = mpi_Command % (processes, node, the_Command)
            else:
                the_Command = mpi_Command % (processes,the_Command)
            sys.stdout.flush()
            print(the_Command)
            cfd_output = open(basepath + os.path.sep + 'cfd_output_airfoil'+str(i+1)+'_'+objective+'.txt', 'w')
            proc = subprocess.Popen( the_Command, shell=True    ,
                         stdout=cfd_output      ,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
            proc.stderr.close()
            proc.stdin.close()

            # merge

            procTotal.append(copy.deepcopy(proc))
            konfigFTotal.append(copy.deepcopy(konfig))
            # ztateTotal.append(copy.deepcopy(state))

        for i in list(range(len(konfigTotal))):
            while procTotal[i].poll() is None:
                pass
            konfig = konfigFTotal[i]
            konfig['SOLUTION_ADJ_FILENAME'] = konfig['RESTART_ADJ_FILENAME']

            # filenames
            plot_format      = konfig['OUTPUT_FORMAT']
            plot_extension   = SU2.io.get_extension(plot_format)
            history_filename = konfig['CONV_FILENAME'] + plot_extension
            special_cases    = SU2.io.get_specialCases(konfig)

            # get history
            history = SU2.io.read_history( history_filename )

            # update super config
            # config.update({ 'MATH_PROBLEM' : konfig['MATH_PROBLEM'] ,
            #                 'OBJECTIVE_FUNCTION'  : konfig['OBJECTIVE_FUNCTION']   })

            # files out
            objective    = konfig['OBJECTIVE_FUNCTION']
            adj_title    = 'ADJOINT_' + objective
            suffix       = SU2.io.get_adjointSuffix(objective)
            restart_name = konfig['RESTART_FLOW_FILENAME']
            restart_name = SU2.io.add_suffix(restart_name,suffix)

            # info out
            info = SU2.io.State()
            info.FILES[adj_title] = restart_name
            info.HISTORY[adj_title] = history

            ztate.update(info)
            if self.afOptions['CFDOptions']['CFDGradType'] == 'AutoDiff':
                SU2.io.restart2solution(konfig,ztate)
                konfig.DEFINITION_DV['KIND'] = [0]*8
                konfig.DEFINITION_DV['MARKER'] = [0]*8
                konfig.DEFINITION_DV['PARAM'] = [0]*8
                konfig.DEFINITION_DV['FFDTAG'] = [0]*8
                konfig.DEFINITION_DV['SCALE'] = [0]*8
                konfig.DEFINITION_DV['SIZE'] = [0]*8
                ww = [0, 0, 0, 0, 1, 1, 1, 1]
                www = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
                for w in list(range(8)):
                    konfig.DEFINITION_DV['KIND'][w] = 'CST'
                    konfig.DEFINITION_DV['MARKER'][w] = ['airfoil']
                    konfig.DEFINITION_DV['PARAM'][w] = [ww[w], www[w]]
                    konfig.DEFINITION_DV['FFDTAG'][w] = []
                    konfig.DEFINITION_DV['SCALE'][w] = 1.0
                    konfig.DEFINITION_DV['SIZE'][w] = 1

                # Gradient Projection
                step = 1e-3
                oldstdout = sys.stdout
                sys.stdout = open(basepath + os.path.sep + 'projection_output_airfoil'+str(i+1)+'_'+objective+'.txt', 'w')
                info = SU2.run.projection(konfig,step)
                sys.stdout = oldstdout
                ztate.update(info)
                if objective == 'DRAG':
                    df_dSafs.append(np.asarray(ztate.GRADIENTS.DRAG))
                else:
                    df_dSafs.append(np.asarray(ztate.GRADIENTS.LIFT))
            else:
                surface_adjoint = konfig.SURFACE_ADJ_FILENAME + '.csv'
                ztate.update(info)
                df_dx, xl, xu, dy = self.su2Gradient(points_sorted, surface_adjoint)
                dy_dSaf = self.__cstYDerivatives(self.wl, self.wu, len(df_dx), self.dz, xl, xu)
                dSaf_dx_ = np.matrix(dy_dSaf)
                df_dx_ = np.matrix(df_dx)
                df_dSafs.append(np.asarray(dSaf_dx_ * df_dx_.T).reshape(8))
                SU2.io.restart2solution(konfig, ztate)
            try:
                if objective == 'DRAG':
                    df_dalphas.append(ztate.HISTORY.ADJOINT_DRAG.Sens_AoA[-1])
                else:
                    df_dalphas.append(ztate.HISTORY.ADJOINT_LIFT.Sens_AoA[-1])
            except:
                df_dalphas.append(0.0)

        return df_dSafs, df_dalphas

    def __su2GradientSort(self, loop_sorted, surface_adjoint):
        """ sorts the data
        Parameters
        ----------
        loop_sorted : array
            mesh array points in order from leading edge clockwise
        surface_adjoint : str
            filename of surface_adjoint.csv file

        Returns
        -------
        None

        """
        data = np.zeros([500, 10])
        with open(surface_adjoint, 'rb') as f1:
            reader = csv.reader(f1, dialect='excel', quotechar='|')
            i = 0
            for row in reader:
                if i > 0:
                    data[i, :] = row[0:10]
                i += 1
            f1.close()
        N = len(loop_sorted)
        dobj_dx_raw = data[:, 1][1:N+1].reshape(N,1)
        point_ids = data[:, 0][1:N+1].reshape(N,1)
        x = data[:, 6][1:N+1].reshape(N,1)
        y = data[:, 7][1:N+1].reshape(N,1)
        dx = data[:, 8][1:N+1].reshape(N,1)
        dy = data[:, 9][1:N+1].reshape(N,1)
        xu, xl, yu, yl, dobj_dxl, dobj_dxu = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),  np.zeros(0), np.zeros(0) #TODO: Initalize
        for i in list(range(N)):
            index = np.where(point_ids == loop_sorted[i])[0][0]
            if i < N/2:
                xl = np.append(xl, x[index])
                yl = np.append(yl, y[index])
                dobj_dxl = np.append(dobj_dxl, dobj_dx_raw[index])
            else:
                xu = np.append(xu, x[index])
                yu = np.append(yu, y[index])
                dobj_dxu = np.append(dobj_dxu, dobj_dx_raw[index])
        return np.concatenate([dobj_dxl, dobj_dxu]), xl, xu, dy

    def __slurm(self):
        """ obtains the hostnames and cores for a slurm environment, tested on Brigham Young's Marylou supmercomputer
        Parameters
        ----------
        None

        Returns
        -------
        hosts : array
            array of strings of the hosts available
        all_nodes : array
            array of ints of the number of processors or cores available on each host
        """
        hosts = os.environ.get('NODEFILE').split()
        all_nodes = []
        nodes = os.environ.get('SLURM_TASKS_PER_NODE')
        nodes_per_host = nodes.split(',')
        for i in list(range(len(nodes_per_host))):
            section1 = nodes_per_host[i].split('x')
            if len(section1) > 1:
                section2 = section1[0][:-1]
                section3 = section1[1][:-1]
                for zz in list(range(int(section3))):
                    all_nodes.append(int(section2))
            else:
                all_nodes.append(int(section1[0]))
        return hosts, all_nodes

    def __naca(self, number, n, finite_TE = False, half_cosine_spacing = False):
        """
        Python 2 and 3 code to generate 4 and 5 digit NACA profiles

        The NACA airfoils are airfoil shapes for aircraft wings developed
        by the National Advisory Committee for Aeronautics (NACA).
        The shape of the NACA airfoils is described using a series of
        digits following the word "NACA". The parameters in the numerical
        code can be entered into equations to precisely generate the
        cross-section of the airfoil and calculate its properties.
            https://en.wikipedia.org/wiki/NACA_airfoil

        Pots of the Matlab code available here:
            http://www.mathworks.com/matlabcentral/fileexchange/19915-naca-4-digit-airfoil-generator
            http://www.mathworks.com/matlabcentral/fileexchange/23241-naca-5-digit-airfoil-generator

        Copyright (C) 2011 by Dirk Gorissen <dgorissen@gmail.com>

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        """
        number = str(number)
        if len(number)==4:
            return self.__naca4(number, n, finite_TE, half_cosine_spacing)
        elif len(number)==5:
            return self.__naca5(number, n, finite_TE, half_cosine_spacing)
        else:
            raise Exception

    def __linspace(self, start,stop,np):
        """
        Emulate Matlab linspace
        """
        return [start+(stop-start)*i/(np-1) for i in list(range(np))]

    def __interpolate(self, xa,ya,queryPoints):
        """
        A cubic spline interpolation on a given set of points (x,y)
        Recalculates everything on every call which is far from efficient but does the job for now
        should eventually be replaced by an external helper class
        """

        # PreCompute() from Paint Mono which in turn adapted:
        # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
        # ISBN 0-521-43108-5, page 113, section 3.3.
        # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

        #number of points
        n = len(xa)
        u, y2 = [0]*n, [0]*n

        for i in list(range(1,n-1)):

            # This is the decomposition loop of the tridiagonal algorithm.
            # y2 and u are used for temporary storage of the decomposed factors.

            wx = xa[i + 1] - xa[i - 1]
            sig = (xa[i] - xa[i - 1]) / wx
            p = sig * y2[i - 1] + 2.0

            y2[i] = (sig - 1.0) / p

            ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])

            u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p


        y2[n - 1] = 0

        # This is the backsubstitution loop of the tridiagonal algorithm
        #((int i = n - 2; i >= 0; --i):
        for i in list(range(n-2,-1,-1)):
            y2[i] = y2[i] * y2[i + 1] + u[i]

        # interpolate() adapted from Paint Mono which in turn adapted:
        # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
        # ISBN 0-521-43108-5, page 113, section 3.3.
        # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

        results = [0]*n

        #loop over all query points
        for i in list(range(len(queryPoints))):
            # bisection. This is optimal if sequential calls to this
            # routine are at random values of x. If sequential calls
            # are in order, and closely spaced, one would do better
            # to store previous values of klo and khi and test if

            klo = 0
            khi = n - 1

            while (khi - klo > 1):
                k = (khi + klo) >> 1
                if (xa[k] > queryPoints[i]):
                    khi = k
                else:
                    klo = k

            h = xa[khi] - xa[klo]
            a = (xa[khi] - queryPoints[i]) / h
            b = (queryPoints[i] - xa[klo]) / h

            # Cubic spline polynomial is now evaluated.
            results[i] = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0

        return results

    def __naca4(self, number, n, finite_TE = False, half_cosine_spacing = False):
        """
        Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
        """

        m = float(number[0])/100.0
        p = float(number[1])/10.0
        t = float(number[2:])/100.0

        a0 = +0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = +0.2843

        if finite_TE:
            a4 = -0.1015 # For finite thick TE
        else:
            a4 = -0.1036 # For zero thick TE

        if half_cosine_spacing:
            beta = self.__linspace(0.0,pi,n+1)
            x = [(0.5*(1.0-cos(xx))) for xx in beta]  # Half cosine based spacing
        else:
            x = self.__linspace(0.0,1.0,n+1)

        yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

        xc1 = [xx for xx in x if xx <= p]
        xc2 = [xx for xx in x if xx > p]

        if p == 0:
            xu = x
            yu = yt

            xl = x
            yl = [-xx for xx in yt]

            xc = xc1 + xc2
            zc = [0]*len(xc)
        else:
            yc1 = [m/pow(p,2)*xx*(2*p-xx) for xx in xc1]
            yc2 = [m/pow(1-p,2)*(1-2*p+xx)*(1-xx) for xx in xc2]
            zc = yc1 + yc2

            dyc1_dx = [m/pow(p,2)*(2*p-2*xx) for xx in xc1]
            dyc2_dx = [m/pow(1-p,2)*(2*p-2*xx) for xx in xc2]
            dyc_dx = dyc1_dx + dyc2_dx

            theta = [atan(xx) for xx in dyc_dx]

            xu = [xx - yy * sin(zz) for xx,yy,zz in list(zip(x,yt,theta))]
            yu = [xx + yy * cos(zz) for xx,yy,zz in list(zip(zc,yt,theta))]

            xl = [xx + yy * sin(zz) for xx,yy,zz in list(zip(x,yt,theta))]
            yl = [xx - yy * cos(zz) for xx,yy,zz in list(zip(zc,yt,theta))]

        X = xu[::-1] + xl[1:]
        Z = yu[::-1] + yl[1:]

        return X,Z

    def __naca5(self, number, n, finite_TE = False, half_cosine_spacing = False):
        """
        Returns 2*n+1 points in [0 1] for the given 5 digit NACA number string
        """

        naca1 = int(number[0])
        naca23 = int(number[1:3])
        naca45 = int(number[3:])

        cld = naca1*(3.0/2.0)/10.0
        p = 0.5*naca23/100.0
        t = naca45/100.0

        a0 = +0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = +0.2843

        if finite_TE:
            a4 = -0.1015 # For finite thickness trailing edge
        else:
            a4 = -0.1036  # For zero thickness trailing edge

        if half_cosine_spacing:
            beta = self.__linspace(0.0,pi,n+1)
            x = [(0.5*(1.0-cos(x))) for x in beta]  # Half cosine based spacing
        else:
            x = self.__linspace(0.0,1.0,n+1)

        yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

        P = [0.05,0.1,0.15,0.2,0.25]
        M = [0.0580,0.1260,0.2025,0.2900,0.3910]
        K = [361.4,51.64,15.957,6.643,3.230]

        m = self.__interpolate(P,M,[p])[0]
        k1 = self.__interpolate(M,K,[m])[0]

        xc1 = [xx for xx in x if xx <= p]
        xc2 = [xx for xx in x if xx > p]
        xc = xc1 + xc2

        if p == 0:
            xu = x
            yu = yt

            xl = x
            yl = [-x for x in yt]

            zc = [0]*len(xc)
        else:
            yc1 = [k1/6.0*(pow(xx,3)-3*m*pow(xx,2)+ pow(m,2)*(3-m)*xx) for xx in xc1]
            yc2 = [k1/6.0*pow(m,3)*(1-xx) for xx in xc2]
            zc  = [cld/0.3 * xx for xx in yc1 + yc2]

            dyc1_dx = [cld/0.3*(1.0/6.0)*k1*(3*pow(xx,2)-6*m*xx+pow(m,2)*(3-m)) for xx in xc1]
            dyc2_dx = [cld/0.3*(1.0/6.0)*k1*pow(m,3)]*len(xc2)

            dyc_dx = dyc1_dx + dyc2_dx
            theta = [atan(xx) for xx in dyc_dx]

            xu = [xx - yy * sin(zz) for xx,yy,zz in list(zip(x,yt,theta))]
            yu = [xx + yy * cos(zz) for xx,yy,zz in list(zip(zc,yt,theta))]

            xl = [xx + yy * sin(zz) for xx,yy,zz in list(zip(x,yt,theta))]
            yl = [xx - yy * cos(zz) for xx,yy,zz in list(zip(zc,yt,theta))]


        X = xu[::-1] + xl[1:]
        Z = yu[::-1] + yl[1:]

        return X,Z

def evaluate_direct_parallel2(alphas, Res, afs, computeAlphaGradient=False, computeSafGradient=False):
        indices_to_compute = []
        n = len(alphas)
        airfoilOptions = afs[-1].airfoilOptions
        cl = np.zeros(n)
        cd = np.zeros(n)
        dcl_dalpha = [0]*n
        dcd_dalpha = [0]*n
        dcl_dSaf = [0]*n
        dcd_dSaf = [0]*n
        dcl_dRe = [0]*n
        dcd_dRe = [0]*n

        for i in list(range(len(alphas))):
            alpha = alphas[i]
            Re = Res[i]
            af = afs[i]
            if af.Saf is not None and abs(np.degrees(alpha)) < af.airfoilOptions['SplineOptions']['maxDirectAoA']:
                if alpha in af.alpha_storage and alpha in af.dalpha_storage:
                    index = af.alpha_storage.index(alpha)
                    cl[i] = af.cl_storage[index]
                    cd[i] = af.cd_storage[index]
                    if computeAlphaGradient:
                        index = af.dalpha_storage.index(alpha)
                        dcl_dalpha[i] = af.dcl_storage[index]
                        dcd_dalpha[i] = af.dcd_storage[index]
                    if computeSafGradient and alpha in af.dalpha_dSaf_storage:
                        index = af.dalpha_dSaf_storage.index(alpha)
                        dcl_dSaf[i] = af.dcl_dSaf_storage[index]
                        dcd_dSaf[i] = af.dcd_dSaf_storage[index]
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    indices_to_compute.append(i)
            else:
                cl[i] = af.cl_spline.ev(alpha, Re)
                cd[i] = af.cd_spline.ev(alpha, Re)
                tck_cl = af.cl_spline.tck[:3] + af.cl_spline.degrees  # concatenate lists
                tck_cd = af.cd_spline.tck[:3] + af.cd_spline.degrees

                dcl_dalpha[i] = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
                dcd_dalpha[i] = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

                if af.one_Re:
                    dcl_dRe[i] = 0.0
                    dcd_dRe[i] = 0.0
                else:
                    dcl_dRe[i] = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                    dcd_dRe[i] = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
                if computeSafGradient and af.Saf is not None:
                    dcl_dSaf[i], dcd_dSaf[i] = af.afShapeGradients(alpha, Re)
                else:
                    dcl_dSaf[i], dcd_dSaf[i] = np.zeros(8), np.zeros(8)
        if indices_to_compute is not None:
            alphas_to_compute = [alphas[i] for i in indices_to_compute]
            Res_to_compute = [Res[i] for i in indices_to_compute]
            Safs_to_compute = [afs[i].Saf for i in indices_to_compute]
            if airfoilOptions['ComputeGradient']:
                cls, cds, dcls_dalpha, dcls_dRe, dcds_dalpha, dcds_dRe, dcls_dSaf, dcds_dSaf = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, Safs_to_compute, airfoilOptions)
                for j in list(range(len(indices_to_compute))):
                    dcl_dalpha[indices_to_compute[j]] = dcls_dalpha[j]
                    dcl_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcd_dalpha[indices_to_compute[j]] = dcds_dalpha[j]
                    dcd_dRe[indices_to_compute[j]] = dcls_dRe[j]
                    dcl_dSaf[indices_to_compute[j]] = dcls_dSaf[j]
                    dcd_dSaf[indices_to_compute[j]] = dcds_dSaf[j]

            else:
                cls, cds = cfdAirfoilsSolveParallel(alphas_to_compute, Res_to_compute, Safs_to_compute, airfoilOptions)
            for j in list(range(len(indices_to_compute))):
                cl[indices_to_compute[j]] = cls[j]
                cd[indices_to_compute[j]] = cds[j]

        if computeSafGradient:
            try:
                return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe, dcl_dSaf, dcd_dSaf
            except:
                raise
        elif computeAlphaGradient:
            return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe
        else:
            return cl, cd

# AnalysisMethod = None, 'XFOIL', 'CFD', 'Files', AirfoilParameterization = None, 'CST', 'Precomputational', 'NACA'
# CFDOptions: iterations = max iterations of CFD, processors = number of processors available to use, configFile = SU2 config file in AirfoilAnalysisFiles, computeAirfoilsInParallel = whether or not to compute multiple airfoils in parallel
# GradientOptions: ComputeGradient = whether or not to compute gradients in CCBlade, ComputeAirfoilGradients = whether or not to calculate airfoil parameterization gradients, fd_step = finite difference step size, cs_step = complex step size
# SplineOptions: AnalysisMethod: 'XFOIL', 'CFD', maxDirectAoA: deg at which spline takes over, alphas: alphas to use to compute spline, Re: Reynolds number to compute spline
# PrecomputationalOptions: AirfoilParameterization = 'TC' (thickness-to-chord ratio), 'Blended' (thickness-to-chord ratio with blended airfoil families)
# CFDGradType='AutoDiff', 'FiniteDiff', 'ContAdjoint'

if __name__ == '__main__':

    import os
    from argparse import ArgumentParser, RawTextHelpFormatter

    # setup command line arguments
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                            description='Preprocessing airfoil data for wind turbine applications.')
    parser.add_argument('src_file', type=str, help='source file')
    parser.add_argument('--stall3D', type=str, nargs=3, metavar=('r/R', 'c/r', 'tsr'), help='2D data -> apply 3D corrections')
    parser.add_argument('--extrap', type=str, nargs=1, metavar=('cdmax'), help='3D data -> high alpha extrapolations')
    parser.add_argument('--blend', type=str, nargs=2, metavar=('otherfile', 'weight'), help='blend 2 files weight 0: sourcefile, weight 1: otherfile')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--plot', action='store_true', help='plot data using matplotlib')
    parser.add_argument('--common', action='store_true', help='interpolate the data at different Reynolds numbers to a common set of angles of attack')


    # parse command line arguments
    args = parser.parse_args()
    fileOut = args.out

    if args.plot:
        import matplotlib.pyplot as plt

    # perform actions
    if args.stall3D is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_3D' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)
        floats = [float(var) for var in args.stall3D]
        af3D = af.correction3D(*floats)

        if args.common:
            af3D = af3D.interpToCommonAlpha()

        af3D.writeToAerodynFile(fileOut)

        if args.plot:

            for p, p3D in list(zip(af.polars, af3D.polars)):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                plt.plot(p.alpha, p.cl, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cl, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend(loc='lower right')

                # plt.subplot(122)
                plt.figure()
                plt.plot(p.alpha, p.cd, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cd, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend(loc='upper center')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/stall3d.pdf')

            plt.show()


    elif args.extrap is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_extrap' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)

        afext = af.extrapolate(float(args.extrap[0]))

        if args.common:
            afext = afext.interpToCommonAlpha()

        afext.writeToAerodynFile(fileOut)

        if args.plot:

            for p, pext in list(zip(af.polars, afext.polars)):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cl, 'r')
                p2, = plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.subplot(122)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cd, 'r')
                p2, = plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='lower right')

                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cm, 'r')
                p2, = plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/extrap.pdf')

            plt.show()


    elif args.blend is not None:

        if fileOut is None:
            name1, ext = os.path.splitext(args.src_file)
            name2, ext = os.path.splitext(os.path.basename(args.blend[0]))
            fileOut = name1 + '+' + name2 + '_blend' + args.blend[1] + ext

        af1 = Airfoil.initFromAerodynFile(args.src_file)
        af2 = Airfoil.initFromAerodynFile(args.blend[0])
        afOut = af1.blend(af2, float(args.blend[1]))

        if args.common:
            afOut = afOut.interpToCommonAlpha()

        afOut.writeToAerodynFile(fileOut)



        if args.plot:

            for p in afOut.polars:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.text(0.6, 0.2, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

            plt.show()