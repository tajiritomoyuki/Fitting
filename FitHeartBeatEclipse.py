#-*-coding:utf-8-*-
from astropy.io import fits
import os
import sys
import glob
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import emcee
import corner
import batman
from mpfit.mpfit import mpfit

per = 2.666965

"""
ecc 離心率
inc 傾斜(ラジアン)
omega ペリアストロン(ラジアン)
t0 ハートビートの開始位置
S ハートビートの振幅
C ハートビートのy軸位置
p 補正関数の係数
q 補正関数の初期位相
tp primary eclipseの中心位置
rp 中心星に対しての2体目の星の半径
a semi-major axis
q1 lim darkning係数1
q2 lim darkning係数2 https://academic.oup.com/mnras/article/435/3/2152/1024138
fp 2体目の星の光度
ts secondary eclipseの中心位置
"""


class HeartBeatModel():
    def __init__(self, ecc, inc, omega, t0, S, C, p, q, rp, a, q1, q2, fp):
        self.ecc = ecc
        self.inc = inc
        self.omega = omega
        self.t0 = t0
        self.S = S
        self.C = C
        self.p = p
        self.q = q

    def t2ma(self, t):
        ma = (t - self.t0) / per * 2. * np.pi
        return ma

    def ma2ea(self, ma):
        ks = pyasl.MarkleyKESolver()
        if isinstance(ma, float):
            ea = ks.getE(ma, self.ecc)
        else:
            ea_list = [ks.getE(_ma, self.ecc) for _ma in ma]
            ea = np.array(ea_list)
        return ea

    def ea2ta(self, ea):
        #ta = 2. * np.arctan2(np.sqrt(1 - self.ecc) * np.cos(ea / 2.), np.sqrt(1 + self.ecc) * np.sin(ea / 2.))
        ta = 2. * np.arctan(np.sqrt((1. + self.ecc) / (1. - self.ecc)) * np.tan(ea / 2.))
        return ta

    def ta2ea(self, ta):
        ea = 2. * np.arctan(np.sqrt((1. - self.ecc) / (1. + self.ecc)) * np.tan(ta / 2.))
        return ea

    def ea2ma(self, ea):
        ma = ea - self.ecc * np.sin(ea)
        return ma

    def ma2t(self, ma):
        t = ma * per / 2. / np.pi + self.t0
        t = np.where(t >= per, t - per, t)
        return t

    def t2ta(self, t):
        ma = self.t2ma(t)
        ea = self.ma2ea(ma)
        ta = self.ea2ta(ea)
        return ta

    def ta2t(self, ta):
        ea = self.ta2ea(ta)
        ma = self.ea2ma(ea)
        t = self.ma2t(ma)
        return t

    def getEclipseTime(self):
        tp = self.ta2t(np.pi / 2. - self.omega)
        ts = self.ta2t(3. * np.pi / 2. - self.omega)
        return tp, ts

    def fixFunc(self, ta):
        cor_val = self.p * np.sin(ta + self.q)
        return cor_val

    def modelFunc(self, t):
        #時間からmean anomalyを計算
        ma = self.t2ma(t)
        #mean anomalyからeccentric anomalyを計算
        ea = self.ma2ea(ma)
        #eccentric anomalyからtrue anomalyを計算
        ta = self.ea2ta(ea)
        #減光度を求める
        numerator = 1. - 3. * np.power(np.sin(self.inc), 2.) * np.power(np.sin(ta + self.omega), 2.)
        denominator = np.power((1. - np.power(self.ecc, 2.)) / (1. + self.ecc * np.cos(ta)), 3.)
        flux = self.S * numerator / denominator + self.C #+ self.fixFunc(ta)
        return flux

    def test(self, t):
        #時間からmean anomalyを計算
        ma = self.t2ma(t)
        #mean anomalyからeccentric anomalyを計算
        ea = self.ma2ea(ma)
        #eccentric anomalyからtrue anomalyを計算
        ta = self.ea2ta(ea)
        #減光度を求める
        numerator = 1. - 3. * np.power(np.sin(self.inc), 2.) * np.power(np.sin(ta - self.omega), 2.)
        denominator = np.power((1. - np.power(self.ecc, 2.)) / (1. + self.ecc * np.cos(ta)), 3.)
        flux = self.S * numerator / denominator + self.C
        return flux, ta


class EclipseModel():
    def __init__(self, ecc, inc, omega, t0, S, C, p, q, rp, a, q1, q2, fp):
        self.ecc = ecc
        self.inc = inc
        self.omega = omega
        self.rp = rp
        self.a = a
        self.q1 = q1
        self.q2 = q2
        self.fp = fp
        self.params = None

    def get_params(self, tp, ts):
        self.params = batman.TransitParams()
        #time of inferior conjunction
        self.params.t0 = tp
        #orbital period
        self.params.per = per
        #planet radius (in units of stellar radii)
        self.params.rp = self.rp
        #semi-major axis (in units of stellar radii)
        self.params.a = self.a
        #orbital inclination (in degrees)
        self.params.inc = np.rad2deg(self.inc)
        #eccentricity
        self.params.ecc = self.ecc
        #longitude of periastron (in degrees)
        self.params.w = np.rad2deg(self.omega)
        #limb darkening coefficients [u1, u2]
        u1 = 2. * np.sqrt(self.q1) * self.q2
        u2 = np.sqrt(self.q1) * (1. - 2. * self.q2)
        self.params.u = [u1, u2]
        #limb darkening model
        self.params.limb_dark = "quadratic"
        # #planet-to-star flux ratio
        self.params.fp = self.fp
        # #the central secondary eclipse time
        self.params.t_secondary = ts

    def primaryModel(self, t):
        model = batman.TransitModel(self.params, t)
        flux = model.light_curve(self.params)
        return flux

    def secondaryModel(self, t):
        model = batman.TransitModel(self.params, t, transittype="secondary")
        flux = model.light_curve(self.params)
        return flux

    def getTrueAnomaly(self, t):
        model = batman.TransitModel(self.params, t)
        ta = model.get_true_anomaly()
        return ta


class MCMC():
    def __init__(self):
        self.param_in = [0.2640, 1.392, -0.1398, 2.54017, 0.001593, 0.9994, 0.000455, 3.52, 0.13, 5.61, 0.78, 0.76, 0.001]
        self.ndim = len(self.param_in)
        self.nwalkers = 100
        self.tot_chain = None
        self.samples = None

    def __lnprior__(self, ecc, inc, omega, t0, S, C, p, q, rp, a, q1, q2, fp):
        if not 0. < ecc < 0.5:
            return -np.inf
        elif not 0. <= inc < np.pi / 2.:
            return -np.inf
        elif not -np.pi / 2. <= omega < np.pi / 2.:
            return -np.inf
        elif not 2.4 <= t0 < per:
            return -np.inf
        elif not 0. <= p < 0.001:
            return -np.inf
        elif not 0. < q <= 2 * np.pi:
            return -np.inf
        elif not 0. <= rp:
            return -np.inf
        elif not 0. <= a:
            return -np.inf
        elif not 0. <= q1 <= 1.:
            return -np.inf
        elif not 0. <= q2 <= 1.:
            return -np.inf
        elif not 0. <= fp:
            return -np.inf
        else:
            return 0.0

    def __lnlike__(self, param, x, y, yerr):
        hbm = HeartBeatModel(*param)
        tp, ts = hbm.getEclipseTime()
        em = EclipseModel(*param)
        em.get_params(tp, ts)
        model_y = hbm.modelFunc(x) * em.primaryModel(x) * em.secondaryModel(x)
        inv_sigma2 = 1.0 / (yerr ** 2)
        lnlike = -0.5 * (np.sum((y - model_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
        if np.isnan(lnlike):
            return -np.inf
        else:
            return lnlike

    def lnprob(self, param, x, y, yerr):
        lp = self.__lnprior__(*param)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__lnlike__(param, x, y, yerr)

    def execute(self, stack_time, stack_flux, stack_err):
        pos = [self.param_in + 1e-4 * np.random.randn(self.ndim) for i in range(self.nwalkers)]
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=(stack_time, stack_flux, stack_err))
        sampler.run_mcmc(pos, 10000)
        self.samples = sampler.chain[:, 5000:9990, :].reshape((-1, self.ndim))
        self.tot_chain = sampler.chain.reshape((-1, self.ndim))

    def result(self):
        label = ["$ecc$", "$inc$", "$omega$", "$t0$", "$S$", "$C$", "$p$", "$q$", "$rp$", "$a$", "$q1$", "$q2$", "$fp$"]
        fig = corner.corner(self.samples, labels=label)
        fig.savefig("corner2.png", bbox_inches="tight", pad_inches=0.0)
        np.savez("C:\\Users\\tajiri tomoyuki\\school\\dat\\fitting\\fitting_chain2.npz", self.tot_chain)
        print(self.param_in)
        print(self.samples[-1])


class LevenbergMarquardt():
    def __init__(self):
        self.m = None

    def getParamInfo(self):
        parinfo = [{"name":"ecc", "value":0., "fixed":0, "limited":[0, 0], "limits":[0., 0.]},
                   {"name":"inc", "value":0, "fixed":0, "limited":[1, 1], "limits":[-np.pi, np.pi]},
                   {"name":"omega", "value":0., "fixed":0, "limited":[1, 1], "limits":[-np.pi / 2., np.pi / 2.]},
                   {"name":"t0", "value":0., "fixed":0, "limited":[0, 0], "limits":[0., 0.]},
                   {"name":"S", "value":0., "fixed":0, "limited":[0, 0], "limits":[0., 0.]},
                   {"name":"C", "value":0., "fixed":0, "limited":[0, 0], "limits":[0., 0.]},
                   {"name":"p", "value":0., "fixed":0, "limited":[1, 0], "limits":[0., 0.]},
                   {"name":"q", "value":0., "fixed":0, "limited":[1, 1], "limits":[0., np.pi * 2.]},
                   {"name":"rp", "value":0., "fixed":0, "limited":[1, 0], "limits":[0., 0.]},
                   {"name":"a", "value":0., "fixed":0, "limited":[1, 0], "limits":[0., 0.]},
                   {"name":"q1", "value":0., "fixed":0, "limited":[1, 1], "limits":[0., 1.]},
                   {"name":"q2", "value":0., "fixed":0, "limited":[1, 1], "limits":[0., 1.]},
                   {"name":"fp", "value":0., "fixed":0, "limited":[1, 0], "limits":[0., 0.]}]
        return parinfo

    def lightcurveFunc(self, p, fjac=None, x=None, y=None, err=None):
        hbm = HeartBeatModel(*p)
        tp, ts = hbm.getEclipseTime()
        em = EclipseModel(*p)
        em.get_params(tp, ts)
        model_y = hbm.modelFunc(x) * em.primaryModel(x) * em.secondaryModel(x)
        status = 0
        return [status, (y - model_y) / err]

    def execute(self, stack_time, stack_flux, stack_er):
        p0 = [0.2640, 1.392, -0.1398, 2.54017, 0.001593, 0.9994, 0.000455, 3.52, 0.13, 5.61, 0.78, 0.76, 0.001]
        parinfo = self.getParamInfo()
        fa = {"x":stack_time, "y":stack_flux, "err":stack_er}
        self.m = mpfit(self.lightcurveFunc, p0, functkw=fa, parinfo=parinfo)

    def result(self):
        #np.savez("C:\\Users\\tajiri tomoyuki\\school\\dat\\fitting\\LM.npz", self.m)
        print(self.m.params)
        print(self.m.perror)


class ImportData():
    def __init__(self, sys_name, period):
        self.sys_name = sys_name
        self.period = period
        self.time_org = np.array([])
        self.flux_org = np.array([])
        self.cdno_org = np.array([])
        self.folded_time = None
        self.stack_time = np.array([])
        self.stack_flux = np.array([])
        self.stack_err = np.array([])
        self.datdir = "C:\\Users\\tajiri tomoyuki\\school\\kepler"

    def loadData(self, normalize=True):
        fitslist = glob.glob(os.path.join(self.datdir, self.sys_name, "*llc.fits"))
        for fitspath in fitslist:
            with fits.open(fitspath) as hdulist:
                t = hdulist["LIGHTCURVE"].data.field("TIME")
                f = hdulist["LIGHTCURVE"].data.field("PDCSAP_FLUX")
                c = hdulist["LIGHTCURVE"].data.field("CADENCENO").astype(np.uint32)
                if normalize == True:
                    mid_val = np.nanmedian(f)
                    f_mid = f / mid_val
                    self.flux_org = np.hstack((self.flux_org, f_mid))
                else:
                    self.flux_org = np.hstack((self.flux_org, f_mid))
                self.time_org = np.hstack((self.time_org, t))
                self.cdno_org = np.hstack((self.cdno_org, c))

    def foldTime(self):
        self.folded_time = np.mod(self.time_org, self.period)

    def stack(self, dev=1000):
        bin_size = self.period / dev
        for i in range(dev):
            t_min = i * bin_size
            t_max = (i + 1) * bin_size
            #一旦nanを-999に変更
            self.folded_time[np.isnan(self.folded_time)] = -999
            bin_flux = self.flux_org[np.where((t_min <= self.folded_time) & (self.folded_time < t_max))]
            self.folded_time[self.folded_time == -999] = -np.nan
            mid_time = (bin_size * (i + 0.5))
            mid_flux = np.nanmedian(bin_flux)
            flux_std = np.nanstd(bin_flux)
            self.stack_time = np.hstack((self.stack_time, mid_time))
            self.stack_flux = np.hstack((self.stack_flux, mid_flux))
            self.stack_err = np.hstack((self.stack_err, flux_std))

    def exportStackData(self, dev):
        self.loadData()
        self.foldTime()
        self.stack(dev=dev)
        return self.stack_time, self.stack_flux, self.stack_err


def maskEclipse(stack_time, stack_flux, mask="all"):
    masked_flux = np.array([])
    t1_init = 0.3
    t1_fin = 0.44
    f1_init = 0.
    f1_len = 0
    t2_init = 2.07
    t2_fin = 2.21
    f2_init = 0.
    f2_len = 0
    tmp_array = np.array([])
    for t, f in zip(stack_time, stack_flux):
        if t < t1_init:
            masked_flux = np.hstack((masked_flux, f))
        elif t1_init <= t < t1_fin:
            if mask == "all":
                if f1_init == 0.:
                    f1_init = f
                f1_len += 1
            else:
                masked_flux = np.hstack((masked_flux, f))
        elif t1_fin <= t < t2_init:
            if mask == "all":
                if f1_init != 0.:
                    tol = (f - f1_init) / (f1_len + 1)
                    cor_array = np.arange(f1_init + tol, f + tol, tol)[0 : f1_len + 1]
                    cor_array = cor_array + 7e-5 * np.random.randn(f1_len + 1)
                    masked_flux = np.hstack((masked_flux, cor_array))
                    f1_init = 0.
                else:
                    masked_flux = np.hstack((masked_flux, f))
            else:
                masked_flux = np.hstack((masked_flux, f))
        elif t2_init <= t < t2_fin:
            if f2_init == 0.:
                f2_init = f
            f2_len += 1
        elif t2_fin <= t:
            if f2_init != 0.:
                tol = (f - f2_init) / (f2_len + 1)
                cor_array = np.arange(f2_init + tol, f + tol, tol)[0 : f2_len + 1]
                cor_array = cor_array + 7e-5 * np.random.randn(f2_len + 1)
                masked_flux = np.hstack((masked_flux, cor_array))
                f2_init = 0.
            else:
                masked_flux = np.hstack((masked_flux, f))
    return masked_flux

def main():
    data = ImportData("003766353", 2.666965)
    stack_time, stack_flux, stack_err = data.exportStackData(1000)
    #masked_flux = maskEclipse(stack_time, stack_flux)
    mcmc = MCMC()
    mcmc.execute(stack_time, stack_flux, stack_err)
    mcmc.result()

def main2():
    data = ImportData("003766353", 2.666965)
    stack_time, stack_flux, stack_err = data.exportStackData(1000)
    lm = LevenbergMarquardt()
    lm.execute(stack_time, stack_flux, stack_err)
    lm.result()

def test():
    param = {
        "ecc": 0.2647,
        "inc": 1.415,
        "omega": -0.1381,
        "t0": 2.540566,
        "S": 0.001579,
        "C": 0.99958,
        "p": 0.000448996,
        "q": 3.48413,
        "rp": 0.1222,
        "a": 6.01312,
        "q1": 0.824589,
        "q2": 1.,
        "fp": 0.00085891
    }
    param = {
        "ecc": 0.2665,
        "inc": 1.5708,
        "omega": -0.133,
        "t0": 2.540964,
        "S": 0.00153811,
        "C": 1.00013668,
        "p": 0.000455,
        "q": 3.52,
        "rp": 0.06053899,
        "a": 10.0782672,
        "q1": 0.953831,
        "q2": 1.,
        "fp": 0.000376788
    }
    data = ImportData("003766353", 2.666965)
    stack_time, stack_flux, stack_err = data.exportStackData(1000)
    #masked_flux = maskEclipse(stack_time, stack_flux)
    hbm = HeartBeatModel(**param)
    tp, ts = hbm.getEclipseTime()
    em = EclipseModel(**param)
    em.get_params(tp, ts)
    model_flux = hbm.modelFunc(stack_time) * em.primaryModel(stack_time) * em.secondaryModel(stack_time)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(stack_time, stack_flux, c="red", s=10, marker="o")
    ax.scatter(stack_time, model_flux, c="blue", s=10, marker="o")
    ax1 = fig.add_subplot(212)
    plt.ylim([-0.001, 0.001])
    ax1.scatter(stack_time, stack_flux - model_flux, c="red", s=10, marker="o")
    plt.show()

def test2():
    param = {
        "ecc": 0.23,
        "inc": 0.84,
        "omega": 0.1,
        "t0": 2.565,
        "S": 0.0028,
        "C": 1.0002,
        "p": 0.00052,
        "q": 3.79,
        "rp": 0.1,
        "a": 15.,
        "q1": 0.11,
        "q2": 0.31,
        "fp": 0.001
    }
    data = ImportData("003766353", 2.666965)
    stack_time, stack_flux, stack_err = data.exportStackData(1000)
    #masked_flux = maskEclipse(stack_time, stack_flux)
    hbm = HeartBeatModel(**param)
    tp, ts = hbm.getEclipseTime()
    #print(tp, ts)
    em = EclipseModel(**param)
    em.get_params(tp, ts)
    ta1 = hbm.t2ta(stack_time)
    m#odel_flux = hbm.modelFunc(stack_time)
    ta2 = em.getTrueAnomaly(stack_time)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ta1, stack_flux, c="blue", s=10, marker="o")
    ax.scatter(ta2, stack_flux, c="red", s=10, marker="o")
    plt.show()

if __name__ == "__main__":
    #main2()
    test()
