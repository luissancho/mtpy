import itertools
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import t, norm
from scipy.stats.mstats import gmean
from tqdm import tqdm

from typing import Any, Literal, Optional
from typing_extensions import Self

from .dates import (
    Freq,
    is_timeseries,
    ts_from_delta,
    ts_to_delta
)
from .helpers import (
    array_adjust,
    array_shape,
    format_number,
    is_array,
    is_number,
    is_pandas
)


class Stat(object):

    def __init__(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        dist: Literal['norm', 't'] = 'norm',
        dropna: bool = False,
        outliers: Literal['keep', 'group', 'remove'] = 'keep',
        n_dev: int | float = 3,
        verbose: int = 0
    ) -> None:
        self.verbose = verbose  # Print progress

        self.data = data
        self.weights = weights
        self.dist = dist

        self.nobs = self.data.shape[0]
        self.dof = self.nobs - 1
        self.data = array_adjust(self.data, 0)
        self.vind = np.ones(self.nobs).astype(bool)

        if self.weights is not None:
            self.weights = array_adjust(self.weights, 0)
            self.weights *= self.nobs / self.weights.sum()
        else:
            self.weights = np.ones(self.nobs)

        if dropna:
            self.dropna()

        if outliers in ['group', 'remove']:
            self.winsorize(n_dev, action=outliers)

    def mean(self) -> float:
        return np.sum(self.weights * self.data) / self.nobs

    def median(self) -> float:
        return self.quantile(0.5)

    def var(self) -> float:
        return np.sum(self.weights * np.square(self.data - self.mean())) / self.dof

    def std(self) -> float:
        return np.sqrt(self.var())

    def quantile(self, q: float) -> float:
        sind = np.argsort(self.data)
        values = self.data[sind]
        cweights = np.cumsum(self.weights[sind])

        tgt = q * self.nobs
        i = np.searchsorted(cweights, tgt)

        if np.isclose(tgt, cweights[i]) and i < self.nobs - 1:
            return (values[i] + values[i + 1]) / 2

        return values[i]

    def dev(self, n_dev: int = 3) -> Self:
        mean = self.mean()
        std = self.std()

        return mean - n_dev * std, mean + n_dev * std

    def conf(self, alpha: int | float = 0.05) -> float:
        if alpha >= 1:  # Corresponding to a multiple of sigma (std)
            q = erf(alpha / np.sqrt(2))
        else:
            q = 1 - alpha / 2

        return self.ppf(q) * self.std()

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        mean = self.mean()
        conf = self.conf(alpha)

        return mean - conf, mean + conf

    def std_mean(self) -> float:
        return self.std() * np.sqrt(self.dof / self.nobs) / np.sqrt(self.dof)

    def conf_mean(self, alpha: float = 0.05) -> float:
        if alpha >= 1:  # Corresponding to a multiple of sigma (std)
            q = erf(alpha / np.sqrt(2))
        else:
            q = 1 - alpha / 2

        return self.ppf(q) * self.std_mean()

    def ci_mean(self, alpha: float = 0.05) -> tuple[float, float]:
        mean = self.mean()
        conf_mean = self.conf_mean(alpha)

        return mean - conf_mean, mean + conf_mean

    def ppf(self, q: float) -> float:
        if self.dist == 't':
            return t.ppf(q, df=self.dof)
        else:
            return norm.ppf(q)

    def dropna(self):
        self.vind = ~np.isnan(self.data)
        self.data = self.data[self.vind]
        self.weights = self.weights[self.vind]

        self.nobs = self.data.shape[0]
        self.dof = self.nobs - 1

        return self

    def winsorize(self, n_dev: int | float = 3, action: Literal['remove', 'group'] = 'remove') -> Self:
        if n_dev >= 1:  # Corresponding to a multiple of sigma (std)
            vmin, vmax = self.dev(n_dev)
        else:
            vmin, vmax = self.ci(1 - n_dev)

        x_in = (self.data >= vmin) & (self.data <= vmax)

        if action == 'remove':
            self.vind[self.vind] = x_in
            self.data = self.data[x_in]
            self.weights = self.weights[x_in]

            self.nobs = self.data.shape[0]
            self.dof = self.nobs - 1
        elif action == 'group':
            self.data = np.where(x_in, self.data, self.data[x_in].max())

        return self

    def scale_standard(self):
        self.data = (self.data - self.mean()) / self.std()

        return self

    def scale_minmax(self, vmin=0, vmax=1):
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min()) * (vmax - vmin) + vmin

        return self

    def scale_weights(self):
        self.data = self.data * self.nobs / self.data.sum()

        return self


class Kernel(object):

    def __init__(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        kernel: Literal['gaussian'] = 'gaussian',
        bw_type: Literal['fixed', 'adaptive'] = 'adaptive',
        bw: list | int | float | Literal['scott', 'silverman', 'isj'] = 'isj',
        n_iter: int = 1,
        min_delta: Optional[float] = None,
        verbose: int = 0
    ) -> None:
        self.verbose = verbose  # Print progress

        self.data = data
        self.weights = weights
        self.kernel = kernel
        self.bw_type = bw_type
        self.bw = bw
        self.n_iter = n_iter
        self.min_delta = min_delta

        self.nobs, self.kvar = array_shape(self.data)
        self.data = array_adjust(self.data, self.kvar)

        if self.weights is not None:
            self.weights = array_adjust(self.weights, 0)
            self.weights *= self.nobs / self.weights.sum()
        else:
            self.weights = np.ones(self.nobs)

        if not is_array(self.bw):
            self.bw = np.repeat(self.bw, self.kvar)

        for i in np.arange(self.kvar):
            if isinstance(self.bw[i], str):
                self.bw[i] = self.get_bw_fixed(self.data[:, i], self.weights, self.bw[i])

        self.bw = np.asfarray(self.bw)

    @staticmethod
    def kernel_gaussian(x, p, bw):
        k = array_shape(x)[1]
        x = array_adjust(x, k)

        if is_number(p):
            p = np.array([p])
        p = array_adjust(p, k)

        if is_number(bw):
            bw = np.repeat(bw, k)
        bw = array_adjust(bw, k)

        return np.exp(-np.sum(np.square(p[:, None] - x), axis=-1) / (2 * np.square(bw)))

    @staticmethod
    def kernel_bartlett(bw):
        return 1 - np.arange(bw) / bw

    @staticmethod
    def isj_fixed_point(x, n, i_sq, a_sq):
        i_sq = np.asfarray(i_sq, dtype=np.float64)
        a_sq = np.asfarray(a_sq, dtype=np.float64)

        ell = 7
        f = 0.5 * np.pi ** (2 * ell) * np.sum(np.power(i_sq, ell) * a_sq * np.exp(-i_sq * np.power(np.pi, 2) * x))

        for s in np.arange(2, ell)[::-1]:
            # Step 1: estimate t_s from |f^(s+1)|^2
            k0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(0.5 * np.pi)
            k1 = (1 + 0.5 ** (s + 0.5)) / 3
            time = np.power(k1 * k0 / (n * f), 2 / (3 + 2 * s))

            # Step 2: estimate |f^s| from t_s
            f = 0.5 * np.power(np.pi, 2 * s) * np.sum(
                np.power(i_sq, s) * a_sq * np.exp(-i_sq * np.power(np.pi, 2) * time)
            )

        return x - np.power(2 * n * np.sqrt(np.pi) * f, -0.4)

    @staticmethod
    def isj_root(fn, n, args):
        n = max(min(1050, n), 50)
        bw = 0
        tol = 10e-12 + 0.01 * (n - 50) / 1000

        found = False
        while not found:
            try:
                bw, res = brentq(fn, 0, 0.01, args=args, full_output=True, disp=False)
                found = res.converged
            except ValueError as e:
                bw = 0
                tol *= 2.0
                found = False

            if bw <= 0:
                found = False
            if tol >= 1:
                return

        return bw

    @staticmethod
    def isj_grid(x, weights=None):
        # Histogram the data to get a crude first approximation of the density
        # Use an integer power of 2 bins and a half stdev padding
        nbins = np.power(2, 8)

        sigma = Stat(x, weights=weights).std()
        r_min = x.min() - 0.5 * sigma
        r_max = x.max() + 0.5 * sigma

        # Build histogram
        grid, _ = np.histogram(x, bins=nbins, range=(r_min, r_max), weights=weights)
        grid /= x.shape[0]

        return grid

    @staticmethod
    def get_bw_isj(x, weights=None):
        grid = Kernel.isj_grid(x, weights=weights)
        x_range = x.max() - x.min()
        nobs = x.shape[0]

        dist = np.diff(np.sort(x))
        min_bw = 2 * np.pi * np.mean(dist)
        min_t = np.power(min_bw / x_range, 2)

        # Compute the type 2 Discrete Cosine Transform (DCT) of the data
        a = fftpack.dct(grid)
        i_sq = np.power(np.arange(1, len(grid)), 2)
        a_sq = np.power(a[1:], 2)

        # Solve for the optimal t (minimize AMISE)
        opt_t = Kernel.isj_root(Kernel.isj_fixed_point, nobs, args=(nobs, i_sq, a_sq))

        if not opt_t > min_t:
            return min_bw

        bw = np.sqrt(opt_t) * x_range

        return bw

    @staticmethod
    def get_bw_scott(x, weights=None):
        sigma = Stat(x, weights=weights).std()
        nobs = x.shape[0]

        bw = 1.059 * sigma * np.power(nobs, -0.2)

        return bw

    @staticmethod
    def get_bw_silverman(x, weights=None):
        sigma = Stat(x, weights=weights).std()
        nobs = x.shape[0]

        q75, q25 = np.percentile(x, [75, 25])
        x_iqr = q75 - q25

        a = min(sigma, x_iqr / 1.349)
        bw = 0.9 * a * np.power(nobs, -0.2)

        return bw

    def get_bw_fixed(self, x=None, weights=None, bw=None):
        if x is None:
            x = self.data.copy()
        if weights is None:
            weights = self.weights.copy()
        if bw is None:
            bw = self.bw.copy()

        if isinstance(bw, str):
            if self.verbose > 1:
                print('Estimate BW with method: {}'.format(bw))

            if bw == 'isj':
                bw = self.get_bw_isj(x, weights=weights)
            elif bw == 'silverman':
                bw = self.get_bw_silverman(x, weights=weights)
            elif bw == 'scott':
                bw = self.get_bw_scott(x, weights=weights)
            else:
                raise ValueError('Method `{}` does not exist.'.format(bw))

        return bw

    def get_bw_adaptive(self, x=None, weights=None, bw=None, p=None, n_iter=None, min_delta=None, ret_iter=False):
        if x is None:
            x = self.data.copy()
        if weights is None:
            weights = self.weights.copy()
        if bw is None:
            bw = self.bw.copy()
        if p is None:
            p = x.copy()
        if n_iter is None:
            n_iter = self.n_iter
        if min_delta is None:
            min_delta = self.min_delta or 0

        if self.verbose > 1:
            print('Iterate with params: bw [{}] | n_iter [{}] | min_delta [{}]'.format(
                format_number(bw, 4), format_number(n_iter), format_number(min_delta)))

        kbw = bw * np.ones(p.shape[0])
        kbws = np.empty((n_iter + 1, p.shape[0]))
        kbws[0, :] = kbw

        n = 0
        for n in np.arange(1, n_iter + 1):
            pdf = np.sum(self.kernel_gaussian(x, p, kbw) * weights, axis=1)

            gm = gmean(pdf)
            kbws[n, :] = kbws[n - 1, :] * np.sqrt(gm / pdf)

            delta = np.sqrt(np.mean(np.square(kbws[n] - kbws[n - 1])))

            if delta < min_delta:
                kbw = kbws[n - 1]
                break
            else:
                kbw = kbws[n]

        if self.verbose > 1:
            print('Iterations performed: {}'.format(n))

        if ret_iter:
            return kbw, kbws[:n + 1]

        return kbw

    def get_weights(self, p=None):
        if p is None:
            pred = self.data.copy()
        else:
            pred = np.array(p)

        pred = array_adjust(pred, self.kvar)
        kernel_fn = getattr(self, 'kernel_' + self.kernel)

        kws = np.ones((pred.shape[0], self.nobs, self.kvar))

        for i in np.arange(self.kvar):
            if self.bw_type == 'fixed':
                kws[:, :, i] = kernel_fn(self.data[:, i], pred[:, i], self.bw[i])
            else:
                bws = self.get_bw_adaptive(
                    self.data[:, i],
                    self.weights,
                    self.bw[i],
                    p=pred[:, i]
                )

                kws[:, :, i] = kernel_fn(self.data[:, i], pred[:, i], bws)

        kws = kws.prod(axis=-1)

        return kws


class Estimator(object):

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        verbose: int = 0
    ) -> None:
        self.verbose = verbose  # Print progress

        self.exog = None
        self.endog = None
        self.weights = None

        self.nobs = None
        self.kvar = None

        self.is_ts = None
        self.ranges = None
        self.ts_freq = None

        self.resid = None
        self.pred = None
        self.result = None

        self.build_input(x, y, weights)

    def build_input(
        self,
        x: np.ndarray | pd.Series,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> Self:
        if y is not None:
            self.exog = np.array(x)
            self.endog = np.array(y)
        elif is_pandas(x):
            self.exog = np.array(x.index)
            self.endog = np.array(x)
        else:
            self.exog = np.array(x)

        self.nobs, self.kvar = array_shape(self.exog)
        self.exog = array_adjust(self.exog, self.kvar)

        if self.endog is not None:
            self.endog = array_adjust(self.endog, 1)

        if weights is not None:
            self.weights = array_adjust(weights, 1)
            self.weights *= self.nobs / self.weights.sum()
        else:
            self.weights = np.ones((self.nobs, 1))

        self.is_ts = np.repeat(False, self.kvar)
        self.ranges = []

        for i in np.arange(self.kvar):
            self.ranges.append((self.exog[:, i].min(), self.exog[:, i].max()))

            if is_timeseries(self.exog[:, i]):
                self.is_ts[i] = True

                if self.ts_freq is None:
                    self.ts_freq = Freq.infer(self.exog[:, i])

        for i in np.arange(self.kvar):
            if self.is_ts[i]:
                self.exog[:, i] = ts_to_delta(
                    self.exog[:, i],
                    dt_from=self.ranges[i][0],
                    freq=self.ts_freq
                )

        if self.is_ts.any() and self.ts_freq is None:
            self.ts_freq = 'D'

        self.exog = np.asfarray(self.exog)
        if self.endog is not None:
            self.endog = np.asfarray(self.endog)
        self.weights = np.asfarray(self.weights)

        return self

    def build_pred(
        self,
        p: Optional[np.ndarray] = None
    ) -> Self:
        if p is None:
            self.pred = self.exog.copy()
        else:
            self.pred = np.array(p)

        self.pred = array_adjust(self.pred, self.kvar)

        for i in np.arange(self.kvar):
            if self.is_ts[i]:
                self.pred[:, i] = ts_to_delta(
                    self.pred[:, i],
                    dt_from=self.ranges[i][0],
                    freq=self.ts_freq
                )

        self.pred = np.asfarray(self.pred)

        return self

    @property
    def r2_score(self) -> float:
        y = self.endog.squeeze()
        resid = self.resid.squeeze()
        weights = self.weights.squeeze()

        y_bar = np.average(y, weights=weights)

        ssr = np.sum(np.square(resid))
        sst = np.sum(weights * np.square(y - y_bar))

        return 1 - (ssr / sst)

    def fit(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> Self | pd.Series | pd.DataFrame:
        raise NotImplementedError('Defined in subclasses.')

    def predict(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> pd.Series | pd.DataFrame:
        raise NotImplementedError('Defined in subclasses.')


class LeastSquaresEstimator(Estimator):

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        poly_deg: int = 1,
        cov_type: Optional[Literal['hac']] = 'hac',
        cov_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0
    ) -> None:
        self.poly_deg = poly_deg
        self.cov_type = cov_type
        self.cov_kwargs = cov_kwargs if cov_kwargs is not None else dict()

        self.wexog = None
        self.wendog = None
        self.wresid = None

        self.dof = None  # Degrees of freedom
        self.scale = None  # Variance scale parameter
        self.covb = None  # Covariance params
        self.mcov = None  # Covariance matrix
        self.coef = None  # Regression coefficients

        self.serr = None  # Standard error of the mean
        self.perr = None  # Standard error of the predictions
        self.cint = None  # Prediction intervals

        super().__init__(x, y, weights=weights, verbose=verbose)

    def build_input(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> Self:
        super().build_input(x, y, weights)

        if self.endog is not None:
            vind = np.isfinite(self.endog).any(axis=1)

            self.exog = self.exog[vind]
            self.endog = self.endog[vind]
            self.weights = self.weights[vind]

            self.nobs = self.exog.shape[0]

        if self.poly_deg == 1:
            exog = np.column_stack([np.ones(self.exog.shape[0]), self.exog])
        else:
            exog = np.ones(self.exog.shape[0])
            for p in np.arange(1, self.poly_deg + 1):
                for combination in itertools.combinations_with_replacement(np.arange(self.exog.shape[1]), p):
                    term = np.prod(self.exog[:, combination], axis=1)
                    exog = np.column_stack([exog, term])

        self.exog = exog
        self.dof = self.nobs - self.poly_deg - self.kvar

        # Whitened design matrix (add constant term)
        self.wexog = np.sqrt(self.weights) * self.exog
        self.wendog = np.sqrt(self.weights) * self.endog

        return self

    def build_pred(
        self,
        p: Optional[np.ndarray] = None
    ) -> Self:
        super().build_pred(p)

        if self.pred.shape[1] <= self.kvar:
            if self.poly_deg == 1:
                pred = np.column_stack([np.ones(self.pred.shape[0]), self.pred])
            else:
                pred = np.ones(self.pred.shape[0])
                for p in np.arange(1, self.poly_deg + 1):
                    for combination in itertools.combinations_with_replacement(np.arange(self.pred.shape[1]), p):
                        term = np.prod(self.pred[:, combination], axis=1)
                        pred = np.column_stack([pred, term])

            self.pred = pred

        return self

    def get_robust_cov(
        self,
        cov_type: Optional[Literal['hac']] = None,
        hac_lags: int = 1,
        kernel: Literal['bartlett'] = 'bartlett'
    ) -> np.ndarray:
        if cov_type == 'hac':
            # Newey-West (HAC) with kernel weights
            kernel_fn = getattr(Kernel, 'kernel_' + kernel)

            xu = self.wexog * self.wresid

            kws = kernel_fn(hac_lags + 1)
            sigma = kws[0] * np.dot(xu.T, xu)
            for lag in np.arange(1, hac_lags + 1):
                s = np.dot(xu[lag:].T, xu[:-lag])
                sigma += kws[lag] * (s + s.T)

            mcov = self.covb @ sigma @ self.covb.T  # HAC adjusted variance/covariance matrix
            if self.dof > 0:
                mcov *= self.nobs / self.dof  # Smallpop correction
        else:
            raise ValueError('Param `cov_type` does not exist.')

        return mcov

    @property
    def r2_score(self) -> float:
        resid = self.wresid.squeeze()
        y = self.wendog.squeeze()
        y_bar = np.mean(y)

        ssr = np.sum(np.square(resid))
        sst = np.sum(np.square(y - y_bar))

        return 1 - (ssr / sst)

    def fit(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> Self | pd.Series | pd.DataFrame:
        wexog_pinv = np.linalg.pinv(self.wexog)  # Moore-Penrose pseudo-inverse matrix
        self.coef = np.dot(wexog_pinv, self.wendog)  # Model coeficients

        self.resid = self.endog - np.dot(self.exog, self.coef)
        self.wresid = self.wendog - np.dot(self.wexog, self.coef)

        self.covb = np.dot(wexog_pinv, wexog_pinv.T)
        self.scale = np.dot(self.wresid.squeeze(), self.wresid.squeeze()) / self.dof
        self.mcov = self.covb * self.scale

        if self.cov_type is not None:
            self.mcov = self.get_robust_cov(self.cov_type, **self.cov_kwargs)

        self.serr = np.sqrt(self.scale)

        if p is not None:
            return self.predict(p, alpha)

        return self

    def predict(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> pd.Series | pd.DataFrame:
        self.build_pred(p)

        y_hat = np.dot(self.pred, self.coef).squeeze()

        result = pd.Series(y_hat, index=pd.MultiIndex.from_arrays(self.pred[:, 1:(self.kvar + 1)].T), name='mean')

        if alpha is not None:
            self.perr = np.sqrt(np.diag(self.pred @ self.mcov @ self.pred.T))
            self.cint = t.ppf(1 - alpha / 2, self.dof) * self.perr

            result = pd.DataFrame(result)
            result['cmin'] = y_hat - self.cint
            result['cmax'] = y_hat + self.cint
            result['err'] = self.perr

        for i in np.arange(self.kvar):
            if self.is_ts[i]:
                result.index = result.index.set_levels(
                    ts_from_delta(
                        result.index.get_level_values(i),
                        dt_from=self.ranges[i][0],
                        freq=self.ts_freq
                    ),
                    level=i
                )

        if self.kvar == 1:
            result.index = result.index.get_level_values(0)
            if self.is_ts[0]:
                result = result.resample(self.ts_freq.pd_period).asfreq()

        self.result = result.astype(float)

        return self.result


class LocalKernelEstimator(Estimator):

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        kernel: Literal['gaussian'] = 'gaussian',
        bw_type: Literal['fixed', 'adaptive'] = 'adaptive',
        bw: int | float | tuple[int | float] | Literal['scott', 'silverman', 'isj'] = 'isj',
        bw_kwargs: Optional[dict[str, Any]] = None,
        poly_deg: int = 1,
        cov_type: Optional[Literal['hac']] = 'hac',
        cov_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0
    ) -> None:
        self.kernel = kernel
        self.bw_type = bw_type
        self.bw = bw
        self.bw_kwargs = bw_kwargs if bw_kwargs is not None else dict()

        self.poly_deg = poly_deg
        self.cov_type = cov_type
        self.cov_kwargs = cov_kwargs if cov_kwargs is not None else dict()

        super().__init__(x, y, weights=weights, verbose=verbose)

    @property
    def r2_score(self) -> float:
        y = self.endog.squeeze()
        weights = self.weights.squeeze()

        y_hat = self.predict().values
        y_bar = np.average(y_hat, weights=weights)

        r2_numer = np.square(np.sum(weights * (y - y_bar) * (y_hat - y_bar)))
        r2_denom = np.sum(weights * np.square(y - y_bar)) * np.sum(weights * np.square(y_hat - y_bar))

        return r2_numer / r2_denom

    def get_kernel(self) -> Kernel:
        return Kernel(
            data=self.exog,
            weights=self.weights,
            kernel=self.kernel,
            bw_type=self.bw_type,
            bw=self.bw,
            **self.bw_kwargs
        )

    def get_local_estimator(
        self,
        kws: np.ndarray,
        pos: int
    ) -> Optional[LeastSquaresEstimator]:
        weights = kws[pos] * self.weights.squeeze()
        vind = np.abs(weights) >= 1e-2

        if not np.isfinite(self.endog[vind]).any():
            return

        return LeastSquaresEstimator(
            self.exog[vind],
            self.endog[vind],
            weights=weights[vind],
            poly_deg=self.poly_deg,
            cov_type=self.cov_type,
            cov_kwargs=self.cov_kwargs
        )

    def fit(
        self,
            p: Optional[np.ndarray] = None,
            alpha: Optional[float] = None
    ) -> Self | pd.Series | pd.DataFrame:
        if p is not None:
            return self.predict(p, alpha)

        return self

    def predict(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> pd.Series | pd.DataFrame:
        self.build_pred(p)

        kws = self.get_kernel().get_weights(self.pred)

        if alpha is not None:
            result = pd.DataFrame(
                columns=['mean', 'cmin', 'cmax', 'err', 'nobs'],
                index=pd.MultiIndex.from_arrays(self.pred.T)
            )
        else:
            result = pd.Series(
                index=pd.MultiIndex.from_arrays(self.pred.T),
                name='mean'
            )

        locs_ = tqdm(np.arange(self.pred.shape[0])) if self.verbose > 1 else np.arange(self.pred.shape[0])
        for pos in locs_:
            loc_est = self.get_local_estimator(kws, pos)

            if loc_est is not None:
                rloc = loc_est.fit(self.pred, alpha=alpha)

                if not isinstance(rloc.index, pd.MultiIndex):
                    rloc.index = pd.MultiIndex.from_arrays([rloc.index])

                if alpha is not None:
                    rloc['nobs'] = loc_est.nobs

                result.loc[tuple(self.pred[pos])] = rloc.loc[tuple(self.pred[pos])]

        for i in np.arange(self.kvar):
            if self.is_ts[i]:
                result.index = result.index.set_levels(
                    ts_from_delta(
                        result.index.get_level_values(i),
                        dt_from=self.ranges[i][0],
                        freq=self.ts_freq
                    ),
                    level=i
                )

        if self.kvar == 1:
            result.index = result.index.get_level_values(0)
            if self.is_ts[0]:
                result = result.resample(self.ts_freq.pd_period).asfreq()

        self.result = result.astype(float)

        return self.result


class KernelDensityEstimator(Estimator):

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        kernel: Literal['gaussian'] = 'gaussian',
        bw_type: Literal['fixed', 'adaptive'] = 'fixed',
        bw: int | float | Literal['scott', 'silverman', 'isj'] = 'scott',
        bw_kwargs: Optional[dict[str, Any]] = None,
        outliers: Literal['keep', 'group', 'remove'] = 'keep',
        n_dev: int = 3,
        verbose: int = 0
    ) -> None:
        self.data = None

        self.kernel = kernel
        self.bw_type = bw_type
        self.bw = bw
        self.bw_kwargs = bw_kwargs if bw_kwargs is not None else dict()

        self.outliers = outliers
        self.n_dev = n_dev

        super().__init__(x, y, weights=weights, verbose=verbose)

    def build_input(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> Self:
        super().build_input(x, y, weights)

        self.data = self.exog.copy()
        if self.endog is not None:
            self.data = np.column_stack([self.data, self.endog])

        self.nobs, self.kvar = array_shape(self.data)

        self.process_outliers()

        return self

    def build_pred(
        self,
        p: Optional[np.ndarray] = None
    ) -> Self:
        if is_number(p):
            npts = int(p)
            p = np.linspace(self.data[:, 0].min(), self.data[:, 0].max(), npts)
            if self.kvar > 1:
                p = np.column_stack([p, np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), npts)])
        elif is_array(p):
            p = array_adjust(p, self.kvar)
            p = p[(self.data[:, 0].min() <= p[:, 0]) & (p[:, 0] <= self.data[:, 0].max())]
            if self.kvar > 1:
                p = np.column_stack([p, p[(self.data[:, 1].min() <= p[:, 1]) & (p[:, 1] <= self.data[:, 0].max())]])

        self.pred = array_adjust(p, self.kvar)

        return self

    def process_outliers(self) -> Self:
        if self.outliers == 'keep':
            return self

        x = self.data[:, 0]
        y = self.data[:, -1] if self.kvar > 1 else None
        w = self.weights.squeeze()

        x_samp = Stat(x, weights=w, dropna=True, outliers=self.outliers, n_dev=self.n_dev)
        if y is not None:
            y_samp = Stat(y, weights=w, dropna=True, outliers=self.outliers, n_dev=self.n_dev)
            vind = np.all(np.vstack([x_samp.vind, y_samp.vind]), axis=0)
        else:
            vind = x_samp.vind

        x = x[vind]
        if y is not None:
            y = y[vind]
        w = w[vind]

        self.data = array_adjust(x, 1)
        if self.kvar > 1:
            self.data = np.column_stack([self.data, array_adjust(y, 1)])
        self.weights = array_adjust(w, 1)

        self.nobs, self.kvar = array_shape(self.data)

        return self

    @property
    def r2_score(self) -> float:
        return np.NaN

    def get_kernel(self) -> Kernel:
        return Kernel(
            data=self.data,
            weights=self.weights,
            kernel=self.kernel,
            bw_type=self.bw_type,
            bw=self.bw,
            **self.bw_kwargs
        )

    def fit(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> Self | pd.Series | pd.DataFrame:
        if p is not None:
            return self.predict(p, alpha)

        return self

    def predict(
        self,
        p: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> pd.Series | pd.DataFrame:
        self.build_pred(p)

        if self.kvar > 1:
            xx, yy = np.meshgrid(self.pred[:, 0], self.pred[:, 1], indexing='ij')
            pred = np.vstack([xx.ravel(), yy.ravel()]).T
        else:
            pred = self.pred.copy()

        kernel = self.get_kernel()
        kws = self.weights.squeeze() * kernel.get_weights(pred)

        result = np.sum(kws, axis=1)

        if self.kvar > 1:
            result = pd.DataFrame(
                result.reshape(np.repeat(self.pred.shape[0], self.kvar)).T
            ) / self.nobs
        else:
            result = pd.Series(
                result, index=self.pred.squeeze(), name='kde'
            ) / (self.nobs * kernel.bw)

        self.result = result.astype(float)

        return self.result
