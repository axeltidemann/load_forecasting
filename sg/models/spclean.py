"""A filter for smoothing time series data using B-splines. Input is a time
series and a smoothness indicator, output is the cleaned data or the fitted
data with or without the confidence interval."""

import cPickle as pickle
import hashlib
import sys
import os
from tempfile import NamedTemporaryFile
import time
import array
import ctypes
from ctypes import cdll
from ctypes import c_double

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate as si
from numpy import dot, transpose
from numpy.linalg import solve, inv

import sg.utils
import splines as sp
from sg.utils.cache import ATimeCache
from sg.utils.timer import SimpleTimer

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH_TO_BSPLINE_CACHE_PHI = os.path.join(_PATH_TO_HERE,
                                          ".bspline_cache_phi.pickle")
_PATH_TO_BSPLINE_CACHE_ROUGHNESS = os.path.join(_PATH_TO_HERE,
                                            ".bspline_cache_roughness.pickle")

class BSplineBasicSmoother(object):
    """This class implements the B-spline smoothing described in Chen et
    al."Automated Load Curve Data Cleansing in Power Systems", IEEE
    Transactions on Smart Grid 1(2) 2010.

    This implementation assumes an equal number of knots and control points,
    with knots uniformly distributed over the dataset and no knots with
    multiplicity > 1 except at the endpoints.

    This is the slow, manual implementation. Use BSplineSmoother instead."""
        
    def __init__(self, dataset, smoothness, knots=None, num_knots=None):
        """Initialize the smoother.  Control points will be estimated given the
        knots and the smoothness parameter.  Non-uniform knots may be provided
        through the 'knots' argument, see the method 'set_knots' for more
        info.
        """
        self._degree = 3 # Set based on Fig. 2 of Chen et al.
        self._dataset = dataset
        self._knots = None
        self._smoothness = smoothness
        self.set_knots(knots, num_knots)
        
    def set_knots(self, knots=None, num_knots=None):
        """Set or reset knots. If both 'knots' and 'num_knots' are None,
        there will be one knot for each data point, with the necessary
        multiplicity at the ends for the curve to start and end at the first
        and last data point, respectively."""
        if knots is not None and num_knots is not None:
            raise RuntimeError("Either knots or num_knots may " \
                               "be specified, but not both.")
        if knots is not None:
            self._knots = knots
        elif num_knots is not None:
            self._knots = sp.get_uniform_knots(num_knots, self._degree,
                knotrange=(0, len(self._dataset)))
        else:
            self._knots = sp.get_uniform_knots_from_points(
                self._dataset, self._degree,
                knotrange=(0, len(self._dataset) - 1))

    @property
    def knots(self):
        return self._knots

    def get_unique_knots(self):
        """Same as knots property, but will only return a single knot for each
        position, regardless of multiplicity."""
        return np.unique(self._knots)

    @property
    def dataset(self):
        return self._dataset

    def set_dataset(self, dataset, knots=None, num_knots=None):
        """Set new data. Calls set_knots to reset knots. Note that if the knots
        are unchanged (e.g. by sending a new dataset of the same length as the
        previous one and not passing in knots or num_knots, the knots will be
        unchanged and, consequently, the phi and R matrices do not need to be
        recomputed."""
        self._dataset = dataset
        self.set_knots(knots, num_knots)

    @property
    def smoothness(self):
        return self._smoothness
    
    def set_smoothness(self, smoothness):
        self._smoothness = smoothness
        
    def _get_num_coefficients(self):
        """Return the number of coefficients, which is given by the degree and
        the number of knots."""
        return len(self._knots) - self._degree - 1

    def _get_roughness(self):
        """Calculate the "R" matrix as the product of the second derivative of
        the basis functions with its transpose, integrated over time. Returns a
        k*k matrix, where k is the number of basis functions (i.e. the number
        of coefficients). Implements eq. 10 of Chen et al."""
        k = self._get_num_coefficients()
        r = np.zeros((k, k))
        p = self._degree
        d2 = [sp.bsplinebasis_deriv(self._knots, i, p, 2) for i in range(k)]
        t_min = self._knots[0]
        t_max = self._knots[-1]
        for i in range(k):
            jmin = max(0, i - self._degree)
            jmax = min(k, i + self._degree + 1)
            for j in range(jmin, jmax):
                discont = self._knots[min(i, j):max(i, j) + self._degree + 1]
                y, err = scipy.integrate.quad(lambda t: d2[i](t) * d2[j](t),
                                              t_min, t_max, points=discont)
                r[i, j] = y
        return r

    def _get_phi(self):
        """Calculate the n*k matrix where element i,j is the value at time i of
        the j'th b-spline basis function. Implements eq. 5 of Chen et al."""
        k = self._get_num_coefficients()
        n = len(self._dataset)
        phi = np.zeros((n, k))
        bases = [sp.bsplinebasis(self._knots, i, self._degree) \
                 for i in range(k)]
        t_range = self._knots[-1] - self._knots[0]
        for i in range(n):
            for j in range(k):
                t = float(i) / (n - 1) * t_range
                phi[i, j] = bases[j](t)
        return phi

    def _get_hatmatrix_core(self):
        """Calculate most of the hat matrix S, by setting the derivative of
        PENSSE wrt c to zero. Implements the first part of eq. 12 of Chen et
        al. In order to form the actual hat matrix, the output from this
        function must be pre-multiplied with phi."""
        phi = self._get_phi()
        phi_t = transpose(phi)
        phi_2 = dot(phi_t, phi)
        r = self._get_roughness()
        # Use linalg.solve to avoid the matrix inversion. Potentially more
        # numerically stable.
        #hm_core = dot(inv(phi_2 + self.smoothness * r), phi_t)
        #hm_core = np.linalg.solve(phi_2 + self.smoothness * r, phi_t)
        hm_core = sg.utils.qr_solve(phi_2 + self.smoothness * r, phi_t)
        return (hm_core, phi)

    def _get_coefficients(self):
        """Use the hat matrix to update the coefficients.  Implements the
        second part of eq. 12 of Chen et al."""
        (hm_core, phi) = self._get_hatmatrix_core()
        return dot(hm_core, self._dataset)

    def get_hatmatrix(self):
        """Calculate the hat matrix S, which when post-multiplied with the
        observed data y will give the estimated/smoothed data points <y>."""
        (hm_core, phi) = self._get_hatmatrix_core()
        return dot(phi, hm_core)

    def _square_hatmatrix(self, s):
        s_t = np.transpose(s)
        return np.dot(s, s_t)
        
    def get_hatmatrix_squared(self):
        s = self.get_hatmatrix()
        return self._square_hatmatrix(s)

    def splev(self, time):
        """Evaluate the spline at time given the current knots and
        coefficients. Coefficients will be calculated if they haven't been set
        already, otherwise cached coefficients will be used."""
        coeffs = self._get_coefficients()
        tck = (self._knots, coeffs, self._degree)
        y = si.splev(time, tck)
        return y

    def get_smoothed_data(self):
        """Extract the smoothed data directly at the data points by evaluating
        the hat matrix (eq. 14 of Chen et al)."""
        S = self.get_hatmatrix()
        return dot(S, self._dataset)

class BSplineAnalyticSmoother(BSplineBasicSmoother):
    """This class calculates the integral in the roughness matrix analytically
    rather than numerically for the special case where the B-spline degree is
    3.  See superclass for more info."""
        
    def __init__(self, dataset, smoothness, knots=None, num_knots=None):
        """See superclass for info."""
        BSplineBasicSmoother.__init__(self, dataset, smoothness,
                                      knots, num_knots)
    def _get_roughness(self):
        """Calculate the "R" matrix analytically if the B-spline degree is 3,
        taking advantage of integration by parts and the fact that the 3rd
        derivative of a 3rd-degree B-spline is a constant. Further description
        in roughness.tex."""
        p = self._degree
        if p != 3:
            return BSplineBasicSmoother._get_roughness(self)
        k = self._get_num_coefficients()
        r = np.zeros((k, k))
        bases = [sp.bsplinebasis(self._knots, i, p) for i in range(k)]
        d1 = [sp.bsplinebasis_deriv(self._knots, i, p, 1) for i in range(k)]
        d2 = [sp.bsplinebasis_deriv(self._knots, i, p, 2) for i in range(k)]
        t = self._knots
        t_min = t[0]
        t_max = t[-1]
        for i in range(k):
            if t[i+1] > t[i]:
                c_0 = 6./((t[i+3] - t[i]) * (t[i+2] - t[i]) * (t[i+1] - t[i]))
            else:
                c_0 = 0

            div = ((t[i+3] - t[i])*(t[i+2] - t[i])) 
            l1 = 1. / div if div > 0 else 0
            div = ((t[i+3] - t[i])*(t[i+3] - t[i+1]))
            l2 = 1. / div if div > 0 else 0
            div = ((t[i+4] - t[i+1])*(t[i+3] - t[i+1]))
            l3 = 1. / div if div > 0 else 0
            c_1 = - 6./(t[i+2] - t[i+1]) * (l1 + l2 + l3) if t[i+2] > t[i+1] else 0

            div = ((t[i+3] - t[i])*(t[i+3] - t[i+1]))
            l1 = 1. / div if div > 0 else 0
            div = ((t[i+4] - t[i+1])*(t[i+3] - t[i+1]))
            l2 = 1. / div if div > 0 else 0
            div = ((t[i+4] - t[i+1])*(t[i+4] - t[i+2]))
            l3 = 1. / div if div > 0 else 0
            c_2 = 6./(t[i+3] - t[i+2]) * (l1 + l2 + l3) if t[i+3] > t[i+2] else 0

            if t[i+4] > t[i+3]:
                c_3 = - 6./((t[i+4] - t[i+1]) * (t[i+4] - t[i+2]) * (t[i+4] - t[i+3]))
            else:
                c_3 = 0
                
            jmin = max(0, i - self._degree)
            jmax = min(k, i + self._degree + 1)
            for j in range(jmin, jmax):
                phi = bases[j]
                r[i, j] = d2[i](t_max) * d1[j](t_max) - d2[i](t_min) * d1[j](t_min) \
                  - c_0 * (phi(t[i+1]) - phi(t[i])) \
                  - c_1 * (phi(t[i+2]) - phi(t[i+1])) \
                  - c_2 * (phi(t[i+3]) - phi(t[i+2])) \
                  - c_3 * (phi(t[i+4]) - phi(t[i+3]))
        return r
    
class MatrixCache():
    """Class that caches B-spline basis function related matrices. Externally,
    it uses the knots as keys. Internally, it creates a hash digest of the
    knots and uses this as key. It then verifies the stored knots against the
    given knots to make sure there is not a hashkey collision."""

    def __init__(self, path=None):
        """Initialize the cache. If path is given, load an existing cache from
        persistent storage."""
        if path is None:
            self.cache = dict()
        else:
            with open(path, "r") as f:
                self.cache = pickle.load(f)
            if type(self.cache) != dict:
                raise ValueError("Failed to read a dict from the given cache " \
                                 "file %s" % path)

    def has_key(self, knots):
        assert isinstance(knots, np.ndarray)
        key = self._get_hash(knots)
        if self.cache.has_key(key):
            return np.all(self.cache[key][0] == knots)
        else:
            return False

    def store(self, path):
        """Open the file given in 'path' and store the cache."""
        # Pickling large caches may take quite a while, and an interrupt during
        # pickling will render the cache unusable. Therefore we store by means
        # of a temporary file, which is renamed to the given path after
        # pickling has completed. This reduces the risk of error
        # considerably. The operation will not work on Windows, because
        # os.rename fails when the target file exists.
        tmp_path = None
        try:
            with NamedTemporaryFile(prefix="_%s." % os.path.basename(path), 
                                    suffix=".temporary", dir=_PATH_TO_HERE,
                                    delete=False) as f:
                tmp_path = f.name
                pickle.dump(self.cache, f)
            os.rename(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                print >>sys.stderr, "Exception raised while storing cache. " \
                  "Removing temporary cache file."
                os.remove(tmp_path)
            
    def _get_hash(self, knots):
        assert isinstance(knots, np.ndarray)
        m = hashlib.md5()
        m.update(knots.data)
        return m.digest()

    def __eq__(self, other):
        try:
            if len(self.cache) != len(other.cache):
                return False
            for ((key_self, (knots_self, mtx_self))) in self.cache.iteritems():
                (knots_other, mtx_other) = other.cache[key_self]
                if np.any(knots_self != knots_other) or \
                    np.any(mtx_self != mtx_other):
                    return False
            return True
        except:
            return False
    
    def __len__(self):
        return self.cache.__len__()

    def __getitem__(self, knots):
        assert isinstance(knots, np.ndarray)
        key = self._get_hash(knots)
        value = self.cache.__getitem__(key)
        if np.all(value[0] == knots):
            return value[1]
        else:
            raise KeyError("Retrieval from cache failed due to a hashkey "
                           "collision. Your cache unfortunately contains "
                           "another knot sequence that results in the same "
                           "hashkey. Try to delete the cache. If the problem "
                           "occurs frequently, consider reimplementing "
                           "_get_hash with another hashing algorithm ")
        
    def __setitem__(self, knots, value):
        assert isinstance(knots, np.ndarray)
        key = self._get_hash(knots)
        if self.cache.has_key(key) and np.any(self.cache[key][0] != knots):
            raise KeyError("Storing to cache failed due to a hashkey "
                           "collision. Your cache unfortunately contains "
                           "another knot sequence that results in the same "
                           "hashkey. Try to delete the cache. If the problem "
                           "occurs frequently, consider reimplementing "
                           "_get_hash with another hashing algorithm ")
        self.cache.__setitem__(key, (knots, value))

    def __contains__(self, item):
        return self.has_key(item)

    def __str__(self):
        return self.cache.__str__()


class BSplineSmoother(BSplineAnalyticSmoother):
    """This class speeds up the B-spline smoothing performed by the superclass
    BSplineAnalyticSmoother (BSplineBasicSmoother). All the actual work is done
    in the superclasses. See superclasses for more info."""
        
    def __init__(self, dataset, smoothness, knots=None, num_knots=None):
        """See superclass for info."""
        try:
            self._phi_cache = MatrixCache(_PATH_TO_BSPLINE_CACHE_PHI)
        except:
            print >>sys.stderr, "Failed to load PHI cache, recreating."
            self._phi_cache = MatrixCache()
        try:
            self._roughness_cache = MatrixCache(_PATH_TO_BSPLINE_CACHE_ROUGHNESS)
        except:
            print >>sys.stderr, "Failed to load ROUGHNESS cache, recreating."
            self._roughness_cache = MatrixCache()
        self._hatmatrix_cache = ATimeCache(100)
        self._phi = None
        self._roughness = None
        self._hatmatrix = None
        self._smoothed_data = None
        BSplineAnalyticSmoother.__init__(self, dataset, smoothness,
                                      knots, num_knots)

    def _get_hatmatrix_cache_key(self):
        return (tuple(self.knots), self.smoothness)

    def _get_hatmatrix_from_cache(self):
        try:
            self._hatmatrix = self._hatmatrix_cache[self._get_hatmatrix_cache_key()]
        except KeyError:
            self._hatmatrix = None
        
    def set_knots(self, knots=None, num_knots=None):
        """See superclass for info."""
        old_knots = self.knots
        BSplineAnalyticSmoother.set_knots(self, knots, num_knots)
        if not np.all(old_knots == self.knots):
            try:
                self._phi = self._phi_cache[self.knots]
            except KeyError:
                self._phi = None
            try:
                self._roughness = self._roughness_cache[self.knots]
            except KeyError:
                self._roughness = None
            self._get_hatmatrix_from_cache()

    def set_dataset(self, dataset, knots=None, num_knots=None):
        BSplineAnalyticSmoother.set_dataset(self, dataset, knots, num_knots)
        self._smoothed_data = None

    def set_smoothness(self, smoothness):
        old_smoothness = self.smoothness
        BSplineAnalyticSmoother.set_smoothness(self, smoothness)
        if old_smoothness != self.smoothness:
            self._get_hatmatrix_from_cache()
        self._smoothed_data = None

    def _calc_and_cache_hatmatrix(self):
        hm = BSplineAnalyticSmoother.get_hatmatrix(self)
        hm2 = BSplineAnalyticSmoother._square_hatmatrix(self, hm)
        self._hatmatrix = (hm, hm2)
        self._hatmatrix_cache[self._get_hatmatrix_cache_key()] = self._hatmatrix

    def get_hatmatrix(self):
        """See superclass for info."""
        if self._hatmatrix is None:
            self._calc_and_cache_hatmatrix()
        return self._hatmatrix[0]

    def get_hatmatrix_squared(self):
        if self._hatmatrix is None:
            self._calc_and_cache_hatmatrix()
        return self._hatmatrix[1]
    
    def get_smoothed_data(self):
        """See superclass for info."""
        if self._smoothed_data is None:
            self._smoothed_data = BSplineAnalyticSmoother.get_smoothed_data(self)
        return self._smoothed_data

    def _get_from_cache(self, function, value, cache, path_to_storage):
        """Get the desired value. If possible, try to retrieve from cache. If
        not in cache, calculate and store."""
        if value is None:
            value = function(self)
            cache[self._knots] = value
            cache.store(path_to_storage)
        return value
        
    def _get_roughness(self):
        """See superclass for info."""
        self._roughness = self._get_from_cache(
            BSplineAnalyticSmoother._get_roughness, self._roughness,
            self._roughness_cache, _PATH_TO_BSPLINE_CACHE_ROUGHNESS)
        return self._roughness

    def _get_phi(self):
        """See superclass for info."""
        self._phi = self._get_from_cache(
            BSplineAnalyticSmoother._get_phi, self._phi,
            self._phi_cache, _PATH_TO_BSPLINE_CACHE_PHI)
        return self._phi

# class FitpackSplineSmoother(object):
#     """This class implements smoothing by use of the
#     Fitpack/dierckx/scipy.interpolate routines. It presents an interface
#     similar to the B-spline smoothers based on Chen et al, so that the
#     regression cleaner can alternate between the two different smoother
#     implementations."""

#     def __init__(self, dataset=None, smoothness=0):
#         self._degree = 3
#         self._dataset = dataset
#         self._smoothness = smoothness
#         self._set_dirty()

#     def _get_x(self):
#         return np.arange(len(self.dataset))
    
#     def _get_representation(self):
#         if self._is_dirty():
#             x = self._get_x()
#             nx = len(x)
#             t_smooth = np.linspace(
#                 x[0], x[-1], min(nx, max(5, int(nx * (1-self._smoothness)))))[2:-2]
#             self.tck = si.splrep(
#                 self._get_x(), self.dataset, k=self._degree, t=t_smooth)
#             self._set_dirty(False)
#         return self.tck

#     def _set_dirty(self, dirty=True):
#         self._dirty = dirty

#     def _is_dirty(self):
#         return self._dirty is None or self._dirty
    
#     @property
#     def dataset(self):
#         return self._dataset

#     def set_dataset(self, dataset):
#         """Set new data."""
#         self._dataset = dataset
#         self._set_dirty()

#     @property
#     def knots(self):
#         tck = self._get_representation()
#         return tck[0]

#     @property
#     def smoothness(self):
#         return self._smoothness
    
#     def set_smoothness(self, smoothness):
#         if not 0 <= smoothness <= 1:
#             raise ValueError("Smoothness must be in the range [0, 1].")
#         self._smoothness = smoothness
#         self._set_dirty()
        
#     def splev(self, time):
#         """Evaluate the spline at time given the current data and smoothness."""
#         tck = self._get_representation()
#         y = si.splev(time, tck)
#         return y

#     def get_smoothed_data(self):
#         """Extract the smoothed data at the data points."""
#         return self.splev(np.arange(len(self._dataset)))


class RegressionCleaner():
    """This class uses a non-parametric regression class (currently
    BSplineSmoother) to implement the cleaning described in sec. IV.A of Chen
    et al. 'Automated Load Curve Data Cleansing in Power Systems', IEEE
    Transactions on Smart Grid 1(2) 2010."""
        
    def __init__(self, smoother, zscore=1.96):
        """Initialize the cleaner. smoother is and instance of a smoother class
        that can calculate estimated smoothed data points and a hat matrix
        (i.e. up to and including eq. 15 of Chen et al).

        Typical confidence intervals and their corresponding z-score:
        50%: 0.67, 68%: 1.00, 90%: 1.64, 95%: 1.96, 99%: 2.58.
        """
        self._sm = smoother
        self._zscore = zscore
        self._observed = smoother.dataset
        self._smoothed = smoother.get_smoothed_data()
        self._calc_mse()
        self._calc_var_matrix()
        self._calc_predicted_errors()
        
    def _calc_predicted_errors(self):
        """Calculate the predicted errors , eq. 22 of Chen et al."""
        s2 = np.diag(self._var)
        self._predicted_errors = np.sqrt(s2 + self._mse)

    def _calc_mse(self):
        """Calculate mean square error, eq. 23 of Chen et al."""
        total_se = np.power(self._observed - self._smoothed, 2).sum()
        df = np.trace(self._sm.get_hatmatrix())
        divisor = (len(self._observed) - df)
        # Very low total squared error may occur if smoothness is low.  In this
        # case, the degrees of freedom are also very small, and we risk a
        # division by zero.
        if abs(divisor) < 1e-6:
            if abs(total_se) < 1e-6:
                if self._sm.smoothness > 1e-12:
                    print >>sys.stderr, "Warning: Many degrees of freedom " \
                        "and total square error extremely small. This " \
                        "probably occurs due to an attempt to clean the data "\
                        "using a very low smoothing factor. Setting MSE to 0." 
                self._mse = 0
                return
            else:
                raise RuntimeError("Many degrees of freedom but large square "\
                                   "error! Something is wrong...")
        self._mse = total_se / divisor

    def _calc_var_matrix(self):
        """Calculate sampling variance to fit matrix, eq. 24 of Chen et al."""
        sst = self._sm.get_hatmatrix_squared()
        self._var = sst * self._mse
 
    def get_confidence_interval(self):
        """Returns two arrays, giving the lower and upper bound of the
        confidence interval at each data point (eq. 25 of Chen et al)."""
        lower = self._smoothed - self._predicted_errors * self._zscore
        upper = self._smoothed + self._predicted_errors * self._zscore
        return (lower, upper)

    def get_outliers(self):
        bounds = self.get_confidence_interval()
        (lower, upper) = bounds
        return np.where(np.logical_or(self._observed < lower,
                                          self._observed > upper))[0]
        
    def get_cleaned_data(self, method):
        """Returns the cleaned dataset and the outlier indices. Data points
        that are outside the confidence interval are cleaned by applying
        'method' to them, where 'method' is a pointer to a method in this class
        or a subclass."""
        cleaned = self._observed.copy()
        outliers = self.get_outliers()
        for outlier in outliers:
            cleaned[outlier] = method(self, outlier, cleaned)
        return (cleaned, outliers)

    def replace_with_estimate(self, outlier_idx, cleaned):
        """Set the cleaned value at index outlier_idx to the smoothed
        estimate."""
        return self._smoothed[outlier_idx]

    def replace_with_previous(self, outlier_idx, cleaned):
        """Set the cleaned value at index outlier_idx to the previously
        observed value."""
        if outlier_idx > 0:
            return cleaned[outlier_idx - 1]
        else:
            return self.replace_with_estimate(outlier_idx)
    
    def replace_with_bound(self, outlier_idx, cleaned):
        """Set the cleaned value at index outlier_idx to the corresponding
        bound of the confidence interval."""
        bounds = self.get_confidence_interval()
        (lower, upper) = bounds[0][outlier_idx], bounds[1][outlier_idx]
        if self._observed[outlier_idx] < lower:
            return lower
        else:
            return upper

        
class BsplineFastSmoother(object):
    def __init__(self, data_p, n_data, knot_p, n_knot, degree, smoothness, zscore):
	#dataset, n_data, knots, n_knot, degree, smoothness, zscore, n_threads):
        #self._data = data
        #self._datalen = n_data
        #number of threads
    
        self._lib = cdll.LoadLibrary('lib_mkl/libspclean.so')
        #self._obj = self._lib.Soother_new(ds_array, n_data, kn_array, n_knot, 
        #                                  degree, c_double(smoothness), 
        #                                  c_double(zscore), n_threads)
        self._obj = self._lib.Smoother_new(data_p, n_data, knot_p, n_knot, 
                                          degree, smoothness, zscore)

    def __del__(self):
        self._lib.Smoother_delete(self._obj)
        
    def bsm_cleanData(self):
        #res = self._lib.bsm_cleanData(self._obj)
        #convert the pointer to nparray
        #ArrayType = ctypes.c_double*self._datalen
        #array_pointer = ctypes.cast(res, ctypes.POINTER(ArrayType))
        #cleaned = self._data.copy()
        #cleaned[:] = np.frombuffer(array_pointer.contents, dtype=np.double)
        #return cleaned
	return self._lib.bsm_cleanData(self._obj)

    def bsm_smoothedData(self):
        return self._lib.bsm_smoothedData(self._obj)
        

def bspline_clean(data, smoothness, zscore, smoother=None,
                           method=RegressionCleaner.replace_with_bound):
    """Clean a data series using B-spline smoothing."""
    if smoother is None:
        smoother = BSplineSmoother(data, smoothness)
    else:
        smoother.set_dataset(data)
        smoother.set_smoothness(smoothness)
    cleaner = RegressionCleaner(smoother, zscore)
    (cleaned, outliers) = cleaner.get_cleaned_data(method)
    return cleaned
        
def bspline_clean_fast(data, smoothness, zscore):
    """Clean a data series using B-spline smoothing."""
    
    #define the degree
    degree = 3
    
    #create knot vector
    knots = sp.get_uniform_knots_from_points(
		    data, degree, knotrange=(0, len(data) - 1))
	
    #determine datasize
    n_data = len(data)
    n_knot = len(knots)
    
    #create a pointer to the dataset
    ds = np.array(data)
    ds_type = c_double*n_data
    ds_ptr = ds_type(*ds)
    
    #create a pointer to the knot
    kn = np.array(knots)
    kn_type = c_double*n_knot
    kn_ptr = kn_type(*kn)
    
    smoother = BsplineFastSmoother(ds_ptr, n_data, kn_ptr, n_knot, degree, c_double(smoothness), c_double(zscore))
    smoothed = smoother.bsm_cleanData()
    
    #convert the pointer to nparray
    ArrayType = ctypes.c_double*n_data
    array_ptr = ctypes.cast(smoothed, ctypes.POINTER(ArrayType))
    cleaned = data.copy()
    cleaned[:] = np.frombuffer(array_ptr.contents, dtype=np.double)
    
    return cleaned

def clean_entire_dataset(dataset, smoothness):
    """Clean an entire dataset (an instance of class Dataset, where the initial
    time series is partitioned into a number of periods), with default knots,
    for a given smoothness factor. Return a list of the cleaned points for each
    period in the dataset."""
    smoother = None
    outliers = []
    for period_number in range(dataset.num_periods):
        period = dataset.get_period(period_number)
        if smoother is None:
            smoother = BSplineSmoother(period, smoothness=smoothness)
        else:
            smoother.set_dataset(period)
        cleaner = RegressionCleaner(smoother)
        cur_out = cleaner.get_outliers()
        if cur_out is not None:
            outliers.append((period_number, cur_out))
    return outliers

def _show_smoother():
    import sg.data.bchydro as bchydro
    import matplotlib.pyplot as plt
    all_timeseries = bchydro.load()
    week = all_timeseries.data[0:24*7]
    sm = BSplineSmoother(week, smoothness=0)
    t = np.linspace(0, 1, 1000)
    y = sm.splev(t)
    plt.figure()
    plt.plot(t, y)
    plt.title("Show smoother")

if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
