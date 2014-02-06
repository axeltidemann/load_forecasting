import unittest
import tempfile
import os
import cPickle as pickle

import numpy as np
import splines as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d

from spclean import *
import sg.data.bchydro as bchydro
import sg.utils.testutils as testutils

bchydro_timeseries = None

def setUpModule():
    np.seterr(all='raise')
    global bchydro_timeseries
    bchydro_timeseries = bchydro.load()

# @unittest.skip("Debug skip.")
class TestSmoothInit(unittest.TestCase):
    def setUp(self):
        self.dataset = [0, 1, 2, 4, 10]
        self.knots = [0, 0, 0, 0, 2, 4, 4, 4, 4]
        self.sm = self._make_smoother()

    def _make_smoother(self):
        return BSplineSmoother(self.dataset, 0)

    def test_create(self):
        self.assertEqual(list(self.sm.knots), self.knots)

    def test_set_knots(self):
        self.sm.set_knots()
        self.assertEqual(list(self.sm.knots), self.knots)
        self.assertRaises(RuntimeError,
                          self.sm.set_knots, [0, 1, 2], 25)


class TestSmootherFixtures(testutils.ArrayTestCase):
    def setUp(self):
        self._day = bchydro_timeseries[0:24].copy()
        self._week = bchydro_timeseries[0:24*7].copy()
        self.data = self._day
        self.sm = self._make_smoother()
        n = len(self.sm.dataset)
        self.t_data = np.linspace(self.sm.knots[0], self.sm.knots[-1], n)

    def _make_smoother(self):
        return BSplineSmoother(self.data, smoothness=0)

        
# @unittest.skip("Debug skip.")
class TestSmoother(TestSmootherFixtures):
    def test_spline_interpolation(self):
        """Test interpolation by checking that the value of the
        spline is identical to within numerical precision to the
        corresponding point in the dataset."""
        y = self.sm.splev(self.t_data)
        self.assertArraysAlmostEqual(self.sm.dataset, y)

    def test_direct_interpolation(self):
        """Test that direct calculation of <y> estimates by multiplying the hat
        matrix with y and no smoothing returns (almost) y."""
        self.assertArraysAlmostEqual(self.sm.dataset,
                                     self.sm.get_smoothed_data())

    def _direct_equals_spline(self):
        """Check that the <y> estimates that are obtained by multiplying the
        hat matrix with y are (almost) the same as those returned when
        evaluating the spline."""
        y_spline = self.sm.splev(self.t_data)
        y_smooth = self.sm.get_smoothed_data()
        self.assertArraysAlmostEqual(y_smooth, y_spline)

    def test_direct_equals_spline_interp(self):
        self._direct_equals_spline()

    def test_direct_equals_spline_smooth(self):
        self.sm.set_smoothness(1)
        self._direct_equals_spline()

    def _calculate_steps(self, array):
        """Returns an array with len(array)-1 elements, where each element is
        the distance from the corresponding element in the input array to the
        next."""
        # Array subtraction does not work with Pandas time series in this case,
        # as slicing will retain the index, and the subtraction will hence
        # align on dates (exactly the opposite of what we're trying to achieve.
        #return abs(array[0:-1] - array[1:])
        return np.array([abs(array[i] - array[i+1]) for i in range(len(array)-1)])

    def _check_smooth_varies_less(self, rough, smooth):
        """Check that the step size in the rough array is larger than in the
        smooth array."""
        rough_steps = self._calculate_steps(rough)
        smooth_steps = self._calculate_steps(smooth)
        self.assertGreater(rough_steps.sum(), smooth_steps.sum())
    
    def test_smoothing_smoothes(self):
        """Test that the line is smoother with increasing smoothness
        parameter."""
        previous = self.sm.dataset
        for smoothness in (0.001, 0.1, 1, 2):
            self.sm.set_smoothness(smoothness)
            y = self.sm.splev(self.t_data)
            self._check_smooth_varies_less(previous, y)
            previous = y

    def test_set_dataset(self):
        """Make sure interpolation works after setting new data."""
        self.sm.splev(self.t_data)
        self.sm.set_dataset(bchydro_timeseries[24:24*2].copy())
        y = self.sm.splev(self.t_data)
        self.assertArraysAlmostEqual(self.sm.dataset, y)

class TestChenSmootherInternals(TestSmootherFixtures):
    """Test internal stuff specific to the Chen et al.-inspired
    implementation(s)."""
    def test_calc_phi(self):
        """Performs some rudimentary tests, asserting that phi is
        sane."""
        phi = self.sm._get_phi()
        self.assertLessEqual(phi.max(), 1)
        self.assertGreaterEqual(phi.min(), 0)
        self.assertEqual(np.count_nonzero(phi),
                         (len(self.sm.dataset) - 2) * \
                         (self.sm._degree + 1) + 2)
        bf = sp.bsplinebasis(self.sm.knots, 0, self.sm._degree)
        self.assertEqual(phi[0, 0], 1)
        self.assertEqual(phi[-1, 0], 0)
        self.assertEqual(phi[self.sm._degree+2, 0], 0)
        basis = sp.bsplinebasis(self.sm.knots, 6, self.sm._degree)
        values = []
        for t_idx in [5, 6, 7, 8, 9]:
            knotrange = self.sm.knots[-1] - self.sm.knots[0]
            t = float(t_idx) / (len(self.sm.dataset) - 1) * knotrange
            self.assertEqual(phi[t_idx, 6], basis(t))
        self.assertTrue(np.any(phi[5:10, 6] > 0))

    def test_roughness_numeric_resembles_analytic(self):
        """Make sure that analytic and numeric solutions are similar for not
        too long data series."""
        smn = BSplineBasicSmoother(self.data, smoothness=0)
        sma = BSplineAnalyticSmoother(self.data, smoothness=0)
        rn = smn._get_roughness()
        ra = sma._get_roughness()
        self.assertArraysAlmostEqual(rn, ra)


class TestCachingSmoother(TestSmootherFixtures):
    def test_cached_equals_calculated(self):
        """Test that cached and calculated splines are identical."""
        calced = BSplineAnalyticSmoother(self.data, smoothness=0.5)
        cached = BSplineSmoother(self.data, smoothness=0.5)
        knots = calced._knots
        t = np.linspace(knots[0], knots[-1], len(knots) * 10)
        self.assertArraysEqual(calced.splev(t), cached.splev(t))
        

# class TestFitpackSmoother(TestSmoother):
#     def _make_smoother(self):
#         return FitpackSplineSmoother(self.data, smoothness=0)
        
        
# @unittest.skip("Debug skip.")
class TestMatrixCache(testutils.ArrayTestCase):
    """Tests the utility that caches R and Phi."""
    def setUp(self):
        length = 6
        self.knots = np.linspace(0, 10, length)
        self.matrix = np.random.random((length, length))
        self.cache = MatrixCache()
        
    def test_newly_added_key_found(self):
        self.cache[self.knots] = self.matrix
        mtx = self.cache[self.knots]
        self.assertArraysEqual(mtx, self.matrix)

    def test_empty_create_ok(self):
        self.assertFalse(self.cache.has_key(self.knots))
        self.assertEqual(len(self.cache), 0)

    def test_length_is_one(self):
        self.cache[self.knots] = self.matrix
        self.assertEqual(len(self.cache), 1)

    def test_changed_key_found(self):
        self.cache[self.knots] = self.matrix
        mtx = self.matrix * 2
        self.cache[self.knots] = mtx
        mtx2 = self.cache[self.knots]
        self.assertArraysEqual(mtx, mtx2)

    def test_contains(self):
        self.cache[self.knots] = self.matrix
        self.assertIn(self.knots, self.cache)


@unittest.skip("Debug skip.")
class TestMatrixCachePersistent(testutils.ArrayTestCase):
    """Tests persistent storage for the utility that caches R and Phi."""
    def setUp(self):
        handle, self.path = tempfile.mkstemp(
            prefix="cachetest_", suffix='.db', dir=".")
        os.close(handle)
        length = 25
        self.knots = np.linspace(0, 10, length)
        self.matrix = np.random.random((length, length))

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        
    def test_open_nonexisting_file(self):
        self.assertRaises(IOError, MatrixCache,
                          "This file is unlikely to exist.")

    def test_open_empty_file(self):
        self.assertRaises(EOFError, MatrixCache, self.path)

    def test_open_wrong_file(self):
        dummy = [1, 2, 3]
        with open(self.path, "w") as f:
            pickle.dump(dummy, f)
        self.assertRaises(ValueError, MatrixCache, self.path)

    def test_store_restore(self):
        new_cache = MatrixCache()
        new_cache[self.knots] = self.matrix
        new_cache[self.knots * 2] = self.matrix * 2
        new_cache.store(self.path)
        cache = MatrixCache(self.path)
        self.assertEqual(new_cache, cache)

@unittest.skip("Debug skip.")
class TestBSplineCleaner(testutils.ArrayTestCase):
    def setUp(self):
        self.data = bchydro_timeseries[0:24].copy()

    def test_get_all_data_no_smoothing(self):
        sm = BSplineSmoother(self.data, smoothness=0)
        cln = RegressionCleaner(sm)
        (cleaned_data, outliers) = cln.get_cleaned_data(
            RegressionCleaner.replace_with_estimate)
        self.assertArraysAlmostEqual(sm.dataset, cleaned_data)

    def test_outliers_removed(self):
        """Make sure an extreme outlier is removed even with a small amount of
        smoothing."""
        outlier = self.data.max() * 10
        pos = 12
        self.data[pos] = outlier
        sm = BSplineSmoother(self.data, smoothness=1)
        cln = RegressionCleaner(sm)
        (cleaned_data, outliers) = cln.get_cleaned_data(
            RegressionCleaner.replace_with_estimate)
        self.assertIn(pos, outliers)
        self.assertNotAlmostEqual(cleaned_data[pos], self.data[pos],
                                  delta=outlier/2)
        cleaned_data[pos] = self.data[pos]
        self.assertArraysAlmostEqual(sm.dataset, cleaned_data)


@unittest.skip("Skip for automated testing, enable every " \
               "once in a while for manual inspection.")
class VisualTesting(unittest.TestCase):
    def setUp(self):
        self.num_days = 7
        self.days = bchydro_timeseries[0:24*self.num_days].copy()

    @classmethod
    def tearDownClass(self):
        plt.show()

    @unittest.skip("")
    def test_calc_phi(self):
        sm = BSplineSmoother(self.days, smoothness=0)
        phi = sm._get_phi()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d',
            title="%d-day phi: value of basis func y at time x" %\
            self.num_days)
        x = np.arange(0, 20) #len(phi))
        y = np.arange(0, 20) #len(phi[0]))
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, phi[:20,:20], rstride=1, cstride=1,
                               linewidth=1, antialiased=False)

    # @unittest.skip("")
    def test_plot_numeric_vs_analytic_roughness(self):
        """Plot the difference between the numeric and analytic roughness
        calculation. Takes a while both to calculate and render."""
        smn = BSplineBasicSmoother(self.days, smoothness=0)
        sma = BSplineAnalyticSmoother(self.days, smoothness=0)
        rn = smn._get_roughness()
        ra = sma._get_roughness()
        diff = rn - ra
        print "Max roughness is %f (numeric) vs %f (analytic)." \
          % (rn.max(), ra.max())
        print "Min roughness is %f (numeric) vs %f (analytic)." \
          % (rn.min(), ra.min())
        print "Max absolute difference is %e" % np.max(np.abs(diff))
        print "Avg absolute difference is %e" % np.mean(np.abs(diff))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d',
            title="Difference between numeric and analytic solution.")
        x = np.arange(0, diff.shape[0])
        y = np.arange(0, diff.shape[1])
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, diff[:len(x),:len(y)], rstride=1, cstride=1,
                               linewidth=1, antialiased=False)

    # @unittest.skip("")        
    def _plot_splines_and_dataset(self, smoother):
        n = len(smoother._dataset)
        knots = smoother.get_knots()
        t = np.linspace(knots[0], knots[-1], n * 25)
        y = smoother.splev(t)
        plt.hold(True)
        plt.plot(t, y)
        x = np.linspace(knots[0], knots[-1], n)
        plt.plot(x, smoother.dataset, 'x')

    # @unittest.skip("")
    def test_splev_smooth(self, smoothness=1):
        sm = BSplineSmoother(self.days, smoothness=smoothness)
        plt.figure()
        plt.hold(True)
        self._plot_splines_and_dataset(sm)
        plt.title("B-spline with smoothness %f vs data points" % smoothness)

    # @unittest.skip("")
    def test_splev_interpolate(self):
        self.test_splev_smooth(smoothness=0)

    def _plot_cleaned_data(self, smoother, cleaner,
                           clean_data, outliers, title):
        (lower, upper) = cleaner.get_confidence_interval()
        plt.figure()
        plt.hold(True)
        self._plot_splines_and_dataset(smoother)
        plt.plot(lower, 'g-')
        plt.plot(upper, 'g-')
        if len(outliers) > 0:        
            plt.plot(outliers, clean_data[outliers], 'r*', label="Cleaned data")
        plt.title(title)

    # @unittest.skip("")        
    def test_splev_clean(self):
        smooth = 0.5
        sm = BSplineSmoother(self.days, smoothness=smooth)
        cln = RegressionCleaner(sm)
        (clean_data, outliers) = cln.get_cleaned_data(
            RegressionCleaner.replace_with_estimate)
        self._plot_cleaned_data(sm, cln, clean_data, outliers,
                                "Cleaned data, replace with estimate. " \
                                "Smoothness %.3f" % smooth)
        (clean_data, outliers) = cln.get_cleaned_data(
            RegressionCleaner.replace_with_previous)
        self._plot_cleaned_data(sm, cln, clean_data, outliers,
                                "Cleaned data, replace with previous. " \
                                "Smoothness %.3f" % smooth)
        (clean_data, outliers) = cln.get_cleaned_data(
            RegressionCleaner.replace_with_bound)
        self._plot_cleaned_data(sm, cln, clean_data, outliers,
                                "Cleaned data, replace with confidence bound."\
                                " Smoothness %.3f" % smooth)


if __name__ == "__main__":
    unittest.main()
