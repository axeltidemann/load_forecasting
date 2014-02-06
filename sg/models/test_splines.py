import unittest

from splines import *
import scipy.interpolate as si

class TestData():
    def __init__(self):
        self.points = [(1, 2),
                       (2, 4),
                       (4, 1.5),
                       (7, 2),
                       (6, 3.5)]
        self.t = np.linspace(0, 1, 100)
        self.bzknots = self._create_bezier_knots()
        self.multknots = self._create_multiplicity_knots()
        
    def _create_bezier_knots(self):
        """Create a set of knots that will make the B-spline produce a Bezier
        given the current set of points (no internal knots)."""
        n = len(self.points) - 1
        degree = n
        m = n + degree + 1
        knots = np.linspace(0, 1, m - (2*degree - 1))
        knots = np.insert(knots, np.zeros(degree), 0)
        knots = np.append(knots, np.ones(degree))
        return knots

    def _create_multiplicity_knots(self):
        """Create a list of knots with varying multiplicity."""
        return [0, 0, 0, 0,
                0.12, 0.25, 0.37,
                0.5, 0.5,
                0.62, 0.75, 0.87,
                1, 1, 1, 1]
    
    def create_bezier_data(self):
        C = bezierfunc(self.points)
        x, y = C(self.t)
        return x, y

    def create_bspline_data(self):
        c = bsplinefunc(self.bzknots, self.points)
        xy = map(c, self.t)
        x, y = zip(*xy)
        return x, y

    def get_fitpack_tck(self):
        t, c = (self.bzknots, zip(*self.points))
        k = len(t) - len(c[0]) - 1
        return (t, c, k)
    

class TestSplines(unittest.TestCase):
    def setUp(self):
        self.data = TestData()
        
    def test_binomial(self):
        answers = [1, 1, 1, 2, 1,
                   1, 3, 3, 1, 1,
                   4, 6, 4, 1, 1,
                   5, 10, 10, 5, 1]
        for n in [1, 2, 3, 4, 5]:
            for k in range(n+1):
                self.assertEqual(answers.pop(0), binomialcoefficient(n, k))
        
    def test_bernstein(self):
        """Test the Bernstein basis polynomials. The list of test
        lambdas is taken from Wolfram Mathworld."""
        bases = [[lambda t: (t - t) + 1], # Make '1' callable
                 [lambda t: 1 - t,
                  lambda t: t],
                 [lambda t: (1 - t)**2,
                  lambda t: 2 * (1 - t)*t,
                  lambda t: t**2],
                 [lambda t: (1 - t)**3,
                  lambda t: 3 * (1 - t)**2 * t,
                  lambda t: 3 * (1 - t) * t**2,
                  lambda t: t**3]]
        for n in range(4):
            for i in range(n+1):
                bf = bernsteinbasis(i, n)
                t = np.arange(0, 1, 0.25)
                y_func = bf(t)
                y_test = bases[n][i](t)
                for y_f, y_t in zip(y_func, y_test):
                    self.assertAlmostEqual(y_f, y_t)

    def test_calc_multiplicity(self):
        knots = self.data.multknots
        ansmult = [4, 1, 1, 1, 2, 1, 1, 1, 4]
        ansuniq = [0, 0.12, 0.25, 0.37, 0.5, 0.62, 0.75, 0.87, 1]
        uniq, mult = calc_multiplicity(knots)
        self.assertEqual(mult, ansmult)
        self.assertEqual(uniq, ansuniq)

    def test_lengthsanddegrees(self):
        knots = np.linspace(0, 1, len(self.data.points) + 8)
        m, n, p, dim = get_lengths_and_degree(knots, self.data.points)
        self.assertTrue(m == n + p + 1)
        points = [(1, 1, 1)]
        m, n, p, dim = get_lengths_and_degree(knots, points)
        self.assertTrue(m == n + p + 1)
        self.assertEqual(dim, 3)

    def test_get_uniform_knots(self):
        knots = get_uniform_knots_from_points(self.data.points, 4)
        ans = self.data.bzknots
        self.assertEqual(list(knots), list(ans))
        knots = get_uniform_knots_from_points(self.data.points, 4,
                                              knotrange=(-8, 8))
        ans = self.data.bzknots * 16 - 8
        self.assertEqual(list(knots), list(ans))
        knots = get_uniform_knots_from_points(self.data.points, 3)
        ans = [0, 0, 0, 1./4, 1./2, 3./4, 1, 1, 1]
        knots = get_uniform_knots_from_points(self.data.points, 2)
        ans = [0, 0, 0, 1./3, 2./3, 1, 1, 1]
        self.assertEqual(list(knots), ans)
        knots = get_uniform_knots(10, 3)
        ans = [0, 0, 0, 0, 1./3, 2./3, 1, 1, 1, 1]
        self.assertEqual(list(knots), ans)
        knots = get_uniform_knots(6, 2)
        ans = [0, 0, 0, 1, 1, 1]
        self.assertEqual(list(knots), ans)
        knots = get_uniform_knots(7, 2)
        ans = [0, 0, 0, 0.5, 1, 1, 1]
        self.assertEqual(list(knots), ans)

    def test_bases_ranges(self):
        p = 3
        knots = get_uniform_knots_from_points(self.data.points, p)
        n = len(knots) - 1
        imax = n - p
        for i in range(-10, 0):
            self.assertRaises(ValueError, bsplinebasis, knots, i, p)
        for i in range(imax, n*2):
            self.assertRaises(ValueError, bsplinebasis, knots, i, p)
        for i in range(imax):
            basis = bsplinebasis(knots, i, p)
            knot_i = knots[i]
            knot_ip = knots[min(i + p, n)]
            t = np.linspace(-100, knot_i, 100)
            x = map(basis, t)
            self.assertEqual(x[:-1], list(np.zeros(len(x)-1)))
            t = np.linspace(knot_i, knot_ip, 100)
            x = map(basis, t)
            self.assertLessEqual(max(x), 1)
            self.assertGreaterEqual(min(x), 0)
            t = np.linspace(knot_ip, 100, 100)
            x = map(basis, t)
            self.assertEqual(x[1:], list(np.zeros(len(x) - 1)))
            
    def test_bezier_vs_bsplines(self):
        """Test that bezier and bspline returns the same result when the
        b-spline has no internal knots."""
        xbz, ybz = self.data.create_bezier_data()
        xbs, ybs = self.data.create_bspline_data()
        self.assertEqual(len(xbz), len(ybz))
        self.assertEqual(len(xbs), len(ybs))
        self.assertEqual(len(xbz), len(xbs))
        self.assertEqual(xbz[0], xbs[0], self.data.points[0][0])
        self.assertEqual(ybz[0], ybs[0], self.data.points[0][1])
        self.assertEqual(xbz[-1], self.data.points[-1][0])
        self.assertEqual(ybz[-1], self.data.points[-1][1])
        for i in range(len(xbs)):
            self.assertAlmostEqual(xbz[i], xbs[i])
            self.assertAlmostEqual(ybz[i], ybs[i])

    def test_bsplines_vs_fitpack(self):
        """Test that the homemeade bsplines return the same result as
        fitpack/dierckx/scipy.interpolate."""
        tck = self.data.get_fitpack_tck()
        xsi, ysi = si.splev(self.data.t, tck)
        xbs, ybs = self.data.create_bspline_data()
        self.assertEqual(len(xsi), len(ysi))
        self.assertEqual(len(xbs), len(ybs))
        self.assertEqual(len(xsi), len(xbs))
        self.assertEqual(xsi[0], self.data.points[0][0])
        self.assertEqual(ysi[0], self.data.points[0][1])
        self.assertEqual(xsi[-1], self.data.points[-1][0])
        self.assertEqual(ysi[-1], self.data.points[-1][1])
        for i in range(len(xbs)):
            self.assertAlmostEqual(xsi[i], xbs[i])
            self.assertAlmostEqual(ysi[i], ybs[i])

    def test_bspline_altrange_vs_fitpack(self):
        """Test that the homemade bsplines return the same result as
        fitpack/dierckx/scipy.interpolate irrelevant of the range of t."""
        tck = self.data.get_fitpack_tck()
        xsi, ysi = si.splev(self.data.t, tck)
        scale = 17
        alt_knots = self.data.bzknots * scale
        alt_t = self.data.t * scale
        spf = bsplinefunc(alt_knots, self.data.points)
        xbs, ybs = zip(*map(spf, alt_t))
        self.assertEqual(len(xsi), len(ysi))
        self.assertEqual(len(xbs), len(ybs))
        self.assertEqual(len(xsi), len(xbs))
        self.assertEqual(xsi[0], self.data.points[0][0])
        self.assertEqual(ysi[0], self.data.points[0][1])
        self.assertEqual(xsi[-1], self.data.points[-1][0])
        self.assertEqual(ysi[-1], self.data.points[-1][1])
        for i in range(len(xbs)):
            self.assertAlmostEqual(xsi[i], xbs[i])
            self.assertAlmostEqual(ysi[i], ybs[i])

    def test_bsplinederiv_vs_fitpack(self):
        """Test bspline (and necessarily also basis) derivative estimation
        against fitpack/scipy/dierckx."""
        tck = self.data.get_fitpack_tck()
        xsi, ysi = si.splev(self.data.t, tck, der=1)
        deriv = bsplinederivfunc(self.data.bzknots, self.data.points, 1)
        xbs, ybs = zip(*map(deriv, self.data.t))
        self.assertEqual(len(xsi), len(ysi))
        self.assertEqual(len(xbs), len(ybs))
        self.assertEqual(len(xsi), len(xbs))
        for i in range(len(xbs)):
            self.assertAlmostEqual(xsi[i], xbs[i])
            self.assertAlmostEqual(ysi[i], ybs[i])
        
    def test_bsplinederiv_altrange_vs_fitpack(self):
        """Test bspline (and necessarily also basis) derivative estimation
        against fitpack/scipy/dierckx. for alternative range of t. The
        derivatives of the individual basis functions vary with the range of
        t. The slope of the first derivative scales linearly with the change of
        distance between the knots. The derivative of the spline when expressed
        as y = f(x) should be unaffected."""
        tck = self.data.get_fitpack_tck()
        xsi, ysi = si.splev(self.data.t, tck, der=1)
        scale = 17
        alt_knots = self.data.bzknots * scale
        alt_t = self.data.t * scale
        deriv = bsplinederivfunc(alt_knots, self.data.points, 1)
        xbs, ybs = zip(*map(deriv, alt_t))
        for i in range(len(xbs)):
            self.assertAlmostEqual(xsi[i], xbs[i] * scale)
            self.assertAlmostEqual(ysi[i], ybs[i] * scale)

# @unittest.skip("Skip for automated testing, enable every " \
#                "once in a while for manual inspection.")
class BSplineVisualTester(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.show()

    def test_bases_and_first_deriv(self):
        knot_idx = [0, 3, 6]
        degree = [4, 3, 2]
        t = [0.2, 0.6, 0.85]
        plt.figure()
        plt.hold(True)
        plot_bases(knot_idx, degree)
        plot_basis_derivatives(knot_idx, degree, t)
        plt.title("Basis functions of degree %d, %d and %d " \
                  "with derivatives at t=%.2f, %.2f and %.2f." % \
                  (degree[0], degree[1], degree[2],
                   t[0], t[1], t[2]))
        plt.ylim(0, 1)

    def test_basis_second_deriv(self):
        knot_idx = [0, 3, 6]
        degree = [4, 3, 2]
        t = [0.17, 0.6, 0.85]
        knots = np.linspace(0, 1, max(np.add(knot_idx, degree)) + 2)
        x = np.linspace(0, 1, 1000)
        plt.figure()
        for i in range(len(knot_idx)):
            n = knot_idx[i]
            p = degree[i]
            t_ = t[i]
            deriv = bsplinebasis_deriv(knots, n, p, 1)
            y = map(deriv, x)
            plt.plot(x, y, label='1st deriv degree %d' % p, hold=True)
            slope = bsplinebasis_deriv(knots, n, p, 2)(t_)
            offset = deriv(t_)
            y = map(lambda a: (a - t_) * slope + offset, x)
            plt.plot(x, y, label="2nd deriv a t=%.2f" % t_, hold=True)
        plt.ylim(-10, 10)
        plt.title("1st deriv of basis funcs degree %d, %d, %d " \
                  "with 2nd derivs at t=%.2f, %.2f, %.2f." % \
                  (degree[0], degree[1], degree[2],
                   t[0], t[1], t[2]))
        plt.legend(loc=3)

    def test_bspline_deriv(self):
        data = TestData()
        plt.figure()
        plt.hold(True)
        x, y = data.create_bspline_data()
        plt.plot(x, y, label="B-spline/bezier curve")
        deriv = bsplinederivfunc(data.bzknots, data.points, 1)
        x, y = zip(*map(deriv, data.t))
        plt.plot(x, y, label="1st derivative of B-spline")
        for dt in [0.17, 0.45, 0.85]:
            xslope, yslope = deriv(dt)
            xoff, yoff = bsplinefunc(data.bzknots, data.points)(dt)
            x = map(lambda a: (a - dt) * xslope + xoff, data.t)
            y = map(lambda b: (b - dt) * yslope + yoff, data.t)
            plt.plot(x, y, label='1st deriv at t=%.2f' % dt)
        plt.ylim(-2, 4)
        plt.xlim(0.5, 8)
        plt.title("B-spline with first derivative")
        plt.legend(loc=3)
        
if __name__ == "__main__":
    unittest.main()
