import unittest
from datetime import timedelta as dt

import sg.utils.testutils as testutils
from sg.data.eunite.eunite import *

class TestEuniteDataset(testutils.ArrayTestCase):
    def setUp(self):
        self.data = Dataset(period=dt(days=2), step_length=dt(days=1))

    def _test_two_days_correct(self, period, temps, loads):
        self.assertEqual(len(period), 2 * 48)
        temps = [temps[0] for i in range(48)] + [temps[1] for i in range(48)]
        fasit = np.array([[t, l] for (t, l) in zip(temps, loads)])
        self.assertArraysEqual(period.data, fasit)

    def test_first_correct(self):
        day_1_to_2 = self.data.get_period(0)
        temps = [-7.6, -6.3]
        loads = [797, 794, 784, 787, 763, 749, 745, 730, 707, 706, 720, 657, 
                 633, 595, 560, 540, 519, 601, 631, 621, 640, 643, 654, 653, 
                 688, 688, 690, 690, 684, 679, 674, 677, 644, 660, 654, 683, 
                 688, 698, 719, 733, 700, 671, 692, 685, 717, 694, 692, 686,
                 704, 697, 704, 676, 664, 668, 668, 662, 665, 666, 703, 677, 
                 669, 660, 650, 672, 648, 682, 692, 724, 727, 739, 739, 733, 
                 741, 754, 767, 768, 738, 734, 747, 733, 751, 746, 737, 750, 
                 759, 776, 777, 777, 746, 724, 697, 708, 745, 705, 702, 722]
        self._test_two_days_correct(day_1_to_2, temps, loads)

    def test_feb17_18_1998_correct(self):
        days = self.data.get_period(365 + 31 + 16)
        temps = [4.1, 1.8]
        loads = [655, 621, 612, 611, 602, 621, 598, 608, 601, 595, 602, 632, 
                 662, 699, 715, 671, 685, 723, 745, 711, 725, 734, 690, 708, 
                 721, 729, 726, 695, 717, 725, 697, 681, 710, 678, 746, 744, 
                 749, 770, 761, 759, 734, 715, 675, 658, 647, 686, 656, 671,
                 702, 698, 672, 659, 665, 655, 630, 637, 633, 672, 674, 715,
                 708, 747, 709, 711, 725, 719, 738, 742, 725, 729, 707, 715, 
                 738, 746, 750, 712, 728, 709, 709, 698, 711, 720, 734, 751, 
                 759, 782, 760, 773, 729, 707, 647, 660, 659, 643, 648, 658]
        self._test_two_days_correct(days, temps, loads)

    def test_last_correct(self):
        last_2_days = self.data.get_period(2 * 365 + 31 - 2)
        temps = [-7.8, -6.0]
        loads = [716, 714, 697, 686, 680, 686, 641, 658, 658, 645, 673, 640, 
                 630, 604, 615, 628, 634, 660, 699, 696, 702, 732, 726, 717, 
                 740, 753, 749, 734, 743, 718, 705, 708, 711, 727, 736, 747, 
                 744, 740, 751, 763, 741, 714, 698, 701, 710, 697, 687, 703,
                 712, 720, 694, 698, 679, 648, 665, 656, 677, 651, 623, 604, 
                 595, 578, 576, 598, 620, 644, 691, 666, 691, 700, 717, 700, 
                 694, 714, 724, 702, 696, 691, 682, 677, 677, 688, 687, 713, 
                 708, 735, 734, 743, 711, 717, 702, 698, 694, 691, 691, 704]
        self._test_two_days_correct(last_2_days, temps, loads)


if __name__ == '__main__':
    unittest.main()
