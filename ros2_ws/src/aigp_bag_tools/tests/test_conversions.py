import math
import unittest

import numpy as np

from aigp_bag_tools.conversions import (
    enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    q_enu_flu_to_q_ned_frd,
    q_ned_frd_to_q_enu_flu,
)


class ConversionTest(unittest.TestCase):
    def test_ned_enu_vector_round_trip(self):
        v_ned = np.asarray([1.0, 2.0, -3.0])
        self.assertTrue(np.allclose(enu_to_ned(ned_to_enu(v_ned)), v_ned))

    def test_frd_flu_vector_round_trip(self):
        v_frd = np.asarray([1.0, -2.0, 3.0])
        self.assertTrue(np.allclose(flu_to_frd(frd_to_flu(v_frd)), v_frd))

    def test_quaternion_round_trip(self):
        half = math.sqrt(0.5)
        q_ned_frd = np.asarray([half, 0.0, 0.0, half])
        q_round_trip = q_enu_flu_to_q_ned_frd(q_ned_frd_to_q_enu_flu(q_ned_frd))
        self.assertTrue(np.allclose(q_round_trip, q_ned_frd) or np.allclose(q_round_trip, -q_ned_frd))


if __name__ == "__main__":
    unittest.main()
