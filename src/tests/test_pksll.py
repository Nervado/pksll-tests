import unittest


from plc_tester.plc.pksll import PKSLL

from plc_tester.models.signal import Signal


class TestPKSLL(unittest.TestCase):

    def test_should_calculte_corretly_filtered_signal(self):

        sut = PKSLL()

        result = sut.execute(Signal(inputData=[
                             390, 350, 340, 290, 290, 290, 240, 240]))

        self.assertEqual(result.alarms, 1)
