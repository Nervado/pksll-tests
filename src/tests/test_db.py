import unittest

from pksll_tests.db.repository import Repository


class InfluxDbAdapter(Repository):
    ...


class TestDataBase(unittest.TestCase):
    def test_should_connect_to_data_base(self):
        sut = InfluxDbAdapter()
        sut.connect()
        self.assertEqual(sut.isconnected, True)
