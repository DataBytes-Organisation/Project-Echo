import init
import unittest
from entities.clock import Clock


class Simulator():
    def __init__(self) -> None:
        config = init.Config()

        class TestConfig(init.TestConfig):
            def __init__(self, *args, config=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.config = config

        suite = unittest.TestSuite()
        for test_name in unittest.defaultTestLoader.getTestCaseNames(TestConfig):
            suite.addTest(TestConfig(test_name, config=config))

        unittest_runner = unittest.TextTestRunner()
        unittest_runner.run(suite)

        self.SystemClock = Clock()

if __name__ == "__main__":
    Simulator()