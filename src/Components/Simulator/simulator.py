import init
import unittest
from entities.clock import Clock
from entities.animal_factory import AnimalFactory

class TestConfig(init.TestConfig):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
                
class Simulator():
    def __init__(self) -> None:
        self.config = init.Config()
        
    # run the live simulator
    def execute(self):
        pass

    # run some simulator test cases
    def test(self):
        suite = unittest.TestSuite()
        for test_name in unittest.defaultTestLoader.getTestCaseNames(TestConfig):
            suite.addTest(TestConfig(test_name, config=self.config))

        unittest_runner = unittest.TextTestRunner()
        unittest_runner.run(suite)

        self.SystemClock = Clock()
        self.AnimalFactory = AnimalFactory()
        print(self.AnimalFactory.create_random_animal().species)

if __name__ == "__main__":
    sim = Simulator()
    sim.test()
    