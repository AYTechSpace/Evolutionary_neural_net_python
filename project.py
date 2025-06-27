from random import random, randint
from math import e
import numpy as np  

class perceptron_network():
    def __init__(self, bits: int = 4, population_size: int = 100):
        self.input_bits: int = bits
        self.output_bits: int = bits + 1
        self.pop_size: int = population_size

def main():
    
    # Set network parameter

    bits: int = 2
    population_size: int = 5
    generations: int = 2
    test_size = 2

    # Create network
    evo = perceptron_network(bits, population_size)
    evo.generate(generations, training_data(bits, test_size))

    print(evo.test())

def training_data(bits, test_size):

    a = np.random.randint(0, 2, (test_size, bits))
    b = np.random.randint(0, 2, (test_size, bits))
    
    c = cleaner(a) + cleaner(b)

    test_data = np.array([a, b, c])

    return test_data

def cleaner(x):
    ...

def sigmoid(x: int):
    return 1/(1 + 1/e**x)
    
if __name__ == "__main__":
    main()
