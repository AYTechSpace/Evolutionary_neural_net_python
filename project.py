# assert self.networks[net_id].id == net_id
import numpy as np
from math import e
import mypy
import copy

class Population():
    def __init__(self, bits_per_operand: int, signed: bool,
                 population_size: int, layer_sizes: list, elites_chance: float):
        
        self.networks: list = []
        self.population_size: int = population_size                         # Pop = 100

        self.elite_size: int = int(elites_chance * population_size)         # Elite = 10
        self.dregs_size: int = population_size - self.elite_size # Sorry!   # Dreg = 90
        self.species_size: int = self.dregs_size // self.elite_size         # Species = 9
        self.remainder_size: int =  self.dregs_size % self.species_size     # Remainder = 0

        self.layer_sizes: list = layer_sizes
        self.best_fitnesses: list = []
        
        for net_id in range(population_size):
            current_network = Network(bits_per_operand, signed, layer_sizes, net_id)
            self.networks.append(current_network)

    def evolve(self, generations: int, test_data: list, print_results: bool = True):

        for gen in range(generations):

            unique_weights = set()
            for net in self.networks:
                unique_weights.add(str(net.layers[1].node_weights))
            print(f"Unique genomes: {len(unique_weights)}")


            fitnesses = [] 

            for network in self.networks:
                fitnesses.append(network.test_fitness(test_data)) # test_fitness should return a tuple (str, id)

            fitnesses.sort(reverse = True)
            sorted_ids = list(fitness_tuple[1] for fitness_tuple in fitnesses) 

            elite_ids = sorted_ids[0:self.elite_size] # [:-self.elite_size:-1]
            elites = list(self.networks[net_id] for net_id in elite_ids)

            anchor = self.elite_size

            #print(fitnesses)

            current_elite_rank = 0
            for elite in elites:
                child_min = self.elite_size + current_elite_rank * self.species_size  # (10 + 0 * 9) = 10, (10 + 1 * 9) = 19, (10 + 2 * 9) = 28
                child_max = child_min + self.species_size # (10 + 9) = 19, (19 + 9) = 28, (28 + 9) = 37

                if print_results:
                    print(elite.data(gen, elite.layer_sizes[-1], elites.index(elite), elite.id))
                
                for rank in range(child_min, child_max):
                    net_id = fitnesses[rank][1]
                    self.networks[net_id] = (self.networks[net_id].inherit(elite))

                current_elite_rank +=1
            print("\n----------\n")

            self.best_fitnesses.append(fitnesses[0][0])




class Network():
    def __init__(self, bits_per_operand: int, signed: bool, layer_sizes: list, net_id: int):
        self.bits_per_operand: int = bits_per_operand
        self.signed: bool = signed
        self.layer_sizes: list = layer_sizes
        self.id: int = net_id

        # Performance Stats
        self.bit_frequency: np.ndarray
        self.weighted_bit_frequency: np.ndarray
        self.acc: float = 0
        self.rank: int

        self.fitness: float = 0.0

        self.layers: list = self.create_layers(self.layer_sizes)

    def create_layers(self, layer_sizes):

        num_of_layers = len(layer_sizes)
        layers = []


        for layer_level in range(num_of_layers):
            num_of_nodes = layer_sizes[layer_level]
            if layer_level != 0:
                num_of_inputs = layer_sizes[layer_level - 1]
            else:
                num_of_inputs = 0
                
            layers.append(Layer(num_of_nodes, layer_level, num_of_layers, num_of_inputs))
            # if layer_level = 0, is input layer, no weights
            # if layer_level = num_of_layers, is output_layer, special function

        return layers
    
    def find_layer_output(self, layer, layer_input, activation_function, final_activation_function):

        layer_output = []

        for weights in layer.node_weights:
            dot_product = np.dot(layer_input, np.array(weights))
            layer_output.append((dot_product))
        
        # OPTIMISATION POTENTIAL !!!!!!!
        #layer_output = np.array(layer_output)
        #layer.node_biases = np.array(layer.node_biases)


        layer_output = (np.array(layer_output)) + (np.array(layer.node_biases))
        layer_output = np.array(layer_output)

        if layer.type == "output":
            return vectorised_rounded_sigmoid(layer_output)
            #activation_function = final_activation_function

        return vectorised_sigmoid(layer_output)
        #return activation_function(layer_output)

    def test_fitness(self, test_data: list):
        
        self.fitness = 0
        num_of_output_bits = self.layer_sizes[-1]
        self.weighted_bit_frequency = np.zeros((1, num_of_output_bits))
        self.bit_frequency = np.zeros((1, num_of_output_bits))
        self.acc = 0

        for test_question in test_data:
            current_inputs = np.concatenate((test_question[0], test_question[1]))
            for layer in self.layers:
                
                if layer.type != "input":
                    current_inputs = self.find_layer_output(layer, current_inputs, vectorised_sigmoid, vectorised_rounded_sigmoid)

            output_guess = current_inputs

            matches = output_guess == test_question[2]
            if (output_guess == test_question[2]).all(): # OPTIMISATION AVAILABL !!!!!
                self.acc += 1
            self.bit_frequency += matches
            self.bit_percentage = self.bit_frequency / len(test_data)

        self.acc = self.acc / len(test_data)
        self.weighted_bit_frequency = (100 * self.bit_percentage/(2* num_of_output_bits))
        

        self.fitness = np.sum(self.weighted_bit_frequency) + (50 * self.acc)

        return (round(float(self.fitness), 3), self.id)

    def inherit(self, other):

        child_net = copy.deepcopy(self)
        print("A", child_net.layers[1].node_weights)
        
        for layer_level, layer in enumerate(other.layers):
            if layer.type == "input":
                break
            else:
                num_of_nodes = layer.num_of_nodes
                num_of_weights = len(layer.node_weights[0])

                for n in range(num_of_nodes):
                    for w in range(num_of_weights):
                        layer.node_weights[n][w] = mutate(
                                                    child_net.layers[layer_level].node_weights[n][w],
                                                    1,
                                                    10)
                    layer.node_biases[n] = mutate(
                                            child_net.layers[layer_level].node_biases[n],
                                            1,
                                            10)

        print("B", child_net.layers[1].node_weights)
        return child_net

    def data(self, gen, output_layer_size, rank, id):
        msg = f"Gen: {gen} | Rank: {rank:03} | ID: {id:03} | Fitness: {round(self.fitness, 4): 04} | Accuracy: {round(100 * self.acc, 2):04}"
        for i in range(output_layer_size ):
            msg += f" | bit {i}: {round(100 * self.bit_percentage[0][i], 4):04}"

        #if (rank == 0) or (rank == 1):
        #    print(self.layers[2].node_biases)

        return msg

class Layer():
    def __init__(self, num_of_nodes, layer_level, num_of_layers, num_of_inputs):

        if layer_level == 0:
            self.type = "input"
        elif layer_level == num_of_layers - 1:
            self.type = "output"
        else:
            self.type = "hidden"

        self.num_of_nodes = num_of_nodes

        self.node_weights = []
        self.node_biases = []

        if self.type != "input":
            for i in range(num_of_nodes):
                self.node_weights.append(np.random.uniform(-1, 1, num_of_inputs))
                self.node_biases.append(np.random.uniform(-1, 1))

def main():
    
    bits_per_operand: int = 3
    signed: bool = False

    input_bit_size = 2 * bits_per_operand
    output_bit_size = bits_per_operand + 1

    population_size: int = 10
    elites_chance: float = 0.1
    mutation_chance: float = 0.1
    
    h1_size: int = 128
    h2_size: int = 2
    h3_size: int = 2
    h4_size: int = 16

    layer_sizes: list = [input_bit_size, h2_size, h3_size, output_bit_size]

    num_of_tests: int = 50

    test_data_a = generate_test_data(bits_per_operand, num_of_tests, signed)
    #test_data_b = generate_test_data(bits_per_operand, num_of_tests, signed)
    #test_data_c = generate_test_data(bits_per_operand, num_of_tests, signed)
    #test_data_d = generate_test_data(bits_per_operand, num_of_tests, signed)
    #test_data_e = generate_test_data(bits_per_operand, 10, signed)

    population_a = Population(bits_per_operand,
                              signed,
                              population_size,
                              layer_sizes,
                              elites_chance,
                              )

    print("Start")
    population_a.evolve(generations = 4, test_data = test_data_a, print_results = True)


   # print("Test")
   # print(test_data_b[0][0], test_data_b[0][1], test_data_b[0][2])
   # population_a.evolve(generations = 1, test_data = test_data_e, print_results = True)

    #population_a.test_best()

def generate_test_data(bits_per_operand: int, num_of_tests: int, signed: bool):
    
    test_data = []
    
    output_bits = bits_per_operand + 1 

    max = 2 ** (bits_per_operand - signed) - 1
    min = signed * (-max)

    for _ in range(num_of_tests):
        operand_a = np.random.randint(min, max + 1)
        operand_b = np.random.randint(min, max + 1)
        result = operand_a + operand_b

        test_data.append([bin_format(operand_a, bits_per_operand, signed),
                          bin_format(operand_b, bits_per_operand, signed),
                          bin_format(result, output_bits, signed)])

    return test_data

def bin_format(integer: int, bits, signed: bool):

    binary = bin(integer).replace("0b", "")

    if signed:
        if binary[0] == "-":
            binary = binary.replace("-", "1")
        else:
            binary = "0" + binary


    
    missing_0s = bits - len(binary)
    binary = signed*binary[0]  + ("0" * missing_0s) + (not signed)*(binary[0]) + binary[1:]

    bin_form  = np.array(list(int(digit) for digit in binary))

    return bin_form

def sigmoid(x):
    return 1/ (1 + (e ** -x))

def vectorised_sigmoid(x):
    return np.vectorize(sigmoid)(x)

def rounded_sigmoid(x):
    return round((1/ (1 + (e ** -x))))

def vectorised_rounded_sigmoid(x):
    return np.vectorize(rounded_sigmoid)(x)

def mutate(x, chance, strength):
    if np.random.uniform(0, 1) <= chance:
        delta = np.random.uniform(-strength, strength)
    else:
        delta = 0

    return (x * (1 + delta))
    
if __name__ == "__main__":
    main()



