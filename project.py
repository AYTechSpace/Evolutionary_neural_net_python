# assert self.networks[net_id].id == net_id
import numpy as np
from math import e
import matplotlib.pyplot as plt
import tkinter as tk
import ttkbootstrap as ttk


class Population():
    def __init__(self, bits_per_operand: int, signed: bool,
                 population_size: int, layer_sizes: list, elites_chance: float):
        
        self.networks: list = []
        self.population_size: int = population_size                         # Pop = 100

        self.elite_size: int = int(elites_chance * population_size)         # Elite = 10
        self.dregs_size: int = population_size - self.elite_size # Sorry!   # Dreg = 90
        self.species_size: int = self.dregs_size // self.elite_size         # Species = 9
        #self.remainder_size: int =  self.dregs_size % self.species_size     # Remainder = 0 Redundant line as of now

        self.layer_sizes: list = layer_sizes
        self.best_fitnesses: list = []
        
        for net_id in range(population_size):
            current_network = Network(bits_per_operand, signed, layer_sizes, net_id)
            self.networks.append(current_network)

    def evolve(self, _generations: int, test_data: list, print_results: bool = True):

        #gen_start = 0
        #gen_delta = int(generations/self.population_size)

        for gen in range(_generations):

            unique_weights = set()
            for net in self.networks:
                unique_weights.add(str(net.layers[1].node_weights))
            print(f"Unique genomes: {len(unique_weights)}")


            fitness_data = []
            fitnesses = []
            gens = []

            for network in self.networks:
                fitness_result = (network.test_fitness(test_data)) # test_fitness should return a tuple (str, id)
                fitness_data.append(fitness_result)
                fitnesses.append(fitness_result[0])
                gens.append(gen)


            plt.scatter(gens[:], fitnesses[:])
#            plt.scatter(gens[:self.elite_size], fitnesses[:self.elite_size])
            #gen_start += gen_delta

            fitness_data.sort(reverse = True)
            sorted_ids = list(fitness_tuple[1] for fitness_tuple in fitness_data) 

            elite_ids = sorted_ids[0:self.elite_size] # [:-self.elite_size:-1]
            elites = list(self.networks[net_id] for net_id in elite_ids)


            current_elite_rank = 0
            print("Elite size:", len(elites))
            for elite in elites:
                child_min = self.elite_size + current_elite_rank * self.species_size  # (10 + 0 * 9) = 10, (10 + 1 * 9) = 19, (10 + 2 * 9) = 28
                child_max = child_min + self.species_size # (10 + 9) = 19, (19 + 9) = 28, (28 + 9) = 37

                if print_results:
                    print(elite.data(gen, elite.layer_sizes[-1], elites.index(elite), elite.id))
                
                for rank in range(child_min, child_max):
                    net_id = fitness_data[rank][1]
                    (self.networks[net_id].inherit(elite))

                current_elite_rank += 1
            print("\n----------\n" * print_results)

            self.best_fitnesses.append(fitness_data[0][0])

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
            return vectorised_rounded_sigmoid(layer_output) #final_activation_function(layer_output)

        return vectorised_sigmoid(layer_output) #activation_function(layer_output)

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
            if (output_guess == test_question[2]).all(): # OPTIMISATION AVAILABLE !!!!!
                self.acc += 1
            self.bit_frequency += matches
            self.bit_percentage = self.bit_frequency / len(test_data)

        self.acc = self.acc / len(test_data)
        self.weighted_bit_frequency = (100 * self.bit_percentage/(2* num_of_output_bits))

        self.fitness = np.sum(self.weighted_bit_frequency) + (50 * self.acc)

        return (float(self.fitness), self.id)

    def inherit(self, other):

        layer_amount = len(self.layers)

        for layer in range(layer_amount):
            if layer != 0: #not input layer
                current_layer = self.layers[layer]
                previous_layer = self.layers[layer - 1]

                node_amount = current_layer.num_of_nodes
                input_amount = previous_layer.num_of_nodes

                #print(f"\nStart{current_layer.node_weights}\n")
                for n in range(node_amount):
                    #print(f"\nStart{current_layer.node_weights}\n")
                    for w in range(input_amount):
                        current_layer.node_weights[n][w] = mutate(other.layers[layer].node_weights[n][w],
                                                    0.95, 1)# Change to self.
                        
                        current_layer.node_biases[n] = mutate(other.layers[layer].node_biases[n],
                                                            0.95, 1)

                #print(f"\nEnd{current_layer.node_weights}\n")  
                #self.layers[layer].node_weights = current_layer.node_weights
                #self.layers[layer].node_biases = current_layer.node_biases

    def data(self, gen, output_layer_size, rank, id):
        msg = f"Gen: {gen} | Rank: {rank:03} | ID: {id:03} | Fitness: {round(self.fitness, 4): 04} | Accuracy: {round(100 * self.acc, 2):04}"
        for i in range(output_layer_size):
            msg += f" | bit {i}: {round(100 * self.bit_percentage[0][i], 4):04}"

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

    # Create Main Window and Title
    window = ttk.Window(themename = "darkly")
    window.title("AI Binary Sumator")
    window.geometry("1200x800")

    bits_per_operand = tk.IntVar()
    signed: bool = tk.BooleanVar()

    generations: int = tk.IntVar()
    population_size: int = tk.IntVar()
    elites_chance: float = tk.DoubleVar()
#    mutation_chance: float = tk.DoubleVar() Currently redundant as mutation model doesn't allow for enough variation to let lower m_rates thrive
    num_of_tests: int = tk.IntVar()

#####################################################################################################################
    header  = tk.Label(master = window, text = "Enter Neural Network Population Parameters", font = "Calibri 42 bold")

    # Create Input Frame
    input_frame = ttk.Frame(master = window)

    #Create entry boxes and labels
    label_gens, entry_gens = setup_input_field("Generations (int)", "calibri 20 bold", input_frame, generations)
    
    label_pop, entry_pop = setup_input_field("Population Size (int)", "calibri 20 bold", input_frame, population_size)
    
    label_bits, entry_bits = setup_input_field("Bits Per Operand (int):", "calibri 20 bold", input_frame, bits_per_operand)

    label_tests, entry_tests = setup_input_field("Test Size (int)", "calibri 20 bold", input_frame, num_of_tests)

    label_elite_chance, entry_elite_chance = setup_input_field("Elite Percentage (float 0.0-1.0)", "calibri 20 bold", input_frame, elites_chance)

    label_signed, entry_signed = setup_input_field("Signed (bool)", "calibri 20 bold", input_frame, signed)

##########

    def evolve_generations():

        try:
            assert int(bits_per_operand.get()) >= 1
            assert int(num_of_tests.get()) >= 1
            bool(signed.get())
            assert int(population_size.get()) >= 1
            assert float(elites_chance.get()) > 0
            assert 0< elites_chance.get() <= 1.0
        except (ValueError, AssertionError):
            print("\n\nInvalid Input Fields")
        else:

            print("Start")
            

            input_bit_size = 2 * bits_per_operand.get()
            output_bit_size = bits_per_operand.get() + 1

            layer_sizes = [input_bit_size, 64, 32, output_bit_size]
            test_data = generate_test_data(bits_per_operand.get(), num_of_tests.get(), signed.get())

            population = Population(bits_per_operand.get(),
                                    signed.get(),
                                    population_size.get(),
                                    layer_sizes,
                                    elites_chance.get(),
                                    )
            
            population.evolve(generations.get(), test_data, True)

            print("End")

            plt.xlabel("Generations")
            plt.ylabel("Performance out of 100")
            plt.show()

###########


    #Create Generate button
    button_gen = ttk.Button(master = input_frame, text = "Generate", command = evolve_generations)

    # Pack components
    header.pack()

    packer((label_gens, entry_gens), (label_pop, entry_pop),
           (label_bits, entry_bits), (label_tests, entry_tests),
           (label_elite_chance, entry_elite_chance), (label_signed, entry_signed))

    button_gen.pack()

    input_frame.pack(pady = 20)

    window.mainloop()
#####################################################################################################################

def packer(*widgets):
    for widget in widgets:
        widget[0].pack()
        widget[1].pack(padx = 20, pady = 20)


def setup_input_field(name, _font, _master, _text_variable):
    label = ttk.Label(master = _master,
                           text = name,
                           font = _font)
    entry = ttk.Entry(master = _master,
                           textvariable = _text_variable)
    
    return label, entry

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
    limit = 60 # limiting the size of x ensures there's no OverFlowError, as well as needlessly large numbers being used as sigmoid rounds to 1 at 10 dp with a limit of 25
    x = max((min(x, limit)),(-limit))
    return 1/ (1 + (e ** -x))

def vectorised_sigmoid(x):
    return np.vectorize(sigmoid)(x)

def rounded_sigmoid(x):
    if x > 0:
        return 1
    else:
        return 0
    #x = max((min(x, limit)),(-limit))
    #return round((1/ (1 + (e ** -x))))

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



