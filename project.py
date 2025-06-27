from random import random, randint
from math import log
import winsound


class Network():
    def __init__(self, ins, id):

        self.ins: int = ins
        self.weights: dict = {"id" : id,
                              "accuracy" : 0,
                              "fitness" : None,
                              "relative fitness" : None,
                              "bias" : random() - random()}
        self.node: float = 0.0

        self.max = 2 ** (1 + self.ins) - 1
        for i in range(self.ins*2):
            self.weights[i] = (random())
        
    def calculate(self, bin_a: list, bin_b: list):

        self.node = 0

        size = len(bin_a) * 2
        super_list = bin_a + bin_b
        for i in range(size):
            #print(f"{i}) Node: {self.node}, bit: {super_list[i]}, weight: {self.weights[i]}")
            self.node += super_list[i] * self.weights[i]
        #print(32 * self.node/size)

        sum = self.node + self.weights["bias"]
        avg = sum/size
        guess = round(self.max * avg)
        return guess
    
    def check_fitness(self, size, bin_a, bin_b, ans, this_type):
        total = 0
        perfects = 0
        for i in range(size):
            fitness = (ans[i] - self.calculate(bin_a[i], bin_b[i]))
            if fitness == 0:
                perfects += 1
            total += abs(fitness)

        fit_score= total/size

        self.weights["accuracy"] = f"{100 * perfects/size}%"
        self.weights["fitness"] = fit_score
        self.weights["relative fitness"] = f"{round(200 * fit_score/self.max, 2)}%"

        if this_type:
            return fit_score
        return 100 * perfects/size
    
    def inherit_genes(self, other, mutation_factor):
        self.weights["bias"] = other.weights["bias"] * (random() - random())
        for i in range(self.ins):
            mutation_factor = 1 + 4*random()*random() - 4*random()*random()
            self.weights[i] = other.weights[i] * mutation_factor
        

def main():

    print("Start")

    best: int

    calcs = []
    fits = []

    frequency = 300
    length = 800

    species_size = 1_000
    bits = 4
    testing_size = 100
    generations = 10000

    # create Network objects
    for i in range(species_size):
        current_calc = Network(bits, i)
        calcs.append(current_calc)

    operand_a, operand_b, correct_answers = generate_training_data(bits, testing_size)

    for gen in range(generations):
        # check fitnesses
        fits.clear()
        
#        for i in range(species_size):
#            current_fits = calcs[i].check_fitness(testing_size, operand_a, operand_b, correct_answers)
#
#            fits.append(current_fits)

        for i in range(species_size):
            current_accuracy = calcs[i].check_fitness(testing_size, operand_a, operand_b, correct_answers, False)

            fits.append(current_accuracy)



        best_id = find_best_accuracy(fits)
        #print(gen, gen, gen, best_id, len(calcs))

        #print(calcs[best_id].calculate([0,0,1,1],[1,0,0,0]), fits[best_id])
        #print(fits[best_id])
        #print(gen, gen, gen, best_id, len(calcs))
        print(gen,
              "id:", calcs[best_id].weights["id"],
              "acc:", calcs[best_id].weights["accuracy"],
              "fit:", calcs[best_id].weights["fitness"],
              "rel_fit:", calcs[best_id].weights["relative fitness"])

        for i in range(species_size):
            if i != best_id:
                mutation_rate = 1 + 1*random() - 0.1*random()
                calcs[i].inherit_genes(calcs[best_id], mutation_rate)
                
    print(calcs[best_id].weights)

    for _ in range(2):
        winsound.Beep(frequency, length)


def generate_training_data(bits, data_size):

    max = 2 ** bits - 1

    operand_a = []
    operand_b = []
    correct_answers = []

    for _ in range(data_size):
        a = randint(1, max)
        b = randint(1, max)

        operand_a.append(den_to_bin(a, bits))
        operand_b.append(den_to_bin(b, bits))

        correct_answers.append(a + b)

    return operand_a, operand_b, correct_answers

def den_to_bin(denery, bits):
    bin = []
    values = []

    if denery > 2 ** bits:
        raise ValueError("Invalid bit size")

    for i in range(bits):
        values.insert(0, 2 ** i)

    for val in values:
        if val <= denery:
            denery -= val
            bin.append(1)
        else:
            bin.append(0)

    return bin

def find_best_fitness(fits):
    lowest_fitness = 1000
    best_id = 0
    current_id = 0

    for fitness in fits:
        if fitness < lowest_fitness:
            lowest_fitness = fitness
            best_id = current_id

        current_id +=1

    return best_id

def find_best_accuracy(accuracies):
    highest_accuracy = 0
    best_id = 0
    current_id = 0

    for accuracy in accuracies:
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_id = current_id

        current_id +=1

    return best_id

if __name__ == "__main__":
    main()