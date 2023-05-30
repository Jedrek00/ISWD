from pulp import *
import numpy as np
import pandas as pd

INPUT_DATA_PATH = "Project4/data/inputs.csv"
OUTPUT_DATA_PATH = "Project4/data/outputs.csv"
SAMPLES_PATH = "Project4/data/samples_homework.csv"
BINS = 5


def solve_CCR(inputs, outputs, id, super_efficiency=False) -> LpProblem:
    num_examples = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    # Tworzenie problemu optymalizacyjnego
    model = LpProblem("EfficiencyCalculation", LpMaximize)

    # Tworzenie zmiennej decyzyjnej dla każdego wejścia oraz wyjścia
    inputs_weights = [
        LpVariable(f"v_{i}", lowBound=0, cat="Continuous") for i in range(num_inputs)
    ]
    outputs_weights = [
        LpVariable(f"u_{i}", lowBound=0, cat="Continuous") for i in range(num_outputs)
    ]

    # Dodanie ograniczeń
    model += lpSum([inputs_weights[i] * inputs[id][i] for i in range(num_inputs)]) == 1

    for i in range(num_examples):
        if super_efficiency:
            if i == id:
                continue
        model += lpSum(
            [outputs_weights[j] * outputs[i][j] for j in range(num_outputs)]
        ) <= lpSum([inputs_weights[j] * inputs[i][j] for j in range(num_inputs)])

    # Tworzenie funkcji celu
    model += lpSum([outputs_weights[i] * outputs[id][i] for i in range(num_outputs)])

    # Rozwiązanie problemu optymalizacyjnego
    model.solve(PULP_CBC_CMD(msg=0))

    return model


def calculate_efficiency(inputs, outputs, super_efficiency=False):
    efficiencies = []
    for i in range(inputs.shape[0]):
        model = solve_CCR(inputs, outputs, i, super_efficiency)
        efficiency = round(model.objective.value(), 3)
        efficiencies.append(efficiency)
    return efficiencies


def cross_efficiency(inputs, outputs, cities):
    num_examples = inputs.shape[0]
    num_outputs = outputs.shape[1]

    cross_efficiencies = pd.DataFrame(index=cities, columns=cities)

    for i in range(num_examples):
        model = solve_CCR(inputs, outputs, i)
        params = [p.value() for p in model.variables()]
        output_params, input_params = params[:num_outputs], params[num_outputs:]        
        for j in range(num_examples):
            value = round(np.sum(outputs[j] * output_params) / np.sum(inputs[j] * input_params), 3)
            cross_efficiencies.iloc[i, j] = value
    
    cross_efficiencies.loc['CRE_k'] = cross_efficiencies.mean()
    return cross_efficiencies.T


def distribution_of_efficiencies(inputs, outputs, samples, bins):
    num_examples = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_samples = samples.shape[0]

    effs = np.zeros((num_examples, num_samples))
    boundries = np.linspace(0, 1, bins+1)

    for i in range(num_examples):
        for j, sample in enumerate(samples):
            sample_inputs, sample_outputs = sample[:num_inputs], sample[num_inputs:]
            value = np.sum(outputs[i] * sample_outputs) / np.sum(inputs[i] * sample_inputs)
            effs[i, j] = value

    effs /= np.max(effs, axis=0)
    mean_values = np.mean(effs, axis=1)

    descrete = np.digitize(effs, boundries, True)

    number_of_occurence = []
    for row in descrete:
        values, counts = np.unique(row, return_counts=True)
        occurences = [0] * bins
        for i, value in enumerate(values):
            occurences[value-1] = counts[i]
        number_of_occurence.append(occurences)
    number_of_occurence = np.array(number_of_occurence, dtype=np.float32)
    number_of_occurence /= num_samples
    result = np.concatenate([number_of_occurence, np.expand_dims(mean_values, axis=1)], axis=1)
    
    return result

def display_ranking(values: list, cities:list):
    sorted_values, sorted_cities = zip(*sorted(zip(values, cities), reverse=True))
    for i,  (value, city) in enumerate(zip(sorted_values, sorted_cities)):
        print(f"{i+1}. {city}: {value:.3f}")



def main():
    # read data
    inputs = pd.read_csv(INPUT_DATA_PATH, delimiter=";", index_col=0)
    outputs = pd.read_csv(OUTPUT_DATA_PATH, delimiter=";", index_col=0)
    samples = pd.read_csv(SAMPLES_PATH, delimiter=";", index_col=0)

    cities = inputs.index.to_list()

    # transform to numpy arrays
    inputs_np = inputs.to_numpy()
    outputs_np = outputs.to_numpy()
    samples_np = samples.to_numpy()

    # calculate different types of efficiencies
    efficiencies = calculate_efficiency(inputs_np, outputs_np)
    super_efficiencies = calculate_efficiency(inputs_np, outputs_np, super_efficiency=True)
    cross_efficiencies = cross_efficiency(inputs_np, outputs_np, cities)
    distribution = distribution_of_efficiencies(inputs_np, outputs_np, samples_np, BINS)
    
    # display results
    print("Efficiencies")
    for city, e in zip(cities, efficiencies):
        print(f"{city}: {e}")

    print("Super-Efficiencies")
    for city, e in zip(cities, super_efficiencies):
        print(f"{city}: {e}")

    print("Cross Efficiencies")
    print(cross_efficiencies)

    print("Distribution Of Efficiencies")
    print(pd.DataFrame(distribution, index=cities))

    print("==RANKINGS==")
    print("Super-Efficiencies")
    display_ranking(super_efficiencies, cities)
    print("Mean values from Cross Efficiencies")
    display_ranking(cross_efficiencies['CRE_k'].to_list(), cities)
    print("Expected values of Efficiencies")
    display_ranking(list(distribution[:, BINS]), cities)

    
if __name__ == "__main__":
    main()
