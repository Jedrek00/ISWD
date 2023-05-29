from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DATA_PATH = "Project4/data/inputs.csv"
OUTPUT_DATA_PATH = "Project4/data/outputs.csv"


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
    model.solve()

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
    num_inputs = inputs.shape[1]
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


def main():
    inputs = pd.read_csv(INPUT_DATA_PATH, delimiter=";", index_col=0)
    outputs = pd.read_csv(OUTPUT_DATA_PATH, delimiter=";", index_col=0)
    cities = inputs.index.to_list()

    inputs_np = inputs.to_numpy()
    outputs_np = outputs.to_numpy()

    efficiencies = calculate_efficiency(inputs_np, outputs_np)
    super_efficiencies = calculate_efficiency(inputs_np, outputs_np, super_efficiency=True)
    cross_efficiencies = cross_efficiency(inputs_np, outputs_np, cities)
    
    print("Efficiencies")
    for city, e in zip(cities, efficiencies):
        print(f"{city}: {e}")

    print("Super-Efficiencies")
    for city, e in zip(cities, super_efficiencies):
        print(f"{city}: {e}")

    print("Cross Efficiencies")
    print(cross_efficiencies)

    


if __name__ == "__main__":
    main()
