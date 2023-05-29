from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DATA_PATH = "Project4/data/inputs.csv"
OUTPUT_DATA_PATH = "Project4/data/outputs.csv"


def calculate_efficiency(inputs, outputs, id=0) -> float:
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
        model += lpSum(
            [outputs_weights[j] * outputs[i][j] for j in range(num_outputs)]
        ) <= lpSum([inputs_weights[j] * inputs[i][j] for j in range(num_inputs)])

    # Tworzenie funkcji celu
    model += lpSum([outputs_weights[i] * outputs[id][i] for i in range(num_outputs)])

    # Rozwiązanie problemu optymalizacyjnego
    model.solve()

    return round(model.objective.value(), 2)


def main():
    inputs = pd.read_csv(INPUT_DATA_PATH, delimiter=";", index_col=0)
    outputs = pd.read_csv(OUTPUT_DATA_PATH, delimiter=";", index_col=0)
    cities = inputs.index.to_list()

    inputs_np = inputs.to_numpy()
    outputs_np = outputs.to_numpy()

    efficiencies = []
    for i in range(inputs.shape[0]):
        efficiency = calculate_efficiency(inputs_np, outputs_np, id=i)
        efficiencies.append(efficiency)
    
    for city, e in zip(cities, efficiencies):
        print(f"{city}: {e}")


if __name__ == "__main__":
    main()
