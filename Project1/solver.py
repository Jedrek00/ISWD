from pulp import *
import numpy as np
import matplotlib.pyplot as plt


def solve(print_values: bool = False):
    # Create instance of problem
    model = LpProblem(name="nuclear_waste_management", sense=LpMaximize)

    # Create variables for each feature
    epsilon = LpVariable(name="eps", lowBound=0, cat="Continuous")

    # minimal values
    u1_32 = LpVariable(name='u1_32' , lowBound=0, cat='Continuous')
    u2_03 = LpVariable(name='u2_03' , lowBound=0, cat='Continuous')
    u3_00 = LpVariable(name='u3_00' , lowBound=0, cat='Continuous')
    u4_49 = LpVariable(name='u4_49' , lowBound=0, cat='Continuous')

    # maximial values
    u1_100 = LpVariable(name='u1_100' , lowBound=0, cat='Continuous')
    u2_100 = LpVariable(name='u2_100' , lowBound=0, cat='Continuous')
    u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    u4_100 = LpVariable(name='u4_100' , lowBound=0, cat='Continuous')

    # example 11
    u1_61 = LpVariable(name='u1_61' , lowBound=0, cat='Continuous')
    u2_54 = LpVariable(name='u2_54' , lowBound=0, cat='Continuous')
    u3_38 = LpVariable(name='u3_38' , lowBound=0, cat='Continuous')
    u4_49 = LpVariable(name='u4_49' , lowBound=0, cat='Continuous')

    # example 14
    u1_69 = LpVariable(name='u1_69' , lowBound=0, cat='Continuous')
    u2_49 = LpVariable(name='u2_49' , lowBound=0, cat='Continuous')
    u3_56 = LpVariable(name='u3_56' , lowBound=0, cat='Continuous')
    u4_61 = LpVariable(name='u4_61' , lowBound=0, cat='Continuous')

    # example 1
    u1_60 = LpVariable(name='u1_60' , lowBound=0, cat='Continuous')
    u2_93 = LpVariable(name='u2_93' , lowBound=0, cat='Continuous')
    # u3_00 = LpVariable(name='u3_00' , lowBound=0, cat='Continuous')
    u4_73 = LpVariable(name='u4_73' , lowBound=0, cat='Continuous')

    # example 25
    u1_34 = LpVariable(name='u1_34' , lowBound=0, cat='Continuous')
    # u2_100 = LpVariable(name='u2_100' , lowBound=0, cat='Continuous')
    # u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    # u4_100 = LpVariable(name='u4_100' , lowBound=0, cat='Continuous')

    # example 21
    u1_83 = LpVariable(name='u1_83' , lowBound=0, cat='Continuous')
    u2_25 = LpVariable(name='u2_25' , lowBound=0, cat='Continuous')
    u3_80 = LpVariable(name='u3_80' , lowBound=0, cat='Continuous')
    u4_65 = LpVariable(name='u4_65' , lowBound=0, cat='Continuous')

    # example 15
    u1_87 = LpVariable(name='u1_87' , lowBound=0, cat='Continuous')
    # u2_03 = LpVariable(name='u2_03' , lowBound=0, cat='Continuous')
    # u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    # u4_61 = LpVariable(name='u4_61' , lowBound=0, cat='Continuous')

    u1_all = [u1_32, u1_34, u1_60, u1_61, u1_69, u1_83, u1_87, u1_100]
    u2_all = [u2_03, u2_25, u2_49, u2_54, u2_93, u2_100]
    u3_all = [u3_00, u3_38, u3_56, u3_80, u3_100]
    u4_all = [u4_49, u4_61, u4_65, u4_73, u4_100]
    u_all = [u1_all, u2_all, u3_all, u4_all]

    # reference ranking
    model += (u1_61 + u2_54 + u3_38 + u4_49 >= u1_69 + u2_49 + u3_56 + u4_61 + epsilon, "#11_over_14")
    model += (u1_69 + u2_49 + u3_56 + u4_61 == u1_60 + u2_93 + u3_00 + u4_73, "#14_equal_1")
    model += (u1_60 + u2_93 + u3_00 + u4_73 >= u1_34 + u2_100 + u3_100 + u4_100 + epsilon, "#1_over_25")
    model += (u1_34 + u2_100 + u3_100 + u4_100 >= u1_83 + u2_25 + u3_80 + u4_65 + epsilon, "#25_over_21")
    model += (u1_83 + u2_25 + u3_80 + u4_65 == u1_87 + u2_03 + u3_100 + u4_61, "#21_equal_15")

    # normalization
    model += (u1_32 + u2_100 + u3_100 + u4_49 == 1, "#normalization")
    model += (u1_100 == 0, "#u1_min")
    model += (u2_03 == 0, "#u2_min")
    model += (u3_00 == 0, "#u3_min")
    model += (u4_100 == 0, "u4_min")

    # monotonicity: cost criterias
    for u in [u1_all, u4_all]:
        for a, b in zip(u, u[1:]):
            model += a >= b
    
    # monotonicity: gain criterias
    for u in [u2_all, u3_all]:
        for a, b in zip(u, u[1:]):
            model += a <= b

    # non-negativity
    for u in u_all:
        for a in u:
            model += a >= 0

    # maximize
    obj_func = epsilon
    model += obj_func

    # run solver
    status = model.solve()
    print(f"status: {model.status}, {LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")

    if print_values:
        for u in u_all:
            for a in u:
                print(f"{a.name}: {a.value()}")

    criterias = []
    for u in u_all:
        arr = np.array([])
        for a, b in zip(u, u[1:]):
            a_start, a_value = int(a.name.split("_")[1]), a.value()
            b_start, b_value = int(b.name.split("_")[1]), b.value()
            arr = np.concatenate((arr, np.linspace(a_value, b_value, (b_start - a_start))))
        criterias.append(arr)
    
    return criterias


def create_ranking(matrix, criterias, min_values):
    ranking = {}
    for row in matrix:
        id = int(row[0]) - 1
        score = 0
        for col in range(len(criterias)):
            crit_id = int(matrix[id][col+1] * 100) - 1
            score += matrix[id][col+1] * criterias[col][crit_id - min_values[col]]
        ranking[id+1] = score

    ranking = sorted(ranking.items(), key=lambda x:x[1], reverse=True)
    return ranking


def plot_criterias(criterias, min_values):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):
            id = i*2+j
            axes[i][j].plot(np.linspace(min_values[id] / 100, 1.0, 100 - min_values[id]), criterias[id])
            axes[i][j].set_title(f"Criteria {id+1}")
    plt.show()


if __name__ == "__main__":

    matrix = []
    path = "Project1/Nuclear waste management.csv"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            matrix.append(list(map(float, line[:-2].split(","))))
    
    criterias = solve()

    min_values = [100 - len(criteria) for criteria in criterias]

    ranking = create_ranking(matrix, criterias, min_values)

    for i, row in enumerate(ranking):
        print(f"{i+1}: {row}")

    plot_criterias(criterias, min_values)
