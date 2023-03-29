from pulp import *
import numpy as np
import matplotlib.pyplot as plt


def solve(print_values: bool = False):
    # Create instance of problem
    model = LpProblem(name="nuclear_waste_management", sense=LpMaximize)

    # Create variables for each feature
    epsilon = LpVariable(name="eps", lowBound=0, cat="Continuous")

    # minimal values
    u1_32 = LpVariable(name="u1_32", lowBound=0, cat="Continuous")
    u2_03 = LpVariable(name="u2_03", lowBound=0, cat="Continuous")
    u3_00 = LpVariable(name="u3_00", lowBound=0, cat="Continuous")
    u4_49 = LpVariable(name="u4_49", lowBound=0, cat="Continuous")

    # maximial values
    u1_100 = LpVariable(name="u1_100", lowBound=0, cat="Continuous")
    u2_100 = LpVariable(name="u2_100", lowBound=0, cat="Continuous")
    u3_100 = LpVariable(name="u3_100", lowBound=0, cat="Continuous")
    u4_100 = LpVariable(name="u4_100", lowBound=0, cat="Continuous")

    # example 11
    u1_61 = LpVariable(name="u1_61", lowBound=0, cat="Continuous")
    u2_54 = LpVariable(name="u2_54", lowBound=0, cat="Continuous")
    u3_38 = LpVariable(name="u3_38", lowBound=0, cat="Continuous")
    u4_49 = LpVariable(name="u4_49", lowBound=0, cat="Continuous")

    # example 14
    u1_69 = LpVariable(name="u1_69", lowBound=0, cat="Continuous")
    u2_49 = LpVariable(name="u2_49", lowBound=0, cat="Continuous")
    u3_56 = LpVariable(name="u3_56", lowBound=0, cat="Continuous")
    u4_61 = LpVariable(name="u4_61", lowBound=0, cat="Continuous")

    # example 2
    u1_66 = LpVariable(name="u1_66", lowBound=0, cat="Continuous")
    u2_55 = LpVariable(name="u2_55", lowBound=0, cat="Continuous")
    u3_45 = LpVariable(name="u3_45", lowBound=0, cat="Continuous")
    u4_49 = LpVariable(name="u4_49", lowBound=0, cat="Continuous")

    # example 25
    u1_34 = LpVariable(name="u1_34", lowBound=0, cat="Continuous")
    # u2_100 = LpVariable(name='u2_100' , lowBound=0, cat='Continuous')
    # u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    # u4_100 = LpVariable(name='u4_100' , lowBound=0, cat='Continuous')

    # example 27
    u1_80 = LpVariable(name="u1_80", lowBound=0, cat="Continuous")
    u2_06 = LpVariable(name="u2_06", lowBound=0, cat="Continuous")
    # u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    u4_67 = LpVariable(name="u4_67", lowBound=0, cat="Continuous")

    # example 15
    u1_87 = LpVariable(name="u1_87", lowBound=0, cat="Continuous")
    # u2_03 = LpVariable(name='u2_03' , lowBound=0, cat='Continuous')
    # u3_100 = LpVariable(name='u3_100' , lowBound=0, cat='Continuous')
    # u4_61 = LpVariable(name='u4_61' , lowBound=0, cat='Continuous')

    u1_all = [u1_32, u1_34, u1_61, u1_66, u1_69, u1_80, u1_87, u1_100]
    u2_all = [u2_03, u2_06, u2_49, u2_54, u2_55, u2_100]
    u3_all = [u3_00, u3_38, u3_45, u3_56, u3_100]
    u4_all = [u4_49, u4_61, u4_67, u4_100]
    u_all = [u1_all, u2_all, u3_all, u4_all]

    # reference rankingProject1/solver.py Project1/Nuclear waste management.csv
    model += (
        u1_61 + u2_54 + u3_38 + u4_49 >= u1_66 + u2_55 + u3_45 + u4_49 + epsilon,
        "#11_over_2",
    )
    model += (
        u1_66 + u2_55 + u3_45 + u4_49 >= u1_69 + u2_49 + u3_56 + u4_61 + epsilon,
        "#2_over_14",
    )
    model += (
        u1_69 + u2_49 + u3_56 + u4_61 == u1_34 + u2_100 + u3_100 + u4_100,
        "#14_equal_25",
    )
    model += (
        u1_34 + u2_100 + u3_100 + u4_100 >= u1_80 + u2_06 + u3_100 + u4_67 + epsilon,
        "#25_over_27",
    )
    model += (
        u1_80 + u2_06 + u3_100 + u4_67 == u1_87 + u2_03 + u3_100 + u4_61,
        "#27_equal_15",
    )

    # normalization
    model += (u1_32 + u2_03 + u3_00 + u4_49 == 1, "#normalization")
    model += (u1_100 == 0, "#u1_min")
    model += (u2_100 == 0, "#u2_min")
    model += (u3_100 == 0, "#u3_min")
    model += (u4_100 == 0, "#u4_min")

    # monotonicity: cost criterias
    for u in u_all:
        for a, b in zip(u, u[1:]):
            model += a >= b

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

    criteria = []
    min_values = []
    for u in u_all:
        arr = np.array([])
        min_values.append(int(u[0].name.split("_")[1]))
        for a, b in zip(u, u[1:]):
            a_start, a_value = int(a.name.split("_")[1]), a.value()
            b_start, b_value = int(b.name.split("_")[1]), b.value()
            arr = np.concatenate(
                (arr, np.linspace(a_value, b_value, (b_start - a_start)))
            )
        criteria.append(arr)
    return criteria, min_values


def add_to_model_greater(crit, matrix, pair, epsilon):
    a, b = pair
    return (
        crit[0][matrix[a][1]]
        + crit[1][matrix[a][2]]
        + crit[2][matrix[a][3]]
        + crit[3][matrix[a][4]]
        >= crit[0][matrix[b][1]]
        + crit[1][matrix[b][2]]
        + crit[2][matrix[b][3]]
        + crit[3][matrix[b][4]]
        + epsilon
    )


def add_to_model_equal(crit, matrix, pair):
    a, b = pair
    return (
        crit[0][matrix[a][1]]
        + crit[1][matrix[a][2]]
        + crit[2][matrix[a][3]]
        + crit[3][matrix[a][4]]
        == crit[0][matrix[b][1]]
        + crit[1][matrix[b][2]]
        + crit[2][matrix[b][3]]
        + crit[3][matrix[b][4]]
    )


def solve_gms(crit, matrix, pair):
    # Create instance of problem
    model = LpProblem(name="nuclear_waste_management", sense=LpMaximize)

    # Create variables for each feature
    epsilon = LpVariable(name="eps", cat="Continuous")

    u1_all = [y for _, y in sorted(crit[0].items())]
    u2_all = [y for _, y in sorted(crit[1].items())]
    u3_all = [y for _, y in sorted(crit[2].items())]
    u4_all = [y for _, y in sorted(crit[3].items())]
    u_all = [u1_all, u2_all, u3_all, u4_all]

    # reference rankingProject1/solver.py Project1/Nuclear waste management.csv
    model += add_to_model_greater(crit, matrix, (10, 1), epsilon)
    model += add_to_model_greater(crit, matrix, (1, 13), epsilon)
    model += add_to_model_equal(crit, matrix, (13, 24))
    model += add_to_model_greater(crit, matrix, (24, 26), epsilon)
    model += add_to_model_equal(crit, matrix, (26, 14))

    # normalization
    model += (u1_all[0] + u2_all[0] + u3_all[0] + u4_all[0] == 1, "#normalization")
    model += (u1_all[-1] == 0, "#u1_min")
    model += (u2_all[-1] == 0, "#u2_min")
    model += (u3_all[-1] == 0, "#u3_min")
    model += (u4_all[-1] == 0, "#u4_min")

    # monotonicity: cost criterias
    for u in u_all:
        for a, b in zip(u, u[1:]):
            model += a >= b

    # non-negativity
    for u in u_all:
        for a in u:
            model += a >= 0

    model += add_to_model_greater(crit, matrix, pair, epsilon)

    # maximize
    obj_func = epsilon
    model += obj_func

    # run solver
    status = model.solve()

    return model.objective.value()


def create_ranking(matrix, criteria, min_values):
    ranking = {}
    for row in matrix:
        id = int(row[0]) - 1
        score = 0
        print(row)
        for col in range(len(criteria)):
            crit_id = int(matrix[id][col + 1] * 100)
            if crit_id - min_values[col] < len(criteria[col]):
                idx = crit_id - min_values[col]
            else:
                idx = crit_id - min_values[col] - 1
            # score += matrix[id][col+1] * criteria[col][crit_id - min_values[col]]
            score += criteria[col][idx]
        ranking[id + 1] = score

    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    return ranking


def plot_criteria(criteria, min_values):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):
            id = i * 2 + j
            axes[i][j].plot(
                np.linspace(min_values[id] / 100, 1.0, 100 - min_values[id]),
                criteria[id],
            )
            axes[i][j].set_title(f"Criterion {id+1}")
    plt.show()


if __name__ == "__main__":

    matrix = []
    path = "Project1/Nuclear waste management.csv"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            matrix.append(list(map(float, line[:-1].split(","))))
    # print(matrix)

    crit = [{}, {}, {}, {}]
    for line in matrix:
        for var in range(1, len(line)):
            crit[var - 1][line[var]] = LpVariable(
                name=(str(int(line[0])) + '_' + "u" + str(var) + "_" + str(int(line[var] * 100))),
                lowBound=0,
                cat="Continuous",
            )
    outcome = np.zeros([27, 27])
    for x in range(outcome.shape[0]):
        for y in range(outcome.shape[1]):
            if solve_gms(crit, matrix, (x, y)) > 0:
                outcome[x, y] = 1
            elif solve_gms(crit, matrix, (x, y)) == 0:
                outcome[x, y] = 0
            else:
                outcome[x, y] = -1

    for x in range(outcome.shape[0]):
        for y in range(outcome.shape[1]):
            if x == y:
                continue
            print((x + 1, y + 1), end=": ")
            if outcome[x, y] >= 0 and outcome[y, x] <= 0:
                print("konieczne")
            elif outcome[x, y] >= 0:
                print("możliwe")

    
    # for var in range(1, len(line)):
    #   print(crit[var - 1][line[var]])

    # criteria, min_values = solve()
#
# ranking = create_ranking(matrix, criteria, min_values)
#
# for i, row in enumerate(ranking):
#    print(f"{i+1}: {row}")
#
# plot_criteria(criteria, min_values)
