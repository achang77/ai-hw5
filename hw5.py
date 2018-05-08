import scipy
import numpy as np
from sklearn import linear_model


# sample data sent by Freenor
"""
Go through first 20 slot machines 30 pulls each to generate data 

Machine 11 - string, cost, payoff
data["slots"][11][0] = metadata list
data["slots"][11][1] = cost list
data["slots"][11][2] = payoff list
data["slots"][11][3] = average cost
data["slots"][11][4] = beta distribution model
#attributes: alpha, beta, lower limit, scale
"""

state = {"team-code": "eef8976e",
"game": "phase_1",
"pulls-left": 99999,
"last-cost": 0.75,
"last-payoff": 20.7,
"last-metadata": "00110101"}

# where we store our data and models
data = {
    "slots": [[[], [], [], [], []] for x in range(100)],
    "turn": 0,
    "last_slot": None
}

parameters = {
    "high_samples": 30,
    "medium_samples": 20,
    "low_samples": 10,
    "slots_high_sample": range(70, 80),
    "slots_medium_sample": range(80, 100),
    "slots_low_sample": range(0, 70),
    "best_n_to_pull": 10,
    "best_n_to_pull_trials": 10,
}

switches = {
    "high_switch": parameters["high_samples"]*len(parameters["slots_high_sample"]),
    "medium_switch": parameters["medium_samples"]*len(parameters["slots_medium_sample"]) + parameters["high_samples"]*len(parameters["slots_high_sample"]),
    "low_switch": parameters["low_samples"]*len(parameters["slots_low_sample"]) + parameters["medium_samples"]*len(parameters["slots_medium_sample"]) + parameters["high_samples"]*len(parameters["slots_high_sample"])
}

#Slot info class - too ambitious
# class Slot(object):
#     metadata: []
#     cost: []
#     payoff: []
#     average_cost: []
#     beta_model: []
#
#     # The class "constructor" - It's actually an initializer
#     def __init__(self, metadata, cost, payoff, average_cost, beta_model):
#         self.metadata = metadata
#         self.cost = cost
#         self.payoff = payoff
#         self.average_cost = average_cost
#         self.beta_model = beta_model
#
# def make_student(name, age, major):
#     student = Student(name, age, major)
#     return student

#pull n-th slot machine and save data


def set_last_slot_and_ret_slot(n):
    data["last_slot"] = n
    return {
        "team-code": state["team-code"],
        "game": "phase_1",
        "pull": n
    }


def update_beta_coeff(slots_to_train):
    for i in slots_to_train:
        data["slots"][i][3] = np.mean(data["slots"][i][1][0])
        data["slots"][i][4] = scipy.stats.beta.fit(data["slots"][i][2][0])


def prep_linear_model_data(slots_to_train, beta_model_vars, times):
    for h in range(times):
        for i in slots_to_train:
            #append slot machine number, slot machine 8 bits separated into 1s and 0s,
            beta_model_vars.append(i, data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7], data["slots"][i][4][0], data["slots"][i][4][1], data["slots"][i][4][2], data["slots"][i][4][3])


def phase1b(slots_to_train, beta_model_vars, times):
    # create beta prediction models list
    # parse metadata to 8-bit binary variables when creating beta model variable data

    update_beta_coeff(slots_to_train)
    prep_linear_model_data(slots_to_train, beta_model_vars, times)


def phase1c(beta_model_vars):

    # generate beta coefficient prediction model from binary metadata predictors

    predictors = beta_model_vars[:, [1, 2, 3, 4, 5, 6, 7, 8]]

    alpha = linear_model.LinearRegression()
    beta = linear_model.LinearRegression()
    loc = linear_model.LinearRegression()
    scale = linear_model.LinearRegression()

    return {'alpha': alpha.fit(predictors, beta_model_vars[:, 9]), 'beta': beta.fit(predictors, beta_model_vars[:, 10]), 'loc': loc.fit(predictors, beta_model_vars[:, 11]), 'scale': scale.fit(predictors, beta_model_vars[:, 12])}


def predict_and_ret_best_slots(beta_models):
    slots_to_pull = []
    for i in range(0, len(data["slots"])):
        alpha_expected = beta_models['alpha'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        beta_expected = beta_models['beta'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        loc_expected = beta_models['loc'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        scale_expected = beta_models['scale'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        bm_expected = scipy.stats.beta.mean(alpha_expected, beta_expected, loc_expected, scale_expected)
        mean_expected = np.mean(data["slots"][i][2])
        cost = data["slots"][i][1]
        # weight actual means higher than averages
        # model is included because it may provide insight into high impact low probability payout values
        slots_to_pull.append(i, 0.3 * bm_expected + 0.7 * mean_expected - cost)
    sorted(slots_to_pull, key=lambda x: x[1])
    return slots_to_pull


def build_models():
    beta_model_vars = []
    phase1b(parameters["slots_high_sample"], beta_model_vars, 3)
    phase1b(parameters["slots_medium_sample"], beta_model_vars, 2)
    phase1b(parameters["slots_low_sample"], beta_model_vars, 1)

    return phase1c(beta_model_vars)


def rank_slots():
    beta_models = build_models()
    return predict_and_ret_best_slots(beta_models)


def get_move():
    data["turn"] = data["turn"] + 1
    n = data["last_slot"]

    if(data["turn"]==0):
        return {
            "team-code": state["team-code"],
            "game": "phase_1",
            "pull": parameters["slots_high_sample"][0]
        }

    load = load_data()
    # store last metadata string
    data["slots"][n][0].append(load["last-metadata"])
    # store last cost
    data["slots"][n][1].append(load["last-cost"])
    # store last payoff
    data["slots"][n][2].append(load["last-payoff"])

    # pull each high sample slots n times
    if data["turn"] < switches["high_switch"]:
        return set_last_slot_and_ret_slot(data["turn"] % parameters["high_samples"])

    if data["turn"] < switches["medium_switch"]:
        return set_last_slot_and_ret_slot(data["turn"] % parameters["high_samples"])

    if data["turn"] < switches["low_switch"]:
        return set_last_slot_and_ret_slot(data["turn"] % parameters["low_samples"])

    best = rank_slots()