import scipy.stats.beta
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
data= {
"slots": [ [ [], [], [], [], [] ] for x in range(100) ],
"last-slot":None,
"ct":0,
"our_utility":1000000,
"end-explore":0
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

def pull_n_save(n):
    return {
        "team-code": state["team-code"],
        "game": "phase_1",
        "pull": n
    }
    load = load_data()
    # store last metadata string
    data["slots"][n][0].append(load["last-metadata"])
    # store last cost
    data["slots"][n][1].append(load["last-cost"])
    # store last payoff
    data["slots"][n][2].append(load["last-payoff"])
    data["ct"] = ct+1

def phase1a(slots_to_train,y):
    # x is slot machine number and y is number of trials per slot machine
    for i in slots_to_train:
        for j in range(y):
            pull_n_save(i)


def update_beta_coeff(slots_to_train):
    for i in slots_to_train:
        data["slots"][i][3] = np.mean(data["slots"][i][1][0])
        data["slots"][i][4] = scipy.stats.beta.fit(data["slots"][i][2][0])

def add_beta_model_vars(slots_to_train, beta_model_vars, times):
    for h in range(times):
        for i in slots_to_train:
            #append slot machine number, slot machine 8 bits separated into 1s and 0s,
            beta_model_vars.append(i, data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7], data["slots"][i][4][0], data["slots"][i][4][1], data["slots"][i][4][2], data["slots"][i][4][3])

def phase1b(slots_to_train, beta_model_vars, times):
    # create beta prediction models list
    # parse metadata to 8-bit binary variables when creating beta model variable data

    update_beta_coeff(slots_to_train)
    add_beta_model_vars(slots_to_train, beta_model_vars, times)

def phase1c(beta_model_vars, beta_models):

    # generate beta coefficient prediction model from binary metadata predictors

    predictors = beta_model_vars[:, [1, 2, 3, 4, 5, 6, 7, 8]]

    alpha = linear_model.LinearRegression()
    beta = linear_model.LinearRegression()
    loc = linear_model.LinearRegression()
    scale = linear_model.LinearRegression()

    beta_models = {'alpha': alpha.fit(predictors, beta_model_vars[:, 9]), 'beta': beta.fit(predictors, beta_model_vars[:, 10]), 'loc': loc.fit(predictors, beta_model_vars[:, 11]), 'scale': scale.fit(predictors, beta_model_vars[:, 12])}

def phase1d(beta_models, slots_to_pull):
    for i in range(0, len(data["slots"])):
        alpha_expected = beta_models['alpha'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        beta_expected = beta_models['beta'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        loc_expected = beta_models['loc'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        scale_expected = beta_models['scale'].predict([data["slots"][i][0][0], data["slots"][i][0][1], data["slots"][i][0][2], data["slots"][i][0][3], data["slots"][i][0][4], data["slots"][i][0][5], data["slots"][i][0][6], data["slots"][i][0][7]])
        bm_expected = scipy.stats.beta.mean(alpha_expected, beta_expected, loc_expected, scale_expected)
        mean_expected = np.mean(data["slots"][11][i])
        cost = data["slots"][i][1]
        # weight actual means higher than averages. model is included because it may provide insight into high impact low probability payout vallues.
        slots_to_pull.append(i, 0.3 * bm_expected + 0.7 * mean_expected - cost)
    sorted(slots_to_pull, key=lambda x: x[1])

def phase1(state):
    ### to do ###
    # Add if condition to not pull if too many negative expected payouts

    last_slot = data["last-slot"]

    #model alpha, beta, and scale variable for x number of slot machines and y number of trials
    ### to-do ###
    # run phase1a on randomly selected slot machines. People will tend to train on first n slot machines, so we want to take advantage of lesser known slot machines.

    slots_high_sample = range(70, 79)
    slots_medium_sample = range(80, 99)
    slots_low_sample = range(0, 69)
    slots_all = range(0, 99)

    # slots_to_predict = range(0,99)
    # #create list of slots not in train by removing items in 0-99 range list
    # for i in len(slots_to_predict):
    #     if i in slots_to_train:
    #         slots_to_predict.remove(i)


    phase1a(slots_high_sample, 30)
    phase1a(slots_medium_sample, 20)
    phase1a(slots_low_sample, 10)

    beta_model_vars = None
    beta_models = None
    next_slots_to_pull = None

    phase1b(slots_high_sample, beta_model_vars, 3)
    phase1b(slots_medium_sample, beta_model_vars, 2)
    phase1b(slots_low_sample, beta_model_vars, 1)

    phase1c(beta_model_vars, beta_models)
    phase1d(beta_models, next_slots_to_pull)



# def run_regression():
#     return None
# #build logistic model for first 20 slot machines
#
# def identify():
#     return None
# #assign profitable/unprofitable to other 80 machines based on model
# #then pull the 5%??? most profitable
# #also consider track utility? if utility drops below threshold then break-out?

# if len(data["slots"][last_slot][0]) == 0:
    #     data["slots"][last_slot][0].append(state["last-metadata"])
    #
    # if len(data["slots"][last_slot][1]) == 0:
    #     data["slots"][last_slot][1].append(state["last-cost"])
    #
    # data["slots"][last_slot][2].append(state["last-payoff"])
    #
    # if last_slot < 20:
    #     if data["ctr"] < 30:
    #         data["ctr"]+=1
    #     else:
    #         last_slot += 1
    #         data["last-slot"] = last_slot
    #         data["ctr"] = 0
    #     ret = {
    #         "team-code": state["team-code"],
    #         "game": "phase_1",
    #         "pull": last_slot,
    #     }
    # elif last_slot == 99:
    #     data["end-explore"] = 1
    #     for i in range(20):
    #         u = data["slots"][i][1][0]
    #         x = data["slots"][i][2]
    #         a,b,c,d = beta.fit(x) #0 for min?
    #         m = beta.mean(a,b,c,d)
    #         v = beta.var(a,b,c,d)
    #         if m - u > v**.5:
    #             data["slots"][i][0].append(1) #profitable
    #         else:
    #             data["slots"][i][0].append(0) #unprofitable
    #         run_regression()
    #         identify()
    #
    #         ret = {
    #         "team-code": state["team-code"],
    #         "game": "phase_1",
    #         "pull": ""#profitable as identified
    #         }
    #
    # else:
    #     last_slot += 1
    #     data["last-slot"] = last_slot
    #     if last_slot < 100:
    #         ret = {
    #         "team-code": state["team-code"],
    #         "game": "phase_1",
    #         "pull": last_slot,
    #         }
    #     return ret