from scipy.stats import beta

# sample data sent by Freenor
"""
Go through first 20 slot machines 30 pulls each to generate data 

Machine 11 - string, cost, payoff
data["slots"][11][0][0] = string
data["slots"][11][1][0] = cost
data["slots"][11][2][0] = payoff
"""

state = {"team-code": "eef8976e",
"game": "phase_1",
"pulls-left": 99999,
"last-cost": 0.75,
"last-payoff": 20.7,
"last-metadata": "00110101"}

# where we store our data and models
data= {
"slots": [ [ [], [], [] ] for x in range(100) ],
"last-slot":None,
"ct":0,
"our_utility":1000000,
"end-explore":0
}

#pull n-th slot machine and save data
def pull_n_save(n):
    return {
        "team-code": state["team-code"],
        "game": "phase_1",
        "pull": n
    }
    load = load_data()
    # store last metadata string
    data["slots"][n][0][0].append(load["last-metadata"])
    # store last cost
    data["slots"][n][1][0].append(load["last-cost"])
    # store last payoff
    data["slots"][n][2][0].append(load["last-payoff"])

def phase1a(slots_to_test,y):
    # x is slot machine number and y is number of trials per slot machine
    for i in slots_to_test:
        for j in range(y):
            pull_n_save(i)

def phase1b(slots_to_test):

def update_beta_models

def run_regression():
    return None
#build logistic model for first 20 slot machines

def identify():
    return None
#assign profitable/unprofitable to other 80 machines based on model
#then pull the 5%??? most profitable
#also consider track utility? if utility drops below threshold then break-out?

def phase1(state):
    last_slot = data["last-slot"]

    #model alpha, beta, and scale variable for x number of slot machines and y number of trials
    ### to-do ###
    # run phase1a on randomly selected slot machines. People will tend to train on first n slot machines, so we want to take advantage of lesser known slot machines.
    slots_to_test = range(0,29)
    phase1a(slots_to_test, 30)

    beta_models = []
    phase2a(slots_to_test, beta_models)


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