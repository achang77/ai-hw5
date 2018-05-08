import scipy
import numpy as np
from sklearn import linear_model

# where we store our data and models
data = {
    "slots": [[[], [], [], [], []] for x in range(100)],
    "turn": -1,
    "last_slot": None,
    "best": None,
    "models":None,
    "expectations": None,
    "secret-bids": None,
}

parameters = {
    "high_samples": 10,
    "medium_samples": 7,
    "low_samples": 5,
    "slots_high_sample": range(70, 80),
    "slots_medium_sample": range(80, 100),
    "slots_low_sample": range(0, 70),
    "best_n_to_pull": 20,
    "best_n_to_pull_trials": 10,
}

switches = {
    "high_switch": parameters["high_samples"]*len(parameters["slots_high_sample"]),
    "medium_switch": parameters["medium_samples"]*len(parameters["slots_medium_sample"]) + parameters["high_samples"]*len(parameters["slots_high_sample"]),
    "low_switch": parameters["low_samples"]*len(parameters["slots_low_sample"]) + parameters["medium_samples"]*len(parameters["slots_medium_sample"]) + parameters["high_samples"]*len(parameters["slots_high_sample"]),
    "best_n_switch": parameters["best_n_to_pull_trials"]*parameters["best_n_to_pull"]
}

state = {"team-code": "eef8976e",
             "game": "phase_1",
             "pulls-left": 10000,
             "last-cost": 0,
             "last-payoff": 0,
             "last-metadata": "00000001"}

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
            beta_model_vars.append([i, data["slots"][i][0][0][0], data["slots"][i][0][0][1], data["slots"][i][0][0][2], data["slots"][i][0][0][3], data["slots"][i][0][0][4], data["slots"][i][0][0][5], data["slots"][i][0][0][6], data["slots"][i][0][0][7], data["slots"][i][4][0], data["slots"][i][4][1], data["slots"][i][4][2], data["slots"][i][4][3]])


def phase1b(slots_to_train, beta_model_vars, times):
    # create beta prediction models list
    # parse metadata to 8-bit binary variables when creating beta model variable data

    update_beta_coeff(slots_to_train)
    prep_linear_model_data(slots_to_train, beta_model_vars, times)


def phase1c(beta_model_vars):

    # generate beta coefficient prediction model from binary metadata predictors
    temp = np.array(beta_model_vars).astype(float)

    predictors = temp[:, [1, 2, 3, 4, 5, 6, 7, 8]]

    alpha = linear_model.LinearRegression()
    beta = linear_model.LinearRegression()
    loc = linear_model.LinearRegression()
    scale = linear_model.LinearRegression()

    alpha.fit(predictors, temp[:, 9])
    beta.fit(predictors, temp[:, 10])
    loc.fit(predictors, temp[:, 11])
    scale.fit(predictors, temp[:, 12])

    return {'alpha': alpha, 'beta': beta, 'loc': loc, 'scale': scale}


def build_models():
    beta_model_vars = []
    phase1b(parameters["slots_high_sample"], beta_model_vars, 3)
    phase1b(parameters["slots_medium_sample"], beta_model_vars, 2)
    phase1b(parameters["slots_low_sample"], beta_model_vars, 1)

    return phase1c(beta_model_vars)

def predict_and_ret_best_slots(beta_models):
    slots_to_pull = []
    data["expectations"] = []
    diagnostics = []
    for i in range(0, len(data["slots"])):

        p1 = int(data["slots"][i][0][0][0])
        p2 = int(data["slots"][i][0][0][1])
        p3 = int(data["slots"][i][0][0][2])
        p4 = int(data["slots"][i][0][0][3])
        p5 = int(data["slots"][i][0][0][4])
        p6 = int(data["slots"][i][0][0][5])
        p7 = int(data["slots"][i][0][0][6])
        p8 = int(data["slots"][i][0][0][7])

        alpha_expected = beta_models['alpha'].predict([[p1, p2, p3, p4, p5, p6, p7, p8]])
        beta_expected = beta_models['beta'].predict([[p1, p2, p3, p4, p5, p6, p7, p8]])
        loc_expected = beta_models['loc'].predict([[p1, p2, p3, p4, p5, p6, p7, p8]])
        scale_expected = beta_models['scale'].predict([[p1, p2, p3, p4, p5, p6, p7, p8]])
        bm_expected = scipy.stats.beta.mean(alpha_expected[0], beta_expected[0], loc_expected[0], scale_expected[0])
        mean_expected = np.mean(data["slots"][i][2][0])
        cost = data["slots"][i][1][0]
        gain_expected = 0.3 * bm_expected + 0.7 * mean_expected - cost
        diagnostics.append(["Slot: " + str(i), "Mean expected: " + str(mean_expected), "Beta Model Expected: " + str(bm_expected), "Cost: " + str(cost), "Gain expected: " + str(gain_expected)])

        # weight actual means higher than averages
        # model is included because it may provide insight into high impact low probability payout values
        slots_to_pull.append([i, gain_expected])
        data["expectations"].append([i, gain_expected])

    for x in diagnostics:
        print(x)

    ranked_slots = sorted(slots_to_pull, key=lambda x: x[1], reverse=True)
    return ranked_slots


def rank_slots():
    beta_models = build_models()
    data["models"] = beta_models
    return predict_and_ret_best_slots(beta_models)

def phase2a(state):
    best = data["best"]
    public_bids = [ [best[99-i][0] for i in range(10)]]
    data["public-bids"] = public_bids
    return{
    "team-code": state["team-code"],
    "game": "phase_2_a",
    "auctions": public_bids
    }

def phase2b(state):
    if data["secret-bids"] is None:
        popular_slots = []
        for i in range(100):
            popular_slots.append(i, len(state["auction-lists"][i]),)
        sorted(popular_slots, key=lambda x: x[1])
        pop_slots = []
        for i in popular_slots:
            pop_slots.append(i[0])
        best_slots = data["best"]
        b_slots = []
        for i in best_slots:
            b_slots.append(i[0])
        slots_perf_per_pop = []
        for i in range(100):
            value = b_slots.index(i)
            popularity = pop_slots.index(i)
            slots_perf_per_pop.append(i, value-popularity)
        sorted(slots_val_per_pop, key=lambda x: x[1], reverse = True) # higher is better
        public_bids = data["public-bids"]

        secret_bids = [slots_perf_per_pop[x][0] for x in range(100) if slots_perf_per_pop[x][0] not in public_bids]

        data["secret-bids"] = [ secret_bids[x] for x in range(5) ]


    if state["auction-number"] in data["public-bids"] or state["auction-number"] in data["secret-bids"]:
        slot_to_bid = state["auction-number"]
        bid = data["expectations"][slot_to_bid][1]*10000.0

        return {
        "team-code": state["team-code"],
        "game": "phase_2_b",
        "bid": bid,
        }

    return {
        "team-code": state["team-code"],
        "game": "phase_2_b",
        "bid": 0,
        }
def get_move(state):

    if state["game"] == "phase_2_a":
        return  phase2a(state)
    if state["game"] == "phase_2_b":
        return phase2b(state)

    data["turn"] = data["turn"] + 1
    n = data["last_slot"]

    if data["turn"] == 0:
        return set_last_slot_and_ret_slot(parameters["slots_high_sample"][0])

    # store last metadata string
    data["slots"][n][0].append(state["last-metadata"])

    # store last cost
    data["slots"][n][1].append(state["last-cost"])

    # store last payoff
    data["slots"][n][2].append(state["last-payoff"])

    # pull each high sample slots n times
    if data["turn"] < switches["high_switch"]:
        i = data["turn"] % len(parameters["slots_high_sample"])
        return set_last_slot_and_ret_slot(parameters["slots_high_sample"][i])

    if data["turn"] < switches["medium_switch"]:
        i = (data["turn"] - switches["high_switch"]) % len(parameters["slots_medium_sample"])
        return set_last_slot_and_ret_slot(parameters["slots_medium_sample"][i])

    if data["turn"] < switches["low_switch"]:
        i = (data["turn"] - switches["medium_switch"]) % len(parameters["slots_low_sample"])
        return set_last_slot_and_ret_slot(parameters["slots_low_sample"][i])

    i = data["turn"] - switches["low_switch"]
    if i % switches["best_n_switch"] == 0:
        data["best"] = rank_slots()

    if i == 200:
        parameters["best_n_to_pull"] = 10
        switches["best_n_switch"] = parameters["best_n_to_pull_trials"] * parameters["best_n_to_pull"]

    if i == 300:
        parameters["best_n_to_pull"] = 7
        switches["best_n_switch"] = parameters["best_n_to_pull_trials"] * parameters["best_n_to_pull"]

    if i == 400:
        parameters["best_n_to_pull"] = 5
        switches["best_n_switch"] = parameters["best_n_to_pull_trials"] * parameters["best_n_to_pull"]

    if i == 500:
        parameters["best_n_to_pull"] = 3
        switches["best_n_switch"] = parameters["best_n_to_pull_trials"] * parameters["best_n_to_pull"]

    return set_last_slot_and_ret_slot(data["best"][i % parameters["best_n_to_pull"]][0])






