import hw5
import numpy as np
from itertools import product
from random import randint


# two bit to integer + .5
def bit_to_int(tbn):
    return 2*int(tbn[0])+int(tbn[1])+.5


def get_rand_beta_var(n, slot_machines):
    return slot_machines["scale"][n]*np.random.beta(slot_machines["alpha"][n],slot_machines["beta"][n])+slot_machines["loc"][n]


if __name__ == "__main__":
    slot_machines = {
        "alpha": [],
        "beta": [],
        "loc": [],
        "scale": [],
        "cost": [],
        "ebn": []
    }
    for ebn in product('01', repeat=8):
        slot_machines["ebn"].append(''.join(ebn))

    slot_machines["ebn"] = slot_machines["ebn"][:100]
    for i in slot_machines["ebn"]:
        slot_machines["alpha"].append(bit_to_int(i[0:2]))
        slot_machines["beta"].append(bit_to_int(i[2:4]))
        slot_machines["loc"].append(bit_to_int(i[4:6]))
        slot_machines["scale"].append(bit_to_int(i[6:8]))
        slot_machines["cost"].append(randint(0, 9))

    state = {"team-code": "eef8976e",
             "game": "phase_1",
             "pulls-left": 10000,
             "last-cost": 0,
             "last-payoff": 0,
             "last-metadata": "00000001"}

    move = hw5.get_move(state)
    prev_slot = move["pull"]

    for i in range(9999):

        state = {"team-code": "eef8976e",
            "game": "phase_1",
            "pulls-left": 10000-i,
            "last-cost": slot_machines["cost"][prev_slot],
            "last-payoff": get_rand_beta_var(prev_slot, slot_machines),
            "last-metadata": slot_machines["ebn"][prev_slot]}

        print("Slot: " + str(prev_slot) + " Payoff: " + str(state["last-payoff"]) + " Cost: " + str(slot_machines["cost"][prev_slot]))

        move = hw5.get_move(state)
        prev_slot = move["pull"]