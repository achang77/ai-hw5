from hw5 import *
from itertools import product
import string


def load_data():
    return None


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