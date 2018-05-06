from scipy.stats import beta
data= 
{
"slots": [ [ [], [], [] ] for x in range(100) ],
"last-slot":None,
"ct":0,
"our_utility":1000000,
"end-explore":0
}




"""
Go through first 20 slot machines 30 pulls each to generate data 

Machine 11 - string, cost, payoff
data["slots"][11][0][0] = string
data["slots"][11][1][0] = cost
data["slots"][11][2][0] = payoff
"""



state = 
{"team-code": "eef8976e",
"game": "phase_1",
"pulls-left": 99999,
"last-cost": 0.75,
"last-payoff": 20.7,
"last-metadata": "00110101"}


def phase1(state):

last_slot = data["last-slot"]

if data["last-slot"] is None:
	data["last-slot"] = 0
	return {
	"team-code": state["team-code"],
	"game": "phase_1",
	"pull": 0,
	}

if len(data["slots"][last_slot][0]) == 0:
	data["slots"][last_slot][0].append(state["last-metadata"])

if len(data["slots"][last_slot][1]) == 0:
	data["slots"][last_slot][1].append(state["last-cost"])

data["slots"][last_slot][2].append(state["last-payoff"])

if last_slot < 20:
	if data["ctr"] < 30:
		data["ctr"]+=1
	else:
		last_slot += 1
		data["last-slot"] = last_slot
		data["ctr"] = 0
	ret = {
		"team-code": state["team-code"],
		"game": "phase_1",
		"pull": last_slot,
		}
elif last_slot == 99:
	data["end-explore"] = 1
	for i in range(20):
		u = data["slots"][i][1][0]
		x = data["slots"][i][2]
 		a,b,c,d = beta.fit(x) #0 for min? 
 		m = beta.mean(a,b,c,d)
 		v = beta.var(a,b,c,d)
 		if m - u > v**.5:
 			data["slots"][i][0].append(1) #profitable
 		else:
 			data["slots"][i][0].append(0) #unprofitable
 		run_regression()
 		identify()

 		ret = {
		"team-code": state["team-code"],
		"game": "phase_1",
		"pull": #profitable as identified
		}

else:
	last_slot += 1
	data["last-slot"] = last_slot
	if last_slot < 100:
		ret = {
		"team-code": state["team-code"],
		"game": "phase_1",
		"pull": last_slot,
		}

return ret


run_regression():
#build logistic model for first 20 slot machines

identify():
#assign profitable/unprofitable to other 80 machines based on model

#then pull the 5%??? most profitable
#also consider track utility? if utility drops below threshold then break-out?


