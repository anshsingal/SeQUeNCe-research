from absorptive_experiment_misc.definitions import RamanCharacterizationNode
import numpy as np
from src.kernel.event import Event
from src.kernel.process import Process
from src.kernel.timeline import Timeline
from json import dump

SPDC_FREQUENCY = 8e8
MODE_NUM = 1000

# Previously used set of parameters
# params = {
#     # detectors
#     "BSM_DET1_EFFICIENCY" : 0.06, # efficiency of detector 1 of BSM
#     "DARK_COUNTS" : 0,
#     "RESOLUTION": 10,
#     "TEMPORAL_COINCIDENCE_WINDOW": 450,
#     "DEAD_TIME": 6000,

#     # Quantum channel
#     "QUANTUM_WAVELENGTH" : 1536.,
#     "QUANTUM_INDEX" : 1.470,

#     # fibers
#     "DISTANCE" : 25.,  # distance between ANL and ERC, in km
#     "CLASSICAL_ATTENUATION" : [0.5, 0.5],
#     "RAMAN_COEFFICIENTS" : [33.e-10, 33.e-10], # Correct this!!!
#     "DELAY_CLASSICAL" : 5e-3,  # delay for classical communication between BSM node and memory nodes (in s)
#     "CLASSICAL_INDEX" : [1.47, 1.47],
#     "QUNATUM_ATTENUATION" : 0.44,

#     # classical communication 
#     "CLASSICAL_WAVELENGTH" : 1610,
#     "CLASSICAL_RATE" : 1e10,
#     "COMMS_WINDOW_SIZE" : 1e-3*1e12, # Amount of time it takes to complete one batch of MODE_NUM quantum emissions. 
#     "TOTAL_COMM_SIZE": 0.1*1e12, # This is the amount of time that we calculate Raman scatting for.
#     "AVG_POWER": [1,2,3,4,5,6,7,8,9,10,11,12], # Here, you input a list of average powers (dBm, check this) that you want to iterate over. When the results are printed out, they'll contain only the avg_power that was used in that iteration. 
#     # "OMA": 2, # Set later on in the program as a function of the AVG_POWER
#     "DIRECTION": True, # True is co-propagation
#     "NBF_BANDWIDTH" : 0.03,
#     "MODULATION": "PSK",

#     # experiment settings
#     "TIME" : int(1e12),
# }

file_name = "results/ramanCoProp"

h = 6.62607015 * 10**(-34)
c = 3*10**8
params = {
    # detectors
    "BSM_DET1_EFFICIENCY" : 0.172, # efficiency of detector 1 of BSM
    "DARK_COUNTS" : 0,
    "RESOLUTION": 10,
    "TEMPORAL_COINCIDENCE_WINDOW": 450,
    "DEAD_TIME": 6000,

    # Quantum channel
    "QUANTUM_WAVELENGTH" : 1547.72,
    "QUANTUM_INDEX" : 1.470,

    # fibers
    "DISTANCE" : 25.,  # distance between ANL and ERC, in km
    "CLASSICAL_ATTENUATION" : [0.32*np.log(10)/10, 0.32*np.log(10)/10], # Converting dB/km to 1/km units. 
    "RAMAN_COEFFICIENTS" : [7.71e-10, 7.71e-10], # Correct this!!!
    "DELAY_CLASSICAL" : 5e-3,  # delay for classical communication between BSM node and memory nodes (in s)
    "CLASSICAL_INDEX" : [1.47, 1.47],
    "QUNATUM_ATTENUATION" : 0.17*np.log(10)/10,

    # classical communication 
    "CLASSICAL_WAVELENGTH" : 1310.,
    "CLASSICAL_RATE" : 1e10,
    "COMMS_WINDOW_SIZE" : 1e-3*1e12, # Amount of time it takes to complete one batch of MODE_NUM quantum emissions. 
    "TOTAL_COMM_SIZE": 0.1*1e12, # This is the amount of time that we calculate Raman scatting for.
    "AVG_POWER": list(np.logspace(13, 14.765, 12) * h*c/(1310.*10**(-9))), # Here, you input a list of average powers (dBm, check this) that you want to iterate over. When the results are printed out, they'll contain only the avg_power that was used in that iteration. 
    # "OMA": 2, # Set later on in the program as a function of the AVG_POWER
    "DIRECTION": True, # True is co-propagation
    "NBF_BANDWIDTH" : 0.57,
    "MODULATION": "PSK",

    # experiment settings
    "TIME" : int(1e12),
}

timeline = Timeline(params["TIME"])

raman_file = open(file_name+".csv", "a+")
raman_file.write("power,total_raman_counts\n")

timeline.init()

node = RamanCharacterizationNode("RamanCharacterizationNode0", timeline, params)

for power in params["AVG_POWER"]: 
    # power = 10**( power_dbm /10)/1000 # Uncomment if powers are i  dBm
    OMA = 0.4*power
    params["CLASSICAL_POWERS"] = [[power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2], [power-OMA/2, power-OMA/6, power+OMA/6, power+OMA/2]]

    process = Process(node, "start", [])
    event = Event(0, process)
    timeline.schedule(event)

    timeline.run()

    raman_file.write(f"{power},{(1e12/params['TOTAL_COMM_SIZE'])*node.detections}\n")
    node.detections = 0
    raman_file.flush()
    timeline.time = 0
    timeline.init()
params_filename = file_name+"_params.json"
file_pointer = open(params_filename, 'w')
dump(params, file_pointer)
    # print(power_dbm, ":", len(node.detections))

