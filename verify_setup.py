import sys
print("Python:", sys.version)

import numpy
import pandas
import matplotlib
import gymnasium
import ipywidgets
import jupyterlab

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("gymnasium:", gymnasium.__version__)
print("ipywidgets:", ipywidgets.__version__)
print("jupyterlab:", jupyterlab.__version__)

from eldermist import ElderMistReplicaEnv
env = ElderMistReplicaEnv()
obs, info = env.reset(seed=1)

print("\nElder Mist environment works.")
print("Observation shape:", obs.shape)
print("Valid actions:", info["valid_actions"])

obs, reward, terminated, truncated, info = env.step(0)
print("One env.step(0) worked.")
print("Reward:", reward)
print("Event:", info["event"])
print("\nSETUP OK")
