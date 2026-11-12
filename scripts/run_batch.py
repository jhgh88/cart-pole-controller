import numpy as np
import pandas as pd
from cartpole.simulate import run_simulation
from cartpole.animate import run_animation

from pathlib import Path

results_dir = Path(__file__).resolve().parent.parent / "results" # Directory for output csv file
results_dir.mkdir(exist_ok=True)

# Initial conditions
v = 0
a = 0
omega = 0

#----------------------- Main code --------------------------

failed = []
num_of_sims = 100000
sim_data = np.empty(num_of_sims, dtype=[('x', float), ('theta', float), ('success', bool), ('settle_time', float)])
last_reset_states = np.empty(num_of_sims, dtype=[('success', bool), ('x', float), ('v', float), ('theta', float), ('omega', float), ('energy', float)])

for i in range(num_of_sims):

    x = np.random.rand() * 0.6 - 0.3
    theta = np.random.rand() * 2*np.pi - np.pi
    mode = "reset"
    last_reset_index = 0
    last_reset_energy = 0

    success, settle_time, history, mode_history, settled_history = run_simulation(
        x=x,
        v=v, 
        a=a, 
        theta=theta, 
        omega=omega, 
        mode=mode
    )
    
    if settle_time == -1:
        print(f"Run {i}: Failed. x={x:.3f}, theta={np.rad2deg(theta):.1f}deg")
        failed.append(i)
    else:
        print(f"Run {i}: Succeeded.")
    sim_data[i] = (x, np.rad2deg(theta), success, settle_time)

    # Extract state values immediately before swingup
    last_reset_index = int(np.argmax(mode_history == "swingup") - 1)
    if last_reset_index >= 0:
        last_reset_x = history[last_reset_index, 0]
        last_reset_v = history[last_reset_index, 1]
        last_reset_theta = history[last_reset_index, 2]
        last_reset_omega = history[last_reset_index, 3]
        last_reset_energy = history[last_reset_index, 5]
    else:
        last_reset_x, last_reset_v, last_reset_theta, last_reset_omega, last_reset_energy = 1, 0, 0, 0, 0
    last_reset_states[i] = (success, last_reset_x, last_reset_v, last_reset_theta, last_reset_omega, last_reset_energy)

success_rate = sum(1 for run in sim_data if run['success']==True) / len(sim_data)
print(f"Succeeded: {len(sim_data) - len(failed)}, Failed: {len(failed)}")
print(f"Success rate: {success_rate*100:.2f}%")

sim_dataframe = pd.DataFrame(sim_data)
sim_dataframe.to_csv(results_dir/"batch_results.csv", index=False) # Export data to results/batch_results.csv
last_reset_dataframe = pd.DataFrame(last_reset_states)
last_reset_dataframe.to_csv(results_dir/"last_reset_states.csv", index=False)