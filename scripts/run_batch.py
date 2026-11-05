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
num_of_sims = 5000
sim_data = np.empty(num_of_sims, dtype=[('x', float), ('theta', float), ('success', bool), ('settle_time', float)])

for i in range(num_of_sims):

    x = np.random.rand() * 0.6 - 0.3
    theta = np.random.rand() * 2*np.pi
    mode = "reset"

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

success_rate = sum(1 for run in sim_data if run['success']==True) / len(sim_data)
print(f"Succeeded: {len(sim_data) - len(failed)}, Failed: {len(failed)}")
print(f"Success rate: {success_rate*100:.1f}%")

sim_dataframe = pd.DataFrame(sim_data)
sim_dataframe.to_csv(results_dir/"batch_results.csv", index=False) # Export data to results/batch_results.csv