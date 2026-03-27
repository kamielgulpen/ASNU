import pickle, sys
sys.path.insert(0, 'c:/Users/KGulp/PhD/ASNU')
import matplotlib.pyplot as plt
from modeling.experiments.contagion_experiment import run_experiment, plot_results
import time




with open('a.pkl', 'rb') as f:
    G = pickle.load(f)

networks = {'a': G}
t0 = time.time()
results = run_experiment(networks, n_simulations=10)

t1 = time.time()

total = t1-t0

print(total)
fig = plot_results(results, networks)
plt.show()

