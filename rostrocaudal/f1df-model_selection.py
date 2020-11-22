import pandas as pd
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import functools as ft
from os.path import join

from OdeSolver import DetOdeSolver
from odes_costfunction import ode_neuralisation, cost_neuralisation, get_binary_array

# Load Data
data = pd.read_csv('gene_expression_data.csv')
data.rename({'Unnamed: 0': 'ct concentration'}, axis=1, inplace=True)
ct = data['ct concentration'].to_numpy()
data.drop(['ct concentration'], axis=1, inplace=True)

# Scale Data
data = data.apply(lambda x: x/x.max(), axis=0)

# Select representative genes
data = data[['FoxG1', 'En1', 'HoxA2']].to_numpy()

# Load initial conditions, initial parameter guess and optimisation parameters
a = np.load('initial_guess_parameters.npy')  # initial parameter guess
x0 = [1, 1, 1, 1, 1]  # initialisation of the ode system
boundaries = np.load('optimisation_boundaries.npy')  # boundaries of the optimiser
h = 2  # time step
steps = int(3E2)  # number of time steps for which the odes are solved
variation = get_binary_array(12)  # array that contains all permutations of 0 and one for a 12 entry list
                                  # this is used to iterate through the different topologies of the model
                                  # setting the `on` parameter of the ode system.

# Optimisation of all topologies
skip_optimisation = True  # set True if you want to skip the (very long) optimisation and
                          # use the results of a previous run.
if not skip_optimisation:
    for i in tqdm.trange(variation.shape[0]):
        a = np.load('initial_guess_parameters.npy')
        ode_vartop = ft.partial(ode_neuralisation, on=variation[i, :])
        result = minimize(cost_neuralisation, x0=a, method='L-BFGS-B',
                          args=(ode_vartop, x0, h, steps, data),
                          bounds=boundaries,
                          options={'maxiter':20})
        np.save(join('topology_results', str(i)+'_param.npy'), result.x)
        np.save(join('topology_results', str(i)+'_cost.npy'), result.fun)

# Plot the result of the optimisation (Fig 1 D)
cost = []
for i in tqdm.trange(4096):
    cost.append(np.load(join('topology_results', str(i) + '_cost.npy')))
cost = np.array(cost)
plt.rcParams.update({'font.size': 15})
plt.figure()
plt.grid()
plt.plot(cost, '.', zorder=1)
plt.gca().set_prop_cycle(None)
plt.plot([322, 2370], cost[[322, 2370]], '.', zorder=4)
plt.plot([322, 2370], cost[[322, 2370]], 'o', ms=10, zorder=2)
plt.plot([322, 2370], cost[[322, 2370]], 'wo', ms=6, zorder=3)
plt.ylim([0, 5.5])
plt.text(322+80, cost[322], '#322', horizontalalignment='left',
         verticalalignment='center')
plt.text(2370+80, cost[2370], '#2370', horizontalalignment='left',
         verticalalignment='center')
plt.xlabel('model topology #')
plt.ylabel('minimised costfunction $E$')
plt.tight_layout()
plt.show()

# Solve the ode system for the winning topology
number = 322  # index of the winning topology
data = np.concatenate([ct[:, None], data], axis=1)
a = np.load(join('topology_results', str(number) + '_param.npy'))
ode_vartop = ft.partial(ode_neuralisation, on=variation[number,:], final_run=True)
ode_result = []
for i in range(np.shape(data)[0]):
    x0[-1] = data[i, 0]
    solver = DetOdeSolver(ode_vartop, x0=x0, a=a, h=h, steps=steps)
    solver.runge_kutta_4(disable_statusbar=True)
    ode_result.append(solver.x[-1, :3])
ode_result = np.array(ode_result).T

# Plot the optimised model and the data as bar graph (Fig 1 F)
l = ['FB', 'MB', 'HB']
plt.rcParams.update({'font.size': 15})
plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
edges = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
plt.grid(zorder=0)
for i in range(3):
    plt.bar(np.array(range(len(data.T[0,:])))+.0625+i*.3125, data.T[i+1,:],
            width=.125, label=l[i], color = colors[i], edgecolor=edges,
            align='edge', zorder=3)
    plt.bar(np.array(range(len(data.T[0,:])))+.1875+i*.3125, ode_result[i,:],
            width=.125, color = colors[i], edgecolor=edges,
            align='edge', hatch="//", zorder=3)
plt.xticks(np.array(range(len(data.T[0,:])))+.5, data.T[0,:])
plt.legend(bbox_to_anchor=(.30, .65),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('CT concentration [µM]')
plt.ylabel('gene expression [norm. to max.]')
plt.show()


