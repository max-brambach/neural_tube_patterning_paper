import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from odes_costfunction import ode_neuralisation, ode_neuralisation_final, get_binary_array
import sensitivity_analyser as sa
from os.path import join
import functools as ft
from OdeSolver import DetOdeSolver

def cost_neuralisation(a, ode, x0, h, steps, data):
    # ode, x0, h, steps, data = args
    cost = 0
    for i in range(np.shape(data)[0]):
        x0[-1] = data[i, 0]
        solver = DetOdeSolver(ode, x0=x0, a=a, h=h, steps=steps)
        solver.runge_kutta_4(disable_statusbar=True)
        ode_result = solver.x[-1,:3]
        cost += (np.sum( (ode_result[0:3] - data[i, 1:4])**2.))
    # print(cost)
    return cost

# Load Data
data = pd.read_csv('gene_expression_data.csv')
data.rename({'Unnamed: 0': 'ct concentration'}, axis=1, inplace=True)
ct = data['ct concentration'].to_numpy()
data.drop(['ct concentration'], axis=1, inplace=True)

# Scale Data
data = data.apply(lambda x: x/x.max(), axis=0)

# Select representative genes
data = data[['FoxG1', 'En1', 'HoxA2']].to_numpy()
data = np.concatenate([ct[:, None], data], axis=1)

# Set model parameters
x0 = [1, 1, 1, 1, 1]
h = 2
steps = int(3E2)
variation = get_binary_array(9)
number = 322
a = np.load(join('topology_results', str(number)+'_param.npy'))
c = a[:14]
delta = a[14:18]
n = a[18:32]
sens_var = np.arange(0, .1, .001)  # range of the sensitivity analysis (in percent)

skip_sensitivity = True # Avoid the (very long) calculation of the sensitivity results and load the results from a
                        # previous run.
if not skip_sensitivity:
    # Sensitivity for parameters c_i
    ode_c = ft.partial(ode_neuralisation_final, on=variation[number,:],
                       n=n, delta=delta)
    cost_c = ft.partial(cost_neuralisation, ode=ode_c,
                              x0=x0, h=h, steps=steps, data=data)
    sens_c = sa.SensitivityAnalyser(cost_c, c, variations=sens_var)
    sens_c.go('sensitivity_results/sens_c')

    # Sensitivity for parameters n_i
    ode_n = ft.partial(ode_neuralisation_final, on=variation[number,:],
                       c=c, delta=delta)
    cost_n = ft.partial(cost_neuralisation, ode=ode_n,
                              x0=x0, h=h, steps=steps, data=data)
    sens_n = sa.SensitivityAnalyser(cost_n, n, variations=sens_var)
    sens_n.go('sensitivity_results/sens_n')

    # Sensitivity for parameters delta_i
    ode_d = ft.partial(ode_neuralisation_final, on=variation[number,:],
                       c=c, n=n)
    cost_d = ft.partial(cost_neuralisation, ode=ode_d,
                              x0=x0, h=h, steps=steps, data=data)
    sens_d = sa.SensitivityAnalyser(cost_d, delta, variations=sens_var)
    sens_d.go('sensitivity_results/sens_d')

    # # robustness
    ode = ft.partial(ode_neuralisation_final, on=variation[number, :])
    cost = ft.partial(cost_neuralisation, ode=ode,
                      x0=x0, h=h, steps=steps, data=data)
    robustness = sa.SensitivityAnalyser(function=cost, parameters=a,
                                        variations=np.arange(.001, .05, .001))
    robustness.robustness()
    robustness.save_robustness('sensitivity_results/rob_data_5perc') # has to be re-run for 1,2,5,10%

# load the sensitivity analysis results
n = np.load('sensitivity_results/sens_n.npy')[:, 1:]
c = np.load('sensitivity_results/sens_c.npy')[:, 1:]
d = np.load('sensitivity_results/sens_d.npy')

rob = [np.load('sensitivity_results/rob_data_1perc.npy'),
       np.load('sensitivity_results/rob_data_2perc.npy'),
       np.load('sensitivity_results/rob_data_5perc.npy'),
       np.load('sensitivity_results/rob_data_10perc.npy')]

# prepare sensitivity analysis data for plot
zero_pattern = n[(len(c)-1)//2, 0]
c = c/c[(len(c)-1)//2, :]-1
n = n/n[(len(n)-1)//2, :]-1
d = d/d[(len(d)-1)//2, :]-1
var = np.arange(.01, .1, .001)
x = np.concatenate([-1*var[::-1], np.array([0]), var])*100
cnd = np.mean(np.concatenate([c,n,d], axis=1), axis=0)
rob = np.array(rob)/zero_pattern
names = []
for i in range(13):
    names.append('$c_{'+str(i+1)+'}$')
for i in range(13):
    names.append('$n_{'+str(i+1)+'}$')
for i in range(4):
    names.append('$\delta_'+str(i+1)+'$')

# Plot the sensitivity diagramm
plt.rcParams.update({'font.size': 20})
fig = plt.figure('sensitivity_analysis_neural_patterning',
           figsize=(16, 7))
grid = plt.GridSpec(3, 9)
ax_d = plt.subplot(grid[2, :2])
ax_c = plt.subplot(grid[0, :2])#, sharex=ax_d)
ax_n = plt.subplot(grid[1, :2])#, sharex=ax_d)
ax_mean = plt.subplot(grid[:, 2:7])
ax_rob = plt.subplot(grid[:, 7:8], sharey=ax_mean)
plt.setp(ax_c.get_xticklabels(), visible=False)
plt.setp(ax_n.get_xticklabels(), visible=False)
plt.sca(ax_mean)
ax_mean.yaxis.tick_right()
plt.grid(zorder=1)
c_off = 100
c_iter = 15
for i in range(np.shape(c)[1]):
    plt.bar(x=i, height=cnd[i], zorder=2, color=plt.cm.Blues(c_off+i*c_iter))
j = 0
for i in range(np.shape(c)[1], np.shape(c)[1]+np.shape(n)[1]):
    plt.bar(x=i, height=cnd[i], zorder=2, color=plt.cm.Reds(c_off+j*c_iter))
    j += 1
j = 0
for i in range(np.shape(c)[1]+np.shape(n)[1], len(cnd)):
    plt.bar(x=i, height=cnd[i], zorder=2, color=plt.cm.Greens(c_off+j*c_iter*3))
    j += 1
plt.xlim([-1., len(cnd)])
plt.xticks(range(len(cnd)), names, rotation='vertical', fontsize=17)
ax_mean.set_yticklabels([])
plt.text(x=2, y=8.2, s='$\Delta p = \pm 10\%$')
plt.sca(ax_c)
xlim = [-10.1, 10.1] # [-50.1, 50.1]
plt.grid()
lw = 3
for i in range(np.shape(c)[1]):
    plt.plot(x, c[:, i], linewidth=lw, color=plt.cm.Blues(c_off+i*c_iter))
plt.text(.48, .8, '$c_i$', transform=ax_c.transAxes,
         horizontalalignment='right')
plt.ylim([-.1, 10.1])
plt.xlim(xlim)
plt.yticks([0, 5, 10])
plt.sca(ax_n)
plt.grid()
for i in range(np.shape(n)[1]):
    plt.plot(x, n[:, i], linewidth=lw, color=plt.cm.Reds(c_off+i*c_iter))
plt.text(.48, .8, '$n_i$', transform=ax_n.transAxes,
         horizontalalignment='right')
plt.ylim([-.1, 4.1])
plt.xlim(xlim)
plt.yticks([0, 2, 4])
plt.ylabel('costfunction change $\Delta E_d(\Delta p)$\n')
plt.sca(ax_d)
plt.grid()
for i in range(np.shape(d)[1]):
    plt.plot(x, d[:, i], linewidth=lw, color=plt.cm.Greens(c_off+i*c_iter*3))
plt.text(.48, .8, '$\delta_i$', transform=ax_d.transAxes,
         horizontalalignment='right')
plt.xlim(xlim)
plt.ylim([-.1, 20.1])
plt.yticks([0, 10, 20])
plt.xlabel('parameter change $\Delta p [\%]$')
plt.sca(ax_rob)
ax_rob.yaxis.tick_right()
ax_rob.yaxis.set_label_position("right")
rob_perc = [1, 2, 5, 10]
plt.grid(zorder=1)
plt.ylabel('mean costfunction change $<\Delta E_d(\Delta p)>$')
for i, x in enumerate(rob_perc):
    plt.bar(i, rob[i], color=plt.cm.Greys(c_off+i*c_iter*3), zorder=2)
plt.xticks(range(len(rob_perc)),
           rob_perc,
           fontsize=17)
plt.xlabel('all parameter change $[\%]$')
plt.subplots_adjust(wspace=.05)
plt.show()