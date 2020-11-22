import OdeSolver as ode
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import tqdm

def ode_neuralisation(x, a):
    """
    Set of ODEs of the rostrocaudal model.
    :param x: list, contains the fb, mb, hb, gsk3 and ct levels at t=i
    :param a: list, contains the parameters c_i (idx 0-13), delta_i (idx 14-17), n_i(18-31)
                    ADDITION: a[32:] are `on` parameters, that are used to set specific ODEs to 0 (--> knockdown)
    :return: list, contains the fb, mb, hb, gsk3 and ct level changes
    """
    fb, mb, hb, gsk3, ct = x
    c = a[:14]
    delta = a[14:18]
    n = a[18:32]
    on = a[32:]
    return [((c[1]*gsk3**n[1]) / (1 +
            c[1]*gsk3**n[1] + c[2]*mb**n[2] + c[3]*hb**n[3]) - delta[0]*fb) * on[0],
            ((c[4]*mb**n[4]) / (1 + c[4]*mb**n[4] + c[5]*fb**n[5] +
            c[6]*hb**n[6] + c[7]*gsk3**n[7]) - delta[1]*mb)  * on[1],
            ((c[8]*hb**n[8]) / (1 + c[8]*hb**n[8] + c[9]*fb**n[9] +
            c[10]*mb**n[10] + c[11]*gsk3**n[11]) - delta[2]*hb)  * on[2],
            (c[12]*gsk3**n[12])/(1+c[12]*gsk3**n[12]+c[13]*ct**n[13])-delta[3]*gsk3,
            0]

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

# Set up remaining parameters
h = 2
steps = int(5E3)

# Solve odes for kd/oex cases

def generate_oex_kd_combinations():
    """
    Generates the kd/oex initial conditions / parameter values.
    :return: kd_oex: array, kd_oex[[0, 2, 4]]: initial conditions for [fb, mb, hb]; 1:oex, 0:normal
                            kd_oex[[1, 3, 5]]: switches to turn specific ODEs off (kd); 1:normal, 0:kd
    """
    wt = np.array([1, 1, 1, 1, 1, 1])
    fb_oex = np.array([1, 0, 1, 1, 1, 1])
    fb_kd = np.array([0, 0, 1, 1, 1, 1])
    mb_oex = np.array([1, 1, 1, 0, 1, 1])
    mb_kd = np.array([1, 1, 0, 0, 1, 1])
    hb_oex = np.array([1, 1, 1, 1, 1, 0])
    hb_kd = np.array([1, 1, 1, 1, 0, 0])
    kd_oex = [wt,
              fb_oex,
              fb_kd,
              mb_oex,
              mb_kd,
              hb_oex,
              hb_kd,
              ]
    labels = ['wt',
              'fb_oex',
              'fb_kd',
              'mb_oex',
              'mb_kd',
              'hb_oex',
              'hb_kd',
              ]
    numbers = np.arange(0, 6)
    permutations = list(np.array(np.meshgrid(numbers,numbers)).T.reshape(-1, 2))
    correct_permutations = []
    for perm in permutations:
        if perm[0] == perm[1]:
            continue
        if perm[0] < perm[1] and perm[0] == perm[1]-1 and perm[0] % 2 == 0:
            continue
        if perm[0] > perm[1] and perm[0]-1 == perm[1] and perm[0] % 2 != 0:
            continue
        correct_permutations.append(perm)
    for perm in correct_permutations:
        kd_oex.append(kd_oex[perm[0] + 1] * kd_oex[perm[1] + 1])
        labels.append(labels[perm[0] + 1] + '-' + labels[perm[1] + 1])
    return kd_oex, labels
kd_oex, kd_oex_labels = generate_oex_kd_combinations()

# calculate the kd/oex cases
skip_kdoex_calculations = True  # skip the (rather lengthy) kd/oex calculations and use results from previous run
if not skip_kdoex_calculations:
    for i in tqdm.trange(len(kd_oex)):
        kd_oex_current = np.array(kd_oex[i])
        x0 = [1, 1, 1, 1, 1]
        x0[:-2] = kd_oex_current[[0, 2, 4]]
        number = 322
        a = np.load(os.path.join('topology_results', str(number) + '_param.npy'))
        a = np.concatenate((a, [1, 1, 1]))
        a[-3:] = kd_oex_current[[1, 3, 5]]
        name = kd_oex_labels[i]
        ode_result = []
        all_results = []
        gsk3 = []
        ct_concentrations = np.linspace(0, 2, num=200)
        for ct in ct_concentrations:
            x0[-1] = ct
            solver = ode.DetOdeSolver(ode_neuralisation, x0=x0, a=a, h=h, steps=steps)
            solver.runge_kutta_4(disable_statusbar=True)
            ode_result.append(solver.x[-1, :3])
            gsk3.append(solver.x[-1, 3])
            all_results.append(solver.x)
        ode_result = np.array(ode_result).T
        all_results = np.array(all_results, dtype='float64')
        np.savetxt('oex_kd_results/ode_result_' + name + '.txt', ode_result, delimiter=',')
        np.save('oex_kd_results/time_series_' + name, all_results)
        np.save('oex_kd_results/gsk3_levels_' + name, gsk3)

# Plot the kd/oex results
def count_through_grid(count, gridwidth):
    """
    Generate row/column indices for a number of intergers.
    :param count: int, how many elements has the grid
    :param gridwidth: int, how wide is the grid
    :return:
    """
    row = int(count / gridwidth)
    column = count - row * gridwidth
    return row, column

# Load kd/oex results and format the plot names based on the file names
files = glob.glob('oex_kd_results/*.txt')
names = []
for file in files:
    names.append(file.split('\\')[-1].split('.')[0].replace('ode_result_', '')
          .replace('_', ' ').replace('-', ' + ').replace('fb', 'FB').replace('mb', 'MB')
          .replace('hb', 'HB'))
ct_concentrations = np.linspace(0, 2, num=200)

# Plot kd/oex results (single files)
for i in range(len(files)):
    fig, ax = plt.subplots()
    ode_result = np.loadtxt(files[i], delimiter=',')
    ax.set_title(names[i], fontsize=26)
    for j in range(3):
        ax.plot(ct_concentrations, ode_result[j, :],
                linewidth=10)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.show()
    plt.close()