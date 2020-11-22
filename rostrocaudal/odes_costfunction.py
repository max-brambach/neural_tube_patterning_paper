import numpy as np

from OdeSolver import DetOdeSolver

def ode_neuralisation(x, a, on, final_run=False):
    """
    Differential equation set for the rostrocaudal model.

    Used for model topology selection. Given the levels of fb, mb, hb, gsk3, ct at t=i this function returns the
    corrseponding level changes.
    :param x: list, contains the fb, mb, hb, gsk3 and ct levels at t=i
    :param a: list, contains the parameters c_i (idx 0-13), delta_i (idx 14-17), n_i(18-31)
    :param on: list, used to switch interactions from activation (entry is 1) to repression (entry is 0)
    :param final_run: bool, if True, rate constant c[0] (fb self-interaction) is fixed to 0. This corresponds to the
                            final network topology.
    :return: list, contains the fb, mb, hb, gsk3 and ct level changes
    """
    fb, mb, hb, gsk3, ct = x
    c = a[:14]
    delta = a[14:18]
    n = a[18:32]
    if final_run:
        c[0] = 0
    return [(on[0]*c[0]*fb**n[0] + on[3]*c[1]*gsk3**n[1] + on[1]*c[2]*mb**n[2] + on[2]*c[3]*hb**n[3]) /
            (1 + c[0]*fb**n[0] + c[1]*gsk3**n[1] + c[2]*mb**n[2] + c[3]*hb**n[3])
            - delta[0]*fb,
            (on[5]*c[4]*mb**n[4] + on[4]*c[5]*fb**n[5] + on[6]*c[6]*hb**n[6] + on[7]*c[7]*gsk3**n[7]) /
            (1 + c[4]*mb**n[4] + c[5]*fb**n[5] + c[6]*hb**n[6] + c[7]*gsk3**n[7])
            - delta[1]*mb,
            (on[10]*c[8]*hb**n[8] + on[8]*c[9]*fb**n[9] + on[9]*c[10]*mb**n[10] + on[11]*c[11]*gsk3**n[11]) /
            (1 + c[8]*hb**n[8] + c[9]*fb**n[9] + c[10]*mb**n[10] + c[11]*gsk3**n[11])
            - delta[2]*hb,
            (c[12]*gsk3**n[12])/(1+c[12]*gsk3**n[12]+c[13]*ct**n[13])-delta[3]*gsk3,
            0]


def ode_neuralisation_final(x, c, n, delta, on):
    """
    Version of the ODE system after topology selection.

    Main difference is that now c, n, delta are given individually.
    corrseponding level changes.
    :param x: list, contains the fb, mb, hb, gsk3 and ct levels at t=i
    :param c, n, delta: list, contains the parameters c_i, delta_i, n_i
    :param on: list, used to switch interactions from activation (entry is 1) to repression (entry is 0)
    :param final_run: bool, if True, rate constant c[0] (fb self-interaction) is fixed to 0. This corresponds to the
                            final network topology.
    :return: list, contains the fb, mb, hb, gsk3 and ct level changes
    :return:
    """
    fb, mb, hb, gsk3, ct = x
    return [(on[3]*c[1]*gsk3**n[1] + on[1]*c[2]*mb**n[2] + on[2]*c[3]*hb**n[3]) /
            (1 + c[1]*gsk3**n[1] + c[2]*mb**n[2] + c[3]*hb**n[3])
            - delta[0]*fb,
            (on[5]*c[4]*mb**n[4] + on[4]*c[5]*fb**n[5]) /
            (1 + c[4]*mb**n[4] + c[5]*fb**n[5] + c[6]*hb**n[6] + c[7]*gsk3**n[7])
            - delta[1]*mb,
            (on[7]*c[8]*hb**n[8] + on[6]*c[9]*fb**n[9] + on[8]*c[11]*gsk3**n[11]) /
            (1 + c[8]*hb**n[8] + c[9]*fb**n[9] + c[10]*mb**n[10] + c[11]*gsk3**n[11])
            - delta[2]*hb,
            (c[12]*gsk3**n[12])/(1+c[12]*gsk3**n[12]+c[13]*ct**n[13])-delta[3]*gsk3,
            0]


def cost_neuralisation(a, *args):
    """
    Sum of squares cost function for the rostrocaudal model.

    Used for model topology selection. Returns the summed square difference between the model and the data.
    :param a: list, contains the parameters c_i (idx 0-13), delta_i (idx 14-17), n_i(18-31)
    :param args: list, contains [ode: differential equation,
                                 x0: initial conditions (state at t=0),
                                 h: time step,
                                 steps: number of iterations]
    :return: float, cost function value
    """
    ode, x0, h, steps, data = args
    print(data)
    cost = 0
    for i in range(np.shape(data)[0]):
        x0[-1] = data[i, 0]
        solver = DetOdeSolver(ode, x0=x0, a=a, h=h, steps=steps)
        solver.runge_kutta_4(disable_statusbar=True)
        ode_result = solver.x[-1,:3]
        cost += (np.sum( (ode_result[0:3] - data[i, :])**2.))
    return cost

def get_binary_array(n):
    """
    Generate an array of all n**2 permutations of 0 and one for a sequence of length n.
    :param n: int, length of sequence
    :return: array, two dimensional, axis=0: permutations, axis=1: sequence
    """
    bin_list = []
    for i in range(2**n):
        b = bin(i)[2:]
        l = len(b)
        b = str(0) * (n - l) + b
        bin_list.append(list(b))
    return np.array(bin_list, dtype="int")
