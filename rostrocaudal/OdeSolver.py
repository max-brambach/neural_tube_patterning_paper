# -*- coding: utf-8 -*-
"""
@author: Max Brambach
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns


class DetOdeSolver(object):
    """Solve ordinary differential equations numerically.

    Attributes
    ----------
    f : function
        Function that returns the set of differential equations as list.
        E.g. f(x,a) = [a[1]*x[2], -a[2]x[1]] for a set of two coupled
        exponential functions.

    x0 : list or array
        Initial condition of the system i.e. state of the system at time 0.

    a : list
        Independent parameters of the function f. E.g. f(x,a) = x*a

    h : float
        Step size of the solver.

    steps : int
        Number of steps computed by the solver.
    """
    def __init__(self, f, x0, a, h, steps):
        """Builder"""
        self.f = f
        self.x0 = x0
        if np.shape(self.x0) == ():
            self.x0 = np.array([self.x0])
        self.a = a
        self.h = h
        self.steps = steps
        self.num_eq = np.shape(self.x0)[0]
        self.x = np.zeros((self.steps, self.num_eq))
        self.x[0, :] = self.x0

    def euler(self):
        for i in range(1, self.steps):
            derivative = self.f(self.x[i - 1, :], self.a)
            self.x[i, :] = self.x[i - 1, :] + np.array(derivative) * self.h

    def runge_kutta_2(self):
        for i in range(1, self.steps):
            k1 = self.h * np.array(self.f(self.x[i - 1, :], self.a))
            k2 = self.h * np.array(self.f(self.x[i - 1, :] + k1 / 2., self.a))
            self.x[i, :] = self.x[i - 1, :] + k2

    def runge_kutta_4(self, disable_statusbar=True):
        for i in tqdm.trange(1, self.steps, disable=disable_statusbar):
            k1 = (self.h * np.array(self.f(self.x[i - 1, :], self.a))).flatten()
            k2 = (self.h * np.array(self.f(self.x[i - 1, :] + k1 / 2., self.a))).flatten()
            k3 = (self.h * np.array(self.f(self.x[i - 1, :] + k2 / 2., self.a))).flatten()
            k4 = (self.h * np.array(self.f(self.x[i - 1, :] + k3, self.a))).flatten()
            self.x[i, :] = self.x[i - 1, :] + k1/6. + k2/3. + k3/3. + k4/6.


class StochOdeSolver(object):
    """Solve ordinary differential equations stochastically.

    [more detail]
    """
    def __init__(self, rates_update, a, x0, steps, mu):
        """Builder"""
        self.a = a
        self.rates_update = rates_update
        self.x0 = x0
        if np.shape(self.x0) == ():
            self.x0 = np.array([self.x0])
        self.rates = self.rates_update(x0, self.a)
        self.rates_sum = np.sum(self.rates)
        self.rates_cumsum = np.cumsum(self.rates.flatten())
        self.steps = steps
        self.num_eq = np.shape(self.x0)[0]
        self.x = np.zeros((self.steps, self.num_eq))
        self.x[0, :] = self.x0
        self.t = np.zeros(self.steps)
        self.mu = mu

    def __str__(self):
        return "Number of equations: "+str(self.num_eq)

    def name_equations(self, names):
        self.eq_names = names

    def print_stats(self, str=None):
        if type(str) != type(None):
            print(str)
        print("rates: ", self.rates)
        print("rates sum: ", self.rates_sum)
        print("cum sum: ", self.rates_cumsum)

    def update_rate_sums(self):
        self.rates_sum = np.sum(self.rates)
        self.rates_cumsum = np.cumsum(self.rates.flatten())

    def gillespie(self, warnings=False, disable_statusbar=False,
                  external_break=None):
        break_flag = False
        if type(external_break) != type(None):
            break_index, break_value = external_break
            break_flag = True
        rnd = np.random.rand(self.steps, 2)
        for i in tqdm.trange(1, self.steps, disable=disable_statusbar):
            self.t[i] = self.t[i-1] + 1./self.rates_sum*np.log(1./rnd[i, 0])
            self.x[i, :] = self.x[i-1, :]
            try:
                idx = np.min(np.where(self.rates_cumsum > rnd[i, 1] *
                                      self.rates_sum)[0])
            except ValueError:
                self.final_steps = i
                if warnings:
                    print("Premature termaination in step ", self.final_steps)
                self.x = self.x[:i, :]
                self.t = self.t[:i]
                break
            if idx % 2 == 0:
                self.x[i, idx//2] -= self.mu
            else:
                self.x[i, (idx-1)//2] += self.mu
            self.rates = self.rates_update(self.x[i, :], self.a)
            self.update_rate_sums()
            if break_flag:
                if self.x[i, break_index] > break_value:
                    self.x = self.x[:i, :]
                    self.t = self.t[:i]
                    self.final_t = self.t[-1]
                    break
                else:
                    self.final_t = np.nan

    def gillespie_time(self, max_time, start_time=0, warnings=False):
        self.x = [self.x0]
        # print(self.x)
        self.t = [start_time]
        i = 1
        time = start_time
        test = []
        # x_temp = self.x0
        while time < max_time:
            rnd = np.random.rand(2)
            time = self.t[i-1] + 1./self.rates_sum*np.log(1./rnd[0])
            x_temp = self.x[i-1]
            # print('beg: ',x_temp)
            try:
                idx = np.min(np.where(self.rates_cumsum > rnd[1] *
                                      self.rates_sum)[0])
            except ValueError:
                self.final_steps = i
                if warnings:
                    print("Premature termaination in step ", self.final_steps)
                break
            # print('mid: ', x_temp)
            if idx % 2 == 0:
                x_temp[idx//2] = x_temp[idx//2] - self.mu
                # print(idx)
            else:
                x_temp[(idx-1)//2] = x_temp[(idx-1)//2] + self.mu
                # print(idx)
            # print('lst: ', x_temp)
            self.rates = self.rates_update(x_temp, self.a)
            self.update_rate_sums()
            self.x.append(x_temp.copy())
            # print('wrt: ', self.x[-1])
            self.t.append(time.copy())
            i += 1
        self.t = np.array(self.t[:-1])
        self.x = np.array(self.x[:-1])


    def plot(self):
        sns.set_palette(sns.color_palette("hls", self.num_eq))
        plt.figure("PLOT")
        plt.grid()
        if hasattr(self, 'eq_names'):
            for i in range(self.num_eq):
                plt.plot(self.t, self.x[:, i], label=self.eq_names[i])
            plt.legend()
        else:
            for i in range(self.num_eq):
                plt.plot(self.t, self.x[:, i])
        plt.show()


if __name__ == "__main__":
    def exp_ode(x, a):
        return a * x

    def exp_decay_rates(x,a):
        return np.array([a*x, 0])

    gil = StochOdeSolver(rates_update=exp_decay_rates,
                         a=1,
                         x0=1,
                         steps=100,
                         mu=.01)
    gil.gillespie()
    # gil.print_stats()
    ode = DetOdeSolver(f=exp_ode,
                       x0=1,
                       a=-1,
                       h=.1,
                       steps=100)
    plt.figure("Test of OdeSolver Class")
    plt.grid()
    plt.text(x=23, y=.28, s="$\dfrac{dy}{dx} = - y$")
    plt.plot(np.exp(-1.*ode.h*np.arange(ode.steps)),"k",
             label="analytical solution")
    ode.euler()
    plt.scatter(np.arange(len(ode.x)), ode.x, s=10,
                label="Euler")
    ode.runge_kutta_2()
    plt.scatter(np.arange(len(ode.x)), ode.x, s=10,
                label="2nd order Runge-Kutta")
    ode.runge_kutta_4()
    plt.scatter(np.arange(len(ode.x)), ode.x, s=10,
                label="4th order Runge-Kutta")
    plt.scatter(gil.t*.1, gil.x, s=10,
                label="Gillespie")

    plt.xlabel("x [steps]")
    plt.ylabel("y")
    plt.legend()
    plt.show()
