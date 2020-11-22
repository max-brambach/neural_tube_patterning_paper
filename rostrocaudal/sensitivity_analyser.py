import numpy as np
import matplotlib.pyplot as plt
import tqdm


class SensitivityAnalyser(object):
    """
    Class for analysing the parameter sensitivity of a model.
    """
    def __init__(self, function, parameters, variations=np.array([.01, .02, .05, .1])):
        """
        :param function: function with one parameter argument (e.g. f(p))
                         has to return one value (costfunction)
        :param parameters: list of parameters p
        :param variations: list of factors the parameters are varied with; default: 1%, 2%, 5%, 10%
        """
        self.f = function
        self.p = np.array(parameters)
        self.n_p = len(self.p)
        self.out_init = self.f(self.p)
        self.var = np.array(variations)
        self.n_var = len(self.var)
        self.all_var = np.concatenate([-1*self.var[::-1], np.array([0]),self.var])
        self.out_var = np.zeros(shape=[self.n_var * 2 + 1, self.n_p])
        self.out_robustness = np.zeros(self.n_var * 2 + 1)

    def go(self, directory, plot=False):
        """
        Convenience function.
        :param directory: str, where to save the results.
        :param plot: bool, if True, a sensitivity diagram is plotted
        :return:
        """
        self.vary_parameters()
        self.robustness()
        self.save_sensitivity(directory)
        if plot:
            self.plot_sensitivity_diagram()

    def vary_parameters(self):
        """
        Compute sensitivity for all parameters.
        :return:
        """
        for i in tqdm.trange(self.n_p):
            for j, var in enumerate(self.all_var):
                p = np.array(self.p).copy()
                p[i] *= 1.+var
                # p_down[i] *= 1.-var
                self.out_var[j, i] = self.f(p)

    def vary_one_parameter(self, i, directory):
        """
        Compute sensitivity for one parameter
        :param i: int, index of the parameter
        :param directory: str, where to save the output
        :return:
        """
        for j, var in enumerate(self.all_var):
            p = np.array(self.p).copy()
            p[i] *= 1. + var
            np.save(directory, self.f(p))

    def robustness(self):
        """
        Compute the robustness (variation upon change of all parameters).
        :return:
        """
        for j, var in enumerate(self.all_var):
            p = np.array(self.p).copy()
            p *= 1. + var
            # p_down[i] *= 1.-var
            self.out_robustness[j] = self.f(p)

    def vary_parameters_int(self):
        """
        Vary all parameters by integer values.
        :return:
        """
        for i in tqdm.trange(self.n_p):
            for j, var in enumerate(self.all_var):
                p = np.array(self.p).copy()
                p[i] += var
                # p_down[i] *= 1.-var
                self.out_var[j, i] = self.f(p)

    def plot_sensitivity_diagram(self, normalise_cost=True):
        """
        Plot a sensitivity diagram
        :param normalise_cost: bool, if True, output is in terms of costfunction fold-change;
                                     if False output is in absolute units of the costfunction
        :return:
        """
        plt.rcParams.update({'font.size': 15})
        if normalise_cost:
            out_var_plot = self.out_var/self.out_init-1
            out_robust_plot= self.out_robustness/self.out_init-1
            y_label = 'costfunction [fold change]'
        else:
            out_var_plot = self.out_var
            out_robust_plot = self.out_robustness
            y_label = 'costfunction [absolute values]'
        plt.figure()
        plt.grid()
        for i in range(self.n_p):
            plt.plot(self.all_var*100, out_var_plot[:, i], linewidth=2)
        plt.plot(self.all_var*100, out_robust_plot, 'k', linewidth=2)
        plt.xlabel('parameter change [%]')
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.show()

    def save_sensitivity(self, directory):
        """
        Save the sensitivity results
        :param directory: str, where to save
        :return:
        """
        np.save(directory+'.npy', self.out_var)

    def save_robustness(self, directory):
        """
        Save the robustnes results.
        :param directory: str, where to save
        :return:
        """
        np.save(directory + '.npy', self.out_robustness)

    def load_sensitivity(self, directory):
        """
        Load results.
        :param directory: str, from where
        :return:
        """
        self.out_var = np.load(directory+'.npy')


if __name__ == "__main__":
    """
    A little example.
    """
    def f(p):
        def model(x, p):
            return p[0]*np.exp(p[1]*x) + p[2]
        x = np.array(range(100))/100
        y_true = model(x, [1, -1, .5]) + (np.random.random(len(x))-0.5)*0.01
        y_model = model(x, p)
        return np.sum((y_true-y_model)**2)
    p = [1, -1, .5]
    sens = SensitivityAnalyser(f, p, variations=np.arange(.001, .02, .0001))
    sens.go('TEST_sensitivity')
