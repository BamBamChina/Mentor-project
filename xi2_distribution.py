import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2

#модели
model_bkg = lambda bkg: bkg
model_bkg_sig = lambda x, bkg, mu, sigma, height: bkg + height * np.exp(-(x - mu)**2 / sigma**2) / (2 * np.pi * sigma**2)**0.5


#генерация данных
x = np.linspace(0, 20, 21)
data = (np.random.uniform(100, 110, size=21) + 100 * norm(10).pdf(x)).astype(int)

plt.bar(np.arange(21), data)
plt.ylim(90, 150)
plt.show()




def chi2_p(data, model):
    return np.sum((data - model)**2/ model)
def chi2_n(data, model):
    return np.sum((data-model)**2/data)
#минимизация хи-квадратов
def minimize_chi2_bkg(data, chi2_func):
    def foo(param):
        bkg = param[0]
        model = model_bkg(bkg)
        return chi2_func(data, model)
    result = minimize(foo, [100], bounds=[(50, 150)])
    return result.fun

def minimize_chi2_bkg_sig(data, chi2_func):
    def foo(params):
        bkg, mu, sigma, height = params
        model = model_bkg_sig(x, bkg, mu, sigma, height)
        return chi2_func(data, model)
    
    #параметры
    x0 = [100, 10, 1, 100]
    bounds = [(50, 150), (5, 15), (0.5, 3), (50, 200)]
    
    result = minimize(foo, x0, bounds=bounds)
    return result.fun

n = 1000
delta_chi2_p = []
delta_chi2_n = []
for i in range(n):
    x = np.linspace(0, 20, 21)
    data = (np.random.uniform(100, 110, size=21) + 100 * norm(10).pdf(x)).astype(int)

    difference1 = minimize_chi2_bkg(data, chi2_p) - minimize_chi2_bkg_sig(data, chi2_p)
    delta_chi2_p.append(difference1)

    difference2 = minimize_chi2_bkg(data, chi2_n) - minimize_chi2_bkg_sig(data, chi2_n)
    delta_chi2_n.append(difference2)
