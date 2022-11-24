import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the expression whose roots we want to find

a = 0.5
R = 2**0.5

func = lambda tau : R - 1/(4*tau) * (np.sqrt(4*tau**2 + 1)*2*tau + np.arcsinh(2*tau))

# Plot it

tau = np.linspace(-0.5, 4.5, 501)

plt.plot(tau, func(tau))
plt.xlabel("tau")
plt.ylabel("expression value")
plt.grid()
plt.show()

# Use the numerical solver to find the roots

tau_initial_guess = 0.5
tau_solution = fsolve(func, tau_initial_guess)

print("The solution is tau = {}".format(tau_solution))
print("at which the value of the expression is %f" % func(tau_solution))