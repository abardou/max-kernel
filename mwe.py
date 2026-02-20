import torch as t
from model import MaxKernel
import gpytorch
import numpy as np
from tools import signed_permutation_matrices
from optimizer import NystromBO
from time import time

def ackley(X, scaling_factor=16):
	"""Ackley objective function

	Args:
			X (torch.Tensor): the inputs

	Returns:
			torch.Tensor: the function values at X
	"""
	X_scaled = X * scaling_factor

	a = 20
	b = 0.2
	c = 2 * np.pi

	norms = t.norm(X_scaled, p=2, dim=1)

	coss = 0
	for i in range(X_scaled.shape[1]):
		coss += t.cos(c * X_scaled[:, i])
	coss /= X_scaled.shape[1]

	return -(- a * t.exp(- b * norms) - t.exp(coss) + a + np.exp(1))


if __name__ == "__main__":
	d = 2
	search_domain = t.Tensor([[-1.0, 1.0] for _ in range(d)]) # normalized domain for the BO algorithms, will be rescaled by the objective if needed

	n_iterations = 30
	cold_start = 5

	invariant_objective = ackley
	max_objective = 0.0 # global maximum of the objective, used to compute regret

	symmetries = signed_permutation_matrices(d) # d! 2^d
	max_kernel_bo_optimizer = NystromBO(
		search_domain, # search space
		gpytorch.kernels.RBFKernel, # base kernel class
		MaxKernel, # max invariant kernel class
		base_kernel_args=[], # arguments for the base kernel (e.g., Matern lengthscale)
		max_invariant_ker_args=[symmetries], # arguments for the max invariant kernel (e.g., the list of symmetry matrices)
		cold_start=cold_start # number of initial random points before fitting the GP and using the acquisition function
	)

	xx = []
	yy = []
	rr = []
	tt = []
	for iteration_id in range(n_iterations):
		start_t = time()
		x_next = max_kernel_bo_optimizer.next_query()
		y_next = invariant_objective(t.unsqueeze(x_next, 0))[0]
		max_kernel_bo_optimizer.tell(x_next, y_next)
		end_t = time()

		xx.append(x_next.detach().numpy())
		yy.append(y_next.item())
		tt.append(end_t - start_t)
		rr.append(max_objective - yy[-1])
		print(f"== Iteration {iteration_id+1} ==")
		print("\tQuery point:", x_next)
		print("\tObjective value at query point:", yy[-1])
		print("\tRegret:", rr[-1])
		print("\tTime taken for iteration:", tt[-1], "seconds")