import torch as t
import numpy as np
from model import EigClippedInvariantGP, EigClippedMaxKernel
from scipy.optimize import minimize
from abc import ABC, abstractmethod

class AbstractEigClippedBO(ABC):
	"""Abstract BO optimizer with eigenvalue clipping. The specific clipping method is determined by the
	clip_kernel_class returned by the clip_kernel_class method.
	"""
	def __init__(self, domain, base_kernel_class, max_invariant_ker_class, base_kernel_args=[], max_invariant_ker_args=[], cold_start=2):
		"""Intialize the optimizer

		Args:
				domain (torch.Tensor): parameters search space
				base_kernel_class (class): base kernel class
				max_invariant_ker_class (class): max kernel class
				base_kernel_args (list, optional): arguments for the base kernel. Defaults to [].
				max_invariant_ker_args (list, optional): arguments for the max kernel. Defaults to [].
				cold_start (int, optional): number of observations for warming up. Defaults to 2.
		"""
		self._train_X = None
		self._train_y = None
		self._domain = domain
		self._d = domain.shape[0]
		self._base_kernel_class = base_kernel_class
		self._base_kernel_args = base_kernel_args
		self._max_invariant_ker_class = max_invariant_ker_class
		self._max_invariant_ker_args = max_invariant_ker_args
		self._gp = None
		self._cold_start = cold_start

	@abstractmethod
	def clip_kernel_class(self):
		"""Return the kernel class and arguments for the clipped kernel used in the GP model

		Returns:
				tuple: the kernel class and a list of arguments for the kernel
		"""
		...

	def get_hyperparameters(self):
		"""Get the hyperparameters of the model

		Returns:
				tuple: the hyperparameters of the GP
		"""
		return self._gp._kernel._max_ker._k_base.outputscale.item() , self._gp._kernel._max_ker._k_base.base_kernel.lengthscale.item()
	
	def standardize_output(self, y):
		"""Standardize the function values

		Args:
				y (torch.Tensor): function values

		Returns:
				torch.Tensor: standardized function values
		"""
		return (y - t.mean(y)) / t.std(y)
	
	def destandardize_output(self, ynorm):
		"""Destandardize function values

		Args:
				ynorm (torch.Tensor): the standardized function values

		Returns:
				torch.Tensor: the unstandardized function values
		"""
		return ynorm * t.std(self._train_y) + t.mean(self._train_y)

	@abstractmethod
	def get_ucb_and_grads(self, x):
		"""Compute UCB values and gradients on inputs

		Args:
				x (torch.Tensor): the inputs
		
		Returns:
				tuple: UCB values and UCB gradients
		"""
		...

	def get_ucb(self, x, beta, minimize=False):
		"""Compute the UCB acquisition function on inputs

		Args:
				x (torch.Tensor): the inputs
				beta (float): the exploration-exploitation parameter
				minimize (bool, optional): flip the sign of UCB values. Defaults to False.

		Returns:
				torch.Tensor: the UCB values on inputs
		"""
		x = t.Tensor(x)
		l = x.ndim
		if l == 1:
			x = x.unsqueeze(0)

		post_mean, post_covar = self._gp.posterior_dist(x)
		post_std = t.sqrt(t.diag(post_covar))

		ucb = post_mean + np.sqrt(beta) * post_std

		if minimize:
			ucb = -ucb
		
		if l == 1:
			if ucb.shape[0] == 1:
				ucb = ucb.item()
			else:
				ucb.squeeze(0)

		return ucb

	def next_query(self, n_restarts=10, raw_samples=512, beta=None):
		"""Compute the next query of the optimizer

		Args:
				n_restarts (int, optional): number of restarts for multi-start gradient ascent. Defaults to 10.
				raw_samples (int, optional): number of initial points. Defaults to 512.
				beta (float, optional): exploration-exploitation trade-off parameter. Defaults to None.

		Returns:
				torch.Tensor: the next input to query
		"""
		x0 = t.rand((self._d,)) * (self._domain[:, 1] - self._domain[:, 0]) + self._domain[:, 0]
		# If the GP is not fitted yet, return a random point in the domain
		if self._gp is None:
			return x0
		
		# If beta is not provided, use a default value based on the number of observations and the input dimension
		if beta is None:
			beta = 0.5 * self._d * np.log(2 * self._train_X.shape[0])
		
		# Sample initial points and select the best ones based on UCB values
		with t.no_grad():
			soboleng = t.quasirandom.SobolEngine(dimension=self._d)
			x = soboleng.draw(raw_samples) * (self._domain[:, 1] - self._domain[:, 0]) + self._domain[:, 0]
			x = t.cat((x, self._train_X[t.topk(self._train_y, 1).indices]))
			ucb_vals = self.get_ucb(x, beta)
			x = x[t.topk(ucb_vals, k=n_restarts).indices]

		# Optimize the acquisition function starting from the selected initial points
		x_opt = None
		ucb_opt = 10000000
		for i in range(n_restarts):
			opt = minimize(self.get_ucb_and_grads, x0=x[i], args=(beta,True), method="L-BFGS-B", jac=True, bounds=self._domain.numpy())

			if ucb_opt > opt.fun:
				ucb_opt = opt.fun
				x_opt = t.Tensor(opt.x)
		
		return x_opt

	def tell(self, x, y):
		"""Add another observation to the dataset.

		Args:
				x (torch.Tensor): the input query
				y (float): the function value
		"""
		# Add the new observation to the training data
		if self._train_X is None:
			self._train_X = t.Tensor(t.unsqueeze(x, 0))
		else:
			self._train_X = t.cat((self._train_X, t.unsqueeze(x, 0)), 0)

		if self._train_y is None:
			self._train_y = t.Tensor(t.unsqueeze(y, 0))
		else:
			self._train_y = t.cat((self._train_y, t.Tensor([y])), 0)

		# Fit the GP model if we have enough observations
		if self._train_X.shape[0] > self._cold_start:
			normalized_train_X = self._train_X
			normalized_train_y = self.standardize_output(self._train_y)
			with t.enable_grad():
				ker_class, args = self.clip_kernel_class()
				self._gp = EigClippedInvariantGP(
					normalized_train_X, normalized_train_y,
					ker_class, self._base_kernel_class,
					self._max_invariant_ker_class,
					clip_kernel_args=args, base_kernel_args=self._base_kernel_args,
					max_invariant_ker_args=self._max_invariant_ker_args)


class NystromBO(AbstractEigClippedBO):
	"""BO optimizer with Nystrom extension
	"""
	def __init__(self, domain, base_kernel_class, max_invariant_ker_class, base_kernel_args=[], max_invariant_ker_args=[], cold_start=2):
		"""Intialize the optimizer

		Args:
				domain (torch.Tensor): parameters search space
				base_kernel_class (class): base kernel class
				max_invariant_ker_class (class): max kernel class
				base_kernel_args (list, optional): arguments for the base kernel. Defaults to [].
				max_invariant_ker_args (list, optional): arguments for the max kernel. Defaults to [].
				cold_start (int, optional): number of observations for warming up. Defaults to 2.
		"""
		super(NystromBO, self).__init__(domain, base_kernel_class, max_invariant_ker_class, base_kernel_args, max_invariant_ker_args, cold_start)

	def clip_kernel_class(self):
		"""Return the kernel class and arguments for the clipped kernel used in the GP model

		Returns:
				tuple: the kernel class and a list of arguments for the kernel
		"""
		return EigClippedMaxKernel, [self._train_X]
	
	def get_ucb_and_grads(self, x, beta, minimize=False):
		"""Compute UCB values and gradients on inputs

		Args:
				x (torch.Tensor): the inputs
				beta (float): the exploration-exploitation trade-off
				minimize (bool, optional): flip the sign of UCB values and gradients. Defaults to False.

		Returns:
				tuple: UCB values and UCB gradients
		"""
		x_g = t.tensor(x, requires_grad=True).float()
		if x_g.ndim == 1:
			x_g = x_g.unsqueeze(0)

		post_mean, post_covar = self._gp.posterior_dist(x_g)
		post_std = t.sqrt(t.diag(post_covar))

		ucb = post_mean + np.sqrt(beta) * post_std
		if minimize:
			ucb *= -1
		gucb = t.autograd.grad(ucb, x_g, retain_graph=True)[0]

		return ucb.detach(), gucb
