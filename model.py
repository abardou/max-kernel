import torch as t
import gpytorch

class MaxKernel:
	"""Simple implementation of the Max kernel on a finite group of symmetries
	"""
	def __init__(self, k_base, symmetries):
		"""Initialize the Max kernel

		Args:
				k_base (callable): a callable function able to compute a kernel matrix given two input tensors
				symmetries (torch.Tensor): the stacked matrices describing the symmetries, of size n x d x d where n is the
				number of symmetries and d the number of dimensions
		"""
		self._k_base = k_base
		self._symmetries = symmetries.unsqueeze(0)
		self._s = symmetries.size()[0]

	def __call__(self, X, Y):
		"""Compute the kernel matrix for two input tensors

		Args:
				X (torch.Tensor): the first input tensor, of size m x d where m is the number of observations and d the number of dimensions
				Y (torch.Tensor): the second input tensor, of size n x d where n is the number of observations and d the number of dimensions

		Returns:
				torch.Tensor: the m x n kernel matrix
		"""
		Y_extended = Y.unsqueeze(1).unsqueeze(-1) # Y is transformed into  n x s x d, where s is the number of symmetries
		Ys = t.matmul(self._symmetries, Y_extended).squeeze(-1)
		Ks = self._k_base(X, Ys).evaluate()
		res = t.max(Ks, -1)[0]

		return res
	
class EigClippedMaxKernel:
	"""Clip-invariant kernel with Nystrom approximation of the covariance matrix
	"""
	def __init__(self, max_kernel, dataset, threshold=1e-4):
		"""Initialize the kernel
		
		Args:
				max_kernel (callable): the max kernel to be approximated, should be a callable function able to compute a kernel
				matrix given two input tensors
				dataset (torch.Tensor): the dataset on which the Nystrom approximation is computed, of size n x d where n is the
				number of observations and d the number of dimensions
				threshold (float, optional): eigenvalue threshold for clipping. Defaults to 1e-4.
		"""
		self._max_ker = max_kernel
		self._dataset = dataset

		# The covariance matrix on the dataset
		cov_mat = self._max_ker(dataset, dataset)

		# Eigen-decomposition
		L, V = t.linalg.eig(cov_mat)

		# Eigenvalue clipping
		idx = t.real(L) >= threshold
		clipped_L = L[idx]
		clipped_V = V[:, idx]

		# Inverted covariance matrix for the Nystrom extension
		inv_L = 1 / clipped_L
		self._inv_clipped_cov_mat = t.real(clipped_V @ t.diag_embed(inv_L) @ t.transpose(clipped_V, 0, 1))
	
	def __call__(self, X, Y):
		"""Compute the kernel matrix for two input tensors
		Args:
				X (torch.Tensor): the first input tensor, of size m x d where m is the number of observations and d the number
				of dimensions
				Y (torch.Tensor): the second input tensor, of size n x d where n is the number of observations and d the number
				of dimensions
		
		Returns:
				torch.Tensor: the m x n kernel matrix
		"""
		# Both inputs are m x d and n x d, output is m x n
		max_ker_X = self._max_ker(X, self._dataset)
		max_ker_Y = self._max_ker(Y, self._dataset)

		return t.transpose(max_ker_X, 0, 1) @ self._inv_clipped_cov_mat @ max_ker_Y

class DummyGP(gpytorch.models.ExactGP):
	"""Basis GP, for hyperparameter inference and naive BO
	"""
	num_outputs = 1

	def __init__(self, train_x, train_y, base_kernel_class, base_kernel_args=[], likelihood=gpytorch.likelihoods.GaussianLikelihood()):
		"""Initialize the Dummy GP

		Args:
				train_x (torch.Tensor): queries in the training set
				train_y (torch.Tensor): function values in the training set
				base_kernel_class (gpytorch.kernels.Kernel): kernel of the GP
				base_kernel_args (list, optional): arguments of the kernel. Defaults to [].
				likelihood (gpytorch.likelihood.Likelihood, optional): likelihood of the GP.
				Defaults to gpytorch.likelihoods.GaussianLikelihood().
		"""
		super(DummyGP, self).__init__(train_x, train_y, likelihood)
		self._train_X = train_x
		self._train_y = train_y
		self._likelihood = likelihood
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel_class(*base_kernel_args))

	def forward(self, x):
		"""Compute the posterior distribution

		Args:
				x (torch.Tensor): the inputs

		Returns:
				gpytorch.distributions.MultivariateNormal: the posterior distribution on the inputs
		"""
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

	def posterior(self, X, posterior_transform=None):
		"""Compute the posterior distribution on inputs
		Args:
				X (torch.Tensor): the inputs
				posterior_transform (callable, optional): a function to transform the posterior distribution. Defaults to None.

		Returns:
				gpytorch.distributions.MultivariateNormal: the posterior distribution on the inputs
		"""
		return self.likelihood(self(X))

	def fit(self):
		"""Fit the hyperparameters of the GP
		"""
		# Find optimal model hyperparameters
		self.train()
		self.likelihood.train()

		# Use the adam optimizer
		optimizer = t.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

		# "Loss" for GPs - the marginal log likelihood
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self)

		for _ in range(50):
			# Zero gradients from previous iteration
			optimizer.zero_grad()
			# Output from model
			output = self(self._train_X)
			# Calc loss and backprop gradients
			loss = -mll(output, self._train_y)
			loss.backward()
			optimizer.step()

		self.eval()
		self._likelihood.eval()


class EigClippedInvariantGP:
	"""Invariant GP with trimming of negative eigenvalues
	"""
	def __init__(self, train_X, train_Y, clip_kernel_class, base_kernel_class, max_invariant_ker_class, clip_kernel_args=[], base_kernel_args=[], max_invariant_ker_args=[]):
		"""Intialize the GP

		Args:
				train_X (torch.Tensor): queries for the training set
				train_Y (torch.Tensor): function values for the training set
				clip_kernel_class (class): clipped kernel class
				base_kernel_class (class): base kernel class
				max_invariant_ker_class (class): max kernel used
				clip_kernel_args (list, optional): arguments for the clip kernel. Defaults to [].
				base_kernel_args (list, optional): arguments for the base kernel. Defaults to [].
				max_invariant_ker_args (list, optional): arguments for the max kernel. Defaults to [].
		"""
		# First things first: learn the hyperparameters
		d_gp = DummyGP(train_X, train_Y, base_kernel_class, base_kernel_args=base_kernel_args)
		d_gp.fit()

		# Extract the hyperparameters and build the kernel
		self._lambda, self._lS, self._noise = d_gp.covar_module.outputscale.item(), d_gp.covar_module.base_kernel.lengthscale.item(), d_gp.likelihood.noise.item()
		base_kernel = gpytorch.kernels.ScaleKernel(base_kernel_class(*base_kernel_args))
		base_kernel.outputscale = self._lambda
		base_kernel.base_kernel.lengthscale = self._lS
		self._kernel = clip_kernel_class(max_invariant_ker_class(base_kernel, *max_invariant_ker_args), *clip_kernel_args)

		self._train_X = train_X
		self._train_y = train_Y

	def posterior_mean(self, ker_Xtrain, inv_psd):
		"""Compute the posterior mean on inputs

		Args:
				ker_Xtrain (torch.Tensor): kernel vector on inputs
				inv_psd (torch.Tensor): inverted covariance matrix

		Returns:
				torch.Tensor: posterior mean on inputs
		"""
		return t.matmul(t.matmul(ker_Xtrain, inv_psd), self._train_y)

	def posterior_covar(self, ker_XX, ker_Xtrain, inv_psd):
		"""Compute the posterior covariance matrix on inputs

		Args:
				ker_XX (torch.Tensor): kernel values on inputs
				ker_Xtrain (torch.Tensor): kernel vector on inputs
				inv_psd (torch.Tensor): inverted covariance matrix

		Returns:
				torch.Tensor: posterior covariance matrix on inputs
		"""
		cov = ker_XX - t.matmul(t.matmul(ker_Xtrain, inv_psd), ker_Xtrain.T)
		return t.max(cov, t.zeros_like(cov))
	
	def posterior_dist(self, X):
		"""Compute posterior mean and covariance matrix on inputs

		Args:
				X (torch.Tensor): inputs

		Returns:
				tuple: posterior mean on inputs, posterior covariance matrix on inputs
		"""
		X = t.Tensor(X)

		n, _ = X.shape
		# Stack X and train_X together
		XX_train = t.cat((X, self._train_X), 0)
		# Compute extended covariance matrix
		cov_mat = self._kernel(XX_train, XX_train)
		ker_XX = cov_mat[:n, :n]
		ker_XXtrain = cov_mat[:n, n:]
		ker_XtrainXtrain = cov_mat[n:, n:]

		inv_psd = t.linalg.inv(ker_XtrainXtrain + self._noise * t.eye(self._train_X.shape[0]))
		
		return self.posterior_mean(ker_Xtrain=ker_XXtrain, inv_psd=inv_psd), self.posterior_covar(ker_XX=ker_XX, ker_Xtrain=ker_XXtrain, inv_psd=inv_psd)