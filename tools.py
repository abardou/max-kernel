import itertools
import torch as t

def grouped_permutation_matrices(group_sizes, device=None) -> t.Tensor:
	"""
	Return all permutation matrices that permute whole groups (blocks).
	- group_sizes: sequence of G positive ints, sizes of groups in original ordering.
	- device: optional torch device
	
	Result shape: (G!, D, D) where D = sum(group_sizes).
	For a permutation `perm`, the resulting matrix P satisfies:
			y = P @ x
	where y is the concatenation of blocks x[perm[0]], x[perm[1]], ...
	"""
	group_sizes = list(group_sizes)
	G = len(group_sizes)
	D = sum(group_sizes)

	# compute start offsets for original groups
	offsets = [0]
	for s in group_sizes:
			offsets.append(offsets[-1] + s)
	# offsets[i] is start index of group i; offsets[-1] == D

	mats = []
	print(group_sizes)
	for perm in itertools.permutations(range(G)):
			P = t.zeros((D, D), device=device)
			dst = 0
			# dst ranges over positions in the permuted vector
			for dst_pos, src_group in enumerate(perm):
					src_start = offsets[src_group]
					src_size = group_sizes[src_group]
					src_slice = slice(src_start, src_start + src_size)
					dst_slice = slice(dst, dst + src_size)
					# put an identity block mapping the src block into dst block
					P[dst_slice, src_slice] = t.eye(src_size, device=device)
					dst += src_size
			mats.append(P)
	return t.stack(mats, dim=0)

def grouped_permutation_matrices_equal_groups(total_dim: int, group_size: int, device=None) -> t.Tensor:
	"""
	Convenience wrapper when groups all have the same size.
	total_dim must be divisible by group_size.
	"""
	assert total_dim % group_size == 0, "total_dim must be divisible by group_size"
	G = total_dim // group_size
	return grouped_permutation_matrices([group_size] * G, device=device)

def sign_flip_matrices(d: int, device=None) -> t.Tensor:
	"""
	Returns a tensor of shape (2^d, d, d) containing alllambda x: ackley(x) diagonal sign-flip matrices.
	"""
	mats = []
	for signs in itertools.product([1.0, -1.0], repeat=d):
		D = t.diag(t.tensor(signs, device=device))
		mats.append(D)
	return t.stack(mats, dim=0)

def signed_permutation_matrices(d: int, device=None) -> t.Tensor:
	"""
	Returns a tensor of shape (2^d * d!, d, d) containing all
	signed-permutation matrices D @ P.
	"""
	P_all = grouped_permutation_matrices_equal_groups(d, 1, device=device) # (d!, d, d)
	D_all = sign_flip_matrices(d, device=device) # (2^d, d, d)

	# expand and multiply: for each D and each P compute D @ P
	# result shape: (2^d, d!, d, d) → reshape to (2^d*d!, d, d)
	DP = D_all.unsqueeze(1) @ P_all.unsqueeze(0)
	return DP.view(-1, d, d)