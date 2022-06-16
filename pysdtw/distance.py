import torch

def pairwise_l2_squared(x, y):
    x_norm = (x**2).sum(-1).unsqueeze(-1)
    y_norm = (y**2).sum(-1).unsqueeze(-2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.mT)
    return torch.clamp(dist, 0.0, torch.inf)

# pairwise
# https://stackoverflow.com/questions/46655878/how-to-calculate-pairwise-distance-matrix-on-the-gpu
# USE_64 = True

# if USE_64:
#     bits = 64
#     np_type = np.float64
# else:
#     bits = 32
#     np_type = np.float32

# @cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
# def distance_matrix(mat, out):
#     m = mat.shape[0]
#     n = mat.shape[1]
#     i, j = cuda.grid(2)
#     d = 0
#     if i < m and j < m:
#         for k in range(n):
#             tmp = mat[i, k] - mat[j, k]
#             d += tmp * tmp
#         out[i, j] = d

# def gpu_dist_matrix(mat):
#     rows = mat.shape[0]

#     block_dim = (16, 16)
#     grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

#     stream = cuda.stream()
#     mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
#     out2 = cuda.device_array((rows, rows))
#     distance_matrix[grid_dim, block_dim](mat2, out2)
#     out = out2.copy_to_host(stream=stream)

#     return out




# def pairwise_l2_squared_diag(x, y, theta):
#     x_norm = (theta * x**2).sum(1).view(-1, 1)
#     y_t = torch.transpose(y, 0, 1)
#     y_norm = (theta * y**2).sum(1).view(1, -1)
#     dist = x_norm + y_norm - 2.0 * torch.mm(theta * x, y_t)
#     return torch.clamp(dist, 0.0, np.inf)

# def pairwise_l2_squared_full(x, y, theta):
#     x_norm = (theta * x**2).sum(1).view(-1, 1)
#     y_t = torch.transpose(y, 0, 1)
#     y_norm = (theta * y**2).sum(1).view(1, -1)
#     dist = x_norm + y_norm - 2.0 * torch.mm(theta * x, y_t)
#     return torch.clamp(dist, 0.0, np.inf)

# def _euclidean_dist_func(x, y):
#     """
#     Calculates the Euclidean distance between each element in x and y per timestep
#     """
#     n = x.size(1)
#     m = y.size(1)
#     d = x.size(2)
#     x = x.unsqueeze(2).expand(-1, n, m, d)
#     y = y.unsqueeze(1).expand(-1, n, m, d)
#     return torch.pow(x - y, 2).sum(3)
