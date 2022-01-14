import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

cos_sims = []
lambda_largest_deltas = []
lambda_largests = []
n = 100
for i in range(n):
    A = torch.randn(6, 1000)
    K = len(A)
    cos = torch.nn.CosineSimilarity(dim=0)

    A_copy = F.normalize(A)
    _, S, V = np.linalg.svd(A_copy)
    pca_direction = torch.tensor(V[0, :])
    lambda_largest = S[0]

    # for i in range(K):
    #     print(f'cosine similarity between A[{i}] and pca_direction: {cos(A[i].view(-1, 1), pca_direction.view(-1, 1))}')
    #
    for i in range(1):
        A[i] = torch.randn_like(A[i])
        # print(f'cosine similarity between A[{i}] and A_copy[{i}]: {cos(A[i].view(-1, 1), A_copy[i].view(-1, 1))}')
    A_copy = F.normalize(A)
    _, S1, V1 = np.linalg.svd(A_copy)
    pca_direction_1 = torch.tensor(V1[0, :])
    lambda_largest1 = S1[0]

    cos_sim = torch.abs(cos(pca_direction_1.view(-1, 1), pca_direction.view(-1, 1)))
    lambda_largest_delta = (lambda_largest1 - lambda_largest)

    cos_sims.append(cos_sim.item())
    lambda_largest_deltas.append(lambda_largest_delta)
    lambda_largests.append((lambda_largest, lambda_largest1))
plt.plot(list(range(n)), cos_sims)
plt.scatter(list(range(n)), lambda_largest_deltas)
plt.show()
stat = [1 if i >= 0.5 else 0 for i in cos_sims]
print(sum(stat))
print(lambda_largest_deltas)
print(lambda_largests)
