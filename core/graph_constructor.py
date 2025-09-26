import torch

def create_2d_grid_graph(dims):
    W, H = dims
    edges = []
    for i in range(W):
        for j in range(H):
            idx = i * H + j
            if i + 1 < W:
                edges.append([idx, (i + 1) * H + j])
                edges.append([(i + 1) * H + j, idx])
            if j + 1 < H:
                edges.append([idx, i * H + (j + 1)])
                edges.append([i * H + (j + 1), idx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
