import torch
import matplotlib.pyplot as plt

def plot_2_dim_rotations(theta):
    torch.manual_seed(42)
    W = torch.randn(2, 2)
    x = torch.randn(2)
    iterations = 10
    xs = torch.zeros(iterations, 2)
    for m in range(iterations):
        scaled_theta = theta * m
        R = torch.tensor([[torch.cos(scaled_theta), -torch.sin(scaled_theta)], [torch.sin(scaled_theta), torch.cos(scaled_theta)]])
        x_rot = R @ W @ x
        xs[m] = x_rot

    plt.subplot(1, 2, 1)
    plt.plot(xs)
    plt.title(f'Rotations by {theta}')
    plt.xlabel('m')
    plt.ylabel('x_rot')

    plt.subplot(1, 2, 2)
    plt.plot(xs[:,0], xs[:,1])
    plt.gca().set_aspect('equal')
    plt.title(f'Rotations by {theta}')
    plt.xlabel('x_rot')
    plt.ylabel('y_rot')

    plt.tight_layout()
    plt.show()

plot_2_dim_rotations(torch.tensor(torch.pi / 8))

