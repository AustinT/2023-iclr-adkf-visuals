import numpy as np
import scipy
from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to control plots
obs_noise = 1e-4
x_plot = np.linspace(0, 1, 1000)


def rbf_kernel(x, y, lengthscale):
    def _to_2d(arr):
        if arr.ndim == 1:
            arr = np.atleast_2d(arr).T
        return arr

    x = _to_2d(x)
    y = _to_2d(y)
    d_xy = cdist(x, y)
    return np.exp(-((d_xy / lengthscale) ** 2))


def gp_posterior(x_data, y_data, x_test, lengthscale, obs_noise=obs_noise):
    k_xx = rbf_kernel(x_test, x_test, lengthscale)
    k_x_data = rbf_kernel(x_test, x_data, lengthscale)
    k_data_data = rbf_kernel(x_data, x_data, lengthscale)
    k_dd_cho = scipy.linalg.cho_factor(
        k_data_data + obs_noise * np.eye(len(k_data_data))
    )
    mu = k_x_data @ scipy.linalg.cho_solve(k_dd_cho, y_data)
    cov = k_xx - k_x_data @ scipy.linalg.cho_solve(k_dd_cho, k_x_data.T)
    return mu, np.diag(cov)


def difficult_function(a):
    return np.sin(np.exp(4 * a))


def animate_gp_samples(N_gp_samples=5, save_file: str = "./gp-samples-animated.mp4"):
    """Draw some samples from a GP and animate them as the lengthscale changes."""

    # Draw samples
    iid_gaussian_samples = np.random.randn(N_gp_samples, len(x_plot))

    # Initialize plot
    fig, ax = plt.subplots()
    lines = tuple([ax.plot([], [], "-")[0] for _ in range(N_gp_samples)])
    (scale_bar,) = ax.plot([], [], "k-")
    text = ax.text(0.5, -2.3, "")

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(-3, 3)
        return lines

    def update(lengthscale):
        # Scale bar
        scale_bar.set_data([0.5 - lengthscale / 2, 0.5 + lengthscale / 2], [-2.5, -2.5])

        # Text
        text.set_text(f"$\\ell={lengthscale:.3f}$")

        k = rbf_kernel(x_plot, x_plot, lengthscale)
        k_sqrt = np.linalg.cholesky(k + obs_noise * np.eye(len(x_plot)))
        gp_samples = iid_gaussian_samples @ k_sqrt.T
        for i in range(N_gp_samples):
            lines[i].set_data(x_plot, gp_samples[i])
        return lines

    ani = FuncAnimation(
        fig, update, frames=np.linspace(0.01, 0.2, 100), init_func=init, blit=True
    )
    ani.save(save_file)
    plt.close()


def plot_gp_fits():
    x_data = np.random.uniform(0, 1, size=10)

    for i, lengthscale in enumerate([0.03, 0.06]):

        # Plot data
        def f(a):
            return np.sin(a / lengthscale)

        y_data = f(x_data)
        plt.plot(x_data, y_data, "o")
        plt.plot(x_plot, f(x_plot), "k--")

        # Plot GP posterior
        mu, var = gp_posterior(x_data, y_data, x_plot, lengthscale)
        (line,) = plt.plot(x_plot, mu)
        plt.fill_between(x_plot, mu - var, mu + var, alpha=0.3, color=line.get_color())
        plt.title(f"GP posterior with good lengthscale, l={lengthscale:.3f}")
        plt.savefig(f"gp-fit{i}.png")
        plt.close()


def animate_poor_fits():

    f = difficult_function
    x_data = np.random.uniform(0, 1, size=10)
    y_data = f(x_data)

    # Initialize plot
    fig, ax = plt.subplots()

    def init():
        pass

    def update(lengthscale):
        ax.clear()

        ax.set_xlim(0, 1)
        ax.set_ylim(-3, 3)

        ax.text(0.5, -2.3, f"$\\ell={lengthscale:.3f}$")

        ax.plot(x_plot, f(x_plot), "k--")
        ax.plot(x_data, y_data, "o")
        (line,) = ax.plot([], [])

        mu, var = gp_posterior(x_data, y_data, x_plot, lengthscale)
        line.set_data(x_plot, mu)
        # line, = ax.plot(x_plot, mu)

        ax.fill_between(x_plot, mu - var, mu + var, alpha=0.3, color=line.get_color())
        return

    ani = FuncAnimation(
        fig, update, frames=np.linspace(0.01, 0.2, 100), init_func=init, blit=False
    )

    ani.save("poor_fitting_without_learned_features.mp4")
    plt.close()


if __name__ == "__main__":
    print("Animating samples.")
    animate_gp_samples()
    print("Showing fits.")
    plot_gp_fits()
    print("Difficult functions.")
    animate_poor_fits()
    print("End of script.")
