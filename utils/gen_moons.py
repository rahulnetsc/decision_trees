from sklearn.datasets import make_moons

def moons_dataset(n_samples:int = 1000, noise: float = 0.2, state = 42):
    """
    Generate a two-moons dataset.

    Args:
        n_samples (int): Number of points.
        noise (float): Standard deviation of Gaussian noise.
        state (int): Random seed.

    Returns:
        X, y: Features and labels from sklearn.datasets.make_moons.
    """
    return make_moons(n_samples=n_samples,noise=noise, random_state=state)