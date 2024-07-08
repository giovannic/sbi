import pytest

from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import likelihood_nn
from torch import ones, zeros

from sbi.inference import (
    likelihood_estimator_based_potential,
    annealed_likelihood_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian

@pytest.mark.parametrize("num_dim", (1, 3))
@pytest.mark.parametrize("num_trials", (4, 8))
def test_likelihood_returns_correct_shape(num_dim, num_trials):
    prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    simulator = diagonal_linear_gaussian
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_trials
    )
    density_estimator = likelihood_nn(
        model='maf',
        hidden_features=10,
        z_score_x='independent',
        z_score_theta='independent',
    )(theta, x)
    x_o = zeros((num_trials, num_dim))
    (
        potential_fn,
        _,
    ) = likelihood_estimator_based_potential(
        density_estimator,
        prior,
        x_o,
        enable_transform=False
    )
    assert potential_fn(theta).shape == (num_trials,)

@pytest.mark.parametrize("num_dim", (1, 3))
@pytest.mark.parametrize("num_trials", (4, 8))
def test_annealed_likelihood_returns_correct_shape(num_dim, num_trials):
    prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    simulator = diagonal_linear_gaussian
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_trials
    )
    density_estimator = likelihood_nn(
        model='maf',
        hidden_features=10,
        z_score_x='independent',
        z_score_theta='independent',
    )(theta, x)
    x_o = zeros((num_trials, num_dim))
    (
        potential_fn,
        _,
    ) = annealed_likelihood_estimator_based_potential(
        density_estimator,
        prior,
        x_o,
        enable_transform=False,
        alpha=.1
    )
    assert potential_fn(theta).shape == (num_trials,)
