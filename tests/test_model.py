import model
import numpy
import numpyro
import jax.numpy as jnp
import jax
import numpy

def test_admissions_model():
    num_data_points = 10
    timestamp_arr = jnp.array(numpy.linspace(0, 1, num_data_points))
    admissions_arr = jnp.array(numpy.random.rand(num_data_points))

    modeller = model.TimeSeriesModeller(model.admissions_model, predictor_name="admissions")
    
    # Fit the model
    modeller.fit(timestamps_train=timestamp_arr, predictors_train=admissions_arr, )

    # Get the samples from the MCMC run
    samples = modeller._mcmc.get_samples()

    # Assertions to ensure necessary variables are in the samples
    assert "intercept" in samples, "Samples should contain 'intercept'"
    assert "gradient" in samples, "Samples should contain 'gradient'"
    assert "noise" in samples, "Samples should contain 'noise'"
    # assert "admissions" in samples, "Samples should contain 'admissions'"

def test_parameter_priors():
    timestamps = jnp.array([1, 2, 3, 4, 5])
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as tr:
        model.admissions_model(timestamps)

    # Validate priors
    assert tr['intercept']['fn'].loc == 1e5, "Intercept mean should be 1e5"
    assert tr['intercept']['fn'].scale == 1e4, "Intercept scale should be 1e4"
    assert tr['gradient']['fn'].loc == 1e3, "Gradient mean should be 1e3"
    assert tr['gradient']['fn'].scale == 1e2, "Gradient scale should be 1e2"

if __name__=='__main__':
    test_admissions_model()
