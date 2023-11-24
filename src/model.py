import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import plotly.express as px
from jax import random
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS

from .download_data import get_admissions_data


def admissions_model(timestamp, admissions=None):
    # Hyper-parameters
    intercept_loc = 1e5
    intercept_scale = 1e4
    gradient_loc = 1e3
    gradient_scale = 1e2
    noise_rate = 1.0

    # Priors
    intercept = numpyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
    gradient = numpyro.sample("gradient", dist.Normal(gradient_loc, gradient_scale))
    admissions_loc = intercept + gradient * timestamp
    admissions_scale = numpyro.sample("noise", dist.Exponential(noise_rate))
    return numpyro.sample(
        "admissions", dist.Normal(admissions_loc, admissions_scale), obs=admissions
    )


if __name__ == "__main__":
    df_admissions = get_admissions_data()

    timestamps = jnp.arange(len(df_admissions.index))
    admissions = df_admissions["Total Emergency Admissions"].values

    nuts_kernel = NUTS(admissions_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, timestamps, admissions=admissions, extra_fields=())
    mcmc.print_summary()

    # Compute empirical posterior distribution over mu
    samples_1 = mcmc.get_samples()
    posterior_mu = jnp.expand_dims(
        samples_1["gradient"], -1
    ) * timestamps + jnp.expand_dims(samples_1["intercept"], -1)

    mean_mu = jnp.mean(posterior_mu, axis=0)
    hpdi_mu = hpdi(posterior_mu, 0.9)
    fig = px.line(x=timestamps, y=mean_mu)
    scatter = px.scatter(x=timestamps, y=admissions)
    fig.add_traces(list(scatter.select_traces()))
    fig.update_yaxes(range=[0, 1e6])
    fig.show()
