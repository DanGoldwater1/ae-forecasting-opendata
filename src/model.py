"""
Functions to build predictive model of admissions.

TODO:
- Fix estimation of uncertainty (95% CI is too narrow!)
- Show how to build hierarchy (need to load data for different locations)
"""

from typing import Optional, Callable

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS

import jax.numpy as jnp
from jax import Array, random

import plotly.express as px
import plotly.graph_objects as go

from .helpers import get_admissions_data

PROP_TRAIN = 0.75


def admissions_model(timestamps: Array, admissions: Optional[Array] = None) -> None:
    """Builds admissions model using numpyro api

    Args:
        timestamp (Array): Timestamp data (must be numeric type)
        admissions (Optional[Array], optional): Observed admissions data. Defaults to None.
    """
    # Hyper-parameters
    intercept_loc = 1e5
    intercept_scale = 1e4
    gradient_loc = 1e3
    gradient_scale = 1e2
    noise_rate = 10.0

    # Priors
    intercept = numpyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
    gradient = numpyro.sample("gradient", dist.Normal(gradient_loc, gradient_scale))
    admissions_loc = intercept + gradient * timestamps
    admissions_scale = numpyro.sample("noise", dist.Exponential(noise_rate))
    numpyro.sample(
        "admissions", dist.Normal(admissions_loc, admissions_scale), obs=admissions
    )


def plot_model_results(
    x: Array, y_observed: Array, y_predicted: Array, y_hpdi: Array
) -> None:
    fig = px.line(x=x, y=y_predicted)
    extra_traces = [
        go.Scatter(
            x=x,
            y=y_hpdi[0],
            fill=None,
            mode="lines",
            line_color="lightblue",
            name="Lower CI",
        ),
        go.Scatter(
            x=x,
            y=y_hpdi[1],
            fill="tonexty",
            mode="lines",
            line_color="lightblue",
            name="Upper CI",
        ),
        list(px.scatter(x=x, y=y_observed).select_traces()),
    ]

    for traces in extra_traces:
        fig.add_traces(traces)

    y_offset = 0.1  # 10% above/below
    y_min = min(y_observed) - y_offset * abs(max(y_observed))
    y_max = (1 + y_offset) * max(y_observed)
    fig.update_yaxes(range=[y_min, y_max])

    fig.show()


class TimeSeriesModeller:
    def __init__(
        self, model_func: Callable[[Array, Array], Array], predictor_name: str
    ):
        nuts_kernel = NUTS(model_func)
        self.predictor_name = predictor_name
        self._mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
        self._fitted = False

    @staticmethod
    def train_test_split(
        timestamps: Array, predictors: Array
    ) -> tuple[Array, Array, Array, Array]:
        n_train = int(PROP_TRAIN * timestamps.size)
        return (
            timestamps[:n_train],
            timestamps[n_train:],
            predictors[:n_train],
            predictors[n_train:],
        )

    def fit(
        self,
        timestamps_train: Array,
        predictors_train: Array,
    ) -> MCMC:
        rng_key = random.PRNGKey(0)
        mcmc_params = {
            "rng_key": rng_key,
            "timestamps": timestamps_train,
            "extra_fields": (),
            self.predictor_name: predictors_train,  # Named
        }
        self._mcmc.run(**mcmc_params)
        self._fitted = True

    def predict(self, timestamps_test: Array) -> Array:
        if not self._fitted:
            raise AttributeError(
                "TimeSeriesModel hasn't been fitted! Call `.fit()` before `.predict()`"
            )
        samples = self._mcmc.get_samples()
        mx_posterior = jnp.expand_dims(samples["gradient"], -1) * timestamps_test
        c_posterior = jnp.expand_dims(samples["intercept"], -1)
        predictors_posterior = mx_posterior + c_posterior
        return predictors_posterior


if __name__ == "__main__":
    df_admissions = get_admissions_data().loc[lambda df: df["org_code"] == "R0A"]

    timestamps = jnp.arange(len(df_admissions.index))
    admissions = jnp.array(df_admissions["ae_admissions_total"].values)
    timestamps_train, timestamps_test, admissions_train, admissions_test = (
        TimeSeriesModeller.train_test_split(timestamps, admissions)
    )

    model = TimeSeriesModeller(admissions_model, predictor_name="admissions")
    model.fit(timestamps_train, admissions_train)
    # mcmc.print_summary()
    admissions_posterior = model.predict(timestamps_test)

    # Compute empirical posterior distribution over mu
    admissions_pred = jnp.mean(admissions_posterior, axis=0)
    admissions_hpdi = hpdi(admissions_posterior, 0.95)

    plot_model_results(
        x=timestamps_test,
        y_observed=admissions_test,
        y_predicted=admissions_pred,
        y_hpdi=admissions_hpdi,
    )
