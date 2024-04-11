"""
Functions to build predictive model of admissions.

TODO:
- Fix estimation of uncertainty (95% CI is too narrow!)
- Show how to build hierarchy (need to load data for different locations)
"""

from typing import Optional, Callable, Any

import numpyro
import pretty_errors
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import jax.numpy as jnp
from jax import Array, random

import plotly.express as px
import plotly.graph_objects as go

from helpers import get_admissions_data

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
    x: Array,
    y_obs: Array,
    y_pred: Array,
    y_lower: Array,
    y_upper: Array,
    pred_length: int,
) -> None:
    fig = px.scatter(
        x=x,
        y=y_obs,
        color_discrete_sequence=["darkblue"],
    )

    samples = [
        {
            "name": "Insample",
            "slice": slice(0, -pred_length),
            "colours": {"light": "lightblue", "dark": "darkblue"},
        },
        {
            "name": "Outsample",
            "slice": slice(-pred_length, None),
            "colours": {"light": "palevioletred", "dark": "mediumvioletred"},
        },
    ]

    for sample in samples:
        name = sample["name"]
        slice_ = sample["slice"]
        extra_traces = [
            go.Scatter(
                x=x[slice_],
                y=y_pred[slice_],
                fill=None,
                mode="lines",
                line_color=sample["colours"]["dark"],
                name=f"Mean ({name})",
            ),
            go.Scatter(
                x=x[slice_],
                y=y_lower[slice_],
                fill=None,
                mode="lines",
                line_color=sample["colours"]["light"],
                name=f"Lower CI ({name})",
            ),
            go.Scatter(
                x=x[slice_],
                y=y_upper[slice_],
                fill="tonexty",
                mode="lines",
                line_color=sample["colours"]["light"],
                name=f"Upper CI ({name})",
            ),
        ]

        for traces in extra_traces:
            fig.add_traces(traces)

    y_axis_offset = 0.1  # Axis 10% above/below
    y_min = min(y_obs) - y_axis_offset * abs(max(y_obs))
    y_max = (1 + y_axis_offset) * max(y_obs)
    fig.update_yaxes(range=[y_min, y_max])

    fig.show()


class TimeSeriesModeller:
    def __init__(
        self,
        model: Callable[[Array, Array], Array],
        predictor_name: str,
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.hyperparams = hyperparams or {}
        nuts_kernel = NUTS(model)
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
            **self.hyperparams,
        }
        self._mcmc.run(**mcmc_params)
        self._fitted = True

    def predict(self, timestamps_test: Array) -> Array:
        if not self._fitted:
            raise AttributeError(
                "TimeSeriesModel hasn't been fitted! Call `.fit()` before `.predict()`"
            )
        posterior_samples = self._mcmc.get_samples()
        rng_key = random.PRNGKey(1)
        posterior_predictive = Predictive(self.model, posterior_samples)
        predictions = posterior_predictive(
            rng_key, timestamps=timestamps_test, **self.hyperparams
        )
        return predictions

    def print_summary(self):
        self._mcmc.print_summary()


if __name__ == "__main__":
    df_admissions = get_admissions_data().loc[lambda df: df["org_code"] == "R0A"]

    # Train/test split
    df_admissions.sort_values(by=["org_code", "source_date"], inplace=True)
    timestamps = jnp.arange(len(df_admissions.index))
    admissions = jnp.array(df_admissions["ae_admissions_total"].values)
    timestamps_train, timestamps_test, admissions_train, admissions_test = (
        TimeSeriesModeller.train_test_split(timestamps, admissions)
    )

    # Fitting
    modeller = TimeSeriesModeller(admissions_model, predictor_name="admissions")
    modeller.fit(timestamps_train, admissions_train)
    modeller.print_summary()

    # Predictions
    admissions_pred = modeller.predict(timestamps)["admissions"]
    quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
    admissions_pred_quantiles = jnp.quantile(
        admissions_pred, jnp.array(quantiles), axis=0
    )

    df_admissions = df_admissions.assign(
        **{
            f"ae_admissions_predicted_{q}": x
            for q, x in zip(quantiles, admissions_pred_quantiles)
        }
    )

    plot_model_results(
        x=df_admissions["source_date"].values,
        y_obs=df_admissions["ae_admissions_total"],
        y_pred=df_admissions["ae_admissions_predicted_0.5"],
        y_lower=df_admissions["ae_admissions_predicted_0.025"],
        y_upper=df_admissions["ae_admissions_predicted_0.975"],
        pred_length=timestamps_test.size,
    )
