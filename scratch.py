import sys
print(sys.path)
import model
import jax.numpy as jnp

if __name__ == "__main__":
    df_admissions = model.get_admissions_data().loc[lambda df: df["org_code"] == "R0A"]

    # Train/test split
    df_admissions.sort_values(by=["org_code", "source_date"], inplace=True)
    timestamps = jnp.arange(len(df_admissions.index)) #Create linspaced timestamps
    admissions = jnp.array(df_admissions["ae_admissions_total"].values) #convert these to jnp array
    timestamps_train, timestamps_test, admissions_train, admissions_test = (
        model.TimeSeriesModeller.train_test_split(timestamps, admissions)
    )

    # Fitting
    modeller = model.TimeSeriesModeller(model.admissions_model, predictor_name="admissions")
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

    model.plot_model_results(
        x=df_admissions["source_date"].values,
        y_obs=df_admissions["ae_admissions_total"],
        y_pred=df_admissions["ae_admissions_predicted_0.5"],
        y_lower=df_admissions["ae_admissions_predicted_0.025"],
        y_upper=df_admissions["ae_admissions_predicted_0.975"],
        pred_length=timestamps_test.size,
    )
