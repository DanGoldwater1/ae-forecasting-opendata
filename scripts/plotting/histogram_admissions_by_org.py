from datetime import date

import plotly.express as px

from src.preprocessing import prepare_admissions_data
from src.download_data import download_admissions_data

if __name__ == "__main__":
    df_admissions = download_admissions_data().pipe(prepare_admissions_data)

    # Plot type 1 admissions for January 2021
    plot_dt = date(2021, 1, 1)
    time_col = "source_date"
    location_col = "org_code"
    metric_col = "ae_admissions_total"

    # Build mask
    location_mask = df_admissions[location_col] != "TOTAL"
    time_mask = df_admissions[time_col] == plot_dt
    row_mask = location_mask * time_mask

    df_plot = df_admissions.loc[row_mask].sort_values(
        by=[metric_col], ignore_index=True, ascending=False
    )

    fig = px.bar(
        df_plot,
        x=location_col,
        y=metric_col,
    )
    fig.show()
