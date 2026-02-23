

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import geopandas as gpd

# -----------------------
# LINKS (EDIT THESE)
# -----------------------
KGS_WIMAS_URL = "https://geohydro.kgs.ku.edu/geohydro/wimas/"
# Put your lab page here (do NOT leave as placeholder if you want the link to work)
PANTHI_LAB_URL = "https://jeebanpanthi.com/"  # <-- replace with the real URL


# Optional: True Mann–Kendall test
try:
    import pymannkendall as mk  # pip install pymannkendall
    HAS_MK = True
except Exception:
    HAS_MK = False


# -----------------------
# Helper: Sen's slope
# -----------------------
def calculate_sens_slope(x, y):
    if len(x) < 2:
        return 0.0, 0.0
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0, float(np.median(y)) if len(y) else 0.0
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


# -----------------------
# Helper: Watermark
# -----------------------
def add_watermark(fig, text="Water Data App"):
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=40, color="rgba(150, 150, 150, 0.3)"),
        textangle=-30
    )
    return fig


# -----------------------
# Helper: Empty figure
# -----------------------
def empty_figure(message="No data for selected filters", height=450):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        annotations=[dict(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )]
    )
    return fig


# -----------------------
# 1) Load data
# -----------------------
df = pd.read_csv("data_set.csv", encoding="latin1")

# Extract years from AF_USED_#### columns
year_columns = [c for c in df.columns if c.startswith("AF_USED_") and c[-4:].isdigit()]
years = sorted([int(c.split("_")[-1]) for c in year_columns])

if not years:
    raise ValueError("No AF_USED_#### year columns found in data_set.csv.")

# Ensure AF_USED_* are numeric to avoid string concat in sums
df[year_columns] = df[year_columns].apply(pd.to_numeric, errors="coerce")

# Extract Counties
counties = sorted(df["county_abrev"].dropna().unique())

# Extract GMD values (handle <Null>)
gmd_values = sorted([x for x in df["gmd"].unique() if str(x) != "<Null>"])
gmd_options = (
    [{"label": "All Entire Kansas", "value": "all"}] +
    [{"label": f"GMD {g}", "value": g} for g in gmd_values] +
    [{"label": "Outside GMD", "value": "outside"}]
)

# Compute max possible average for default max filter
max_possible_avg = round(df[year_columns].replace(0, np.nan).mean(axis=1).max(), 2)
if pd.isna(max_possible_avg):
    max_possible_avg = 1000


# -----------------------
# 2) Load & process shapefile
# -----------------------
try:
    gdf = gpd.read_file("Groundwater_Management_Districts_(GMD).shp")
    gdf = gdf.to_crs(epsg=4326)  # WGS84 for plotly
    print("Shapefile loaded successfully.")
except Exception as e:
    print(f"Error loading shapefile: {e}")
    gdf = None


# -----------------------
# 3) Define GMD colors
# -----------------------
gmd_colors = {
    "1": "#1f77b4",  # Blue
    "2": "#ff7f0e",  # Orange
    "3": "#2ca02c",  # Green
    "4": "#d62728",  # Red
    "5": "#9467bd",  # Purple
}
default_color = "#7f7f7f"  # Gray


# -----------------------
# 4) App layout
# -----------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        # LEFT SIDEBAR
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Filter Options")),
                dbc.CardBody([
                    html.Label("Select Year Range:", className="fw-bold mt-2"),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=min(years), max=max(years),
                        value=[min(years), max(years)],
                        marks={y: str(y) for y in range(min(years), max(years) + 1, 5)},
                        step=1,
                        tooltip={"always_visible": True}
                    ),

                    html.Label("Select County:", className="fw-bold mt-4"),
                    dcc.Dropdown(
                        id="county-dropdown",
                        options=[{"label": "All Counties", "value": "all"}] +
                                [{"label": c, "value": c} for c in counties],
                        value="all"
                    ),

                    html.Label("Select Groundwater Management District (GMD):", className="fw-bold mt-4"),
                    dcc.Dropdown(
                        id="gmd-dropdown",
                        options=gmd_options,
                        value="all",
                        clearable=False
                    ),

                    html.Label("Avg Water Volume Range (AF/yr):", className="fw-bold mt-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Min", className="small text-muted"),
                            dcc.Input(
                                id="min-pumping-volume",
                                type="number",
                                value=0,
                                className="form-control"
                            )
                        ]),
                        dbc.Col([
                            html.Label("Max", className="small text-muted"),
                            dcc.Input(
                                id="max-pumping-volume",
                                type="number",
                                value=max_possible_avg,
                                className="form-control"
                            )
                        ]),
                    ]),

                    # ---- INFO SECTION ----
                    html.Hr(className="my-4"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.P(
                                        "Water-use and pumping data are obtained from publicly available records "
                                        "provided by the Kansas Geological Survey (KGS) Water Information Management "
                                        "and Analysis System (WIMAS)."
                                    ),
                                    html.P(
                                        [
                                            "KGS WIMAS portal: ",
                                            html.A(
                                                KGS_WIMAS_URL,
                                                href=KGS_WIMAS_URL,
                                                target="_blank",
                                                rel="noopener noreferrer"
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        "Reported pumping volumes are available at an annual time scale and primarily "
                                        "represent irrigation-season withdrawals. Questions regarding data collection "
                                        "methods, reporting requirements, or data access should be directed to KGS WIMAS."
                                    ),
                                ],
                                title="Data source",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P(
                                        "This web portal is designed to visualize groundwater pumping across the state of Kansas."
                                    ),
                                    html.P(
                                        "In addition to interactive visualization, we analyze temporal trends with Sen’s "
                                        "slope estimator and assess their statistical significance using the Mann–Kendall "
                                        "test to evaluate the long-term changes in groundwater use."
                                    ),
                                    html.P(
                                        "Note: If the Mann–Kendall library is not installed, the app falls back to Kendall "
                                        "tau for a similar monotonic-trend significance check."
                                    ),
                                ],
                                title="Data analysis",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P(
                                        "Users can explore pumping data at multiple spatial scales, including statewide, "
                                        "Groundwater Management Districts (GMDs), and counties."
                                    ),
                                    html.P(
                                        "For any selected area, the displayed trend represents the cumulative pumping "
                                        "from all wells within that area."
                                    ),
                                    html.P(
                                        "Users can customize the visualization by selecting a start year and end year "
                                        "and by filtering wells based on their average pumping rates."
                                    ),
                                ],
                                title="How to use the portal",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P(
                                        "The data visualization component was led by Govinda Khanal under the mentorship of Jeeban Panthi."
                                    ),
                                    html.P([
                                        "The work was conducted at the ",
                                        html.A(
                                            "Panthi HydroSystem Lab",
                                            href=PANTHI_LAB_URL,
                                            target="_blank",
                                            rel="noopener noreferrer",
                                            style={"textDecoration": "underline"}
                                        ),
                                        " at Kansas State University, where we focus on groundwater quantity and quality "
                                        "research using integrated geophysical methods, remote sensing, and numerical modeling."
                                    ]),
                                ],
                                title="Our team",
                            ),
                        ],
                        start_collapsed=True,
                        flush=True,
                        always_open=False,
                    ),
                ])
            ])
        ], width=3),

        # RIGHT SIDE
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Geospatial Visualization of Groundwater Pumping in Kansas")),
                dbc.CardBody([
                    dcc.Graph(id="map-graph"),
                    html.Div(id="gmd-legend", className="d-flex justify-content-center mt-2 flex-wrap")
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader(html.H5(f"Annual Total Pumping Trend ({min(years)}-{max(years)})")),
                dbc.CardBody([
                    dcc.Graph(id="trend-graph"),
                    html.Hr(),
                    html.H6("Summary Statistics", className="fw-bold"),
                    html.Div(id="summary_stats")
                ])
            ], className="mb-3"),
        ], width=9)
    ])
], fluid=True)


# Disable GMD dropdown if a specific county is selected
@app.callback(
    [Output("gmd-dropdown", "disabled"),
     Output("gmd-dropdown", "value")],
    Input("county-dropdown", "value")
)
def toggle_gmd_options(selected_county):
    if selected_county != "all":
        return True, "all"
    return False, no_update


# -----------------------
# Main visualization callback
# -----------------------
@app.callback(
    [Output("map-graph", "figure"),
     Output("gmd-legend", "children"),
     Output("trend-graph", "figure"),
     Output("summary_stats", "children")],
    [Input("year-range-slider", "value"),
     Input("county-dropdown", "value"),
     Input("gmd-dropdown", "value"),
     Input("min-pumping-volume", "value"),
     Input("max-pumping-volume", "value")]
)
def update_visualizations(year_range, selected_county, selected_gmd, min_volume, max_volume):
    start_year, end_year = year_range
    selected_year_cols = [f"AF_USED_{y}" for y in range(start_year, end_year + 1)]

    # Guard in case of missing columns
    missing_cols = [c for c in selected_year_cols if c not in df.columns]
    if missing_cols:
        map_fig = empty_figure("Selected years not found in dataset", height=600)
        trend_fig = empty_figure("Selected years not found in dataset", height=400)
        return map_fig, [], trend_fig, html.Div("Selected years not found in dataset.")

    # --- Base filtering (county/GMD only) ---
    base_df = df.copy()

    # Defaults
    map_center = {"lat": 38.5, "lon": -98.0}
    map_zoom = 6.5
    map_title = f"Average Water Use per year - All Counties ({start_year}-{end_year})"

    # Apply county/GMD filters
    if selected_county != "all":
        base_df = base_df[base_df["county_abrev"] == selected_county]
        if not base_df.empty:
            map_center = {"lat": float(base_df["lat_nad83"].mean()),
                          "lon": float(base_df["long_nad83"].mean())}
            map_zoom = 9
            full_name = base_df["county_name"].iloc[0] if "county_name" in base_df.columns else selected_county
            map_title = f"Avg Water Usage in {full_name} County ({start_year}-{end_year})"
    elif selected_gmd != "all":
        if selected_gmd == "outside":
            base_df = base_df[base_df["gmd"] == "<Null>"]
            map_title = f"Avg Water Usage - Outside GMD ({start_year}-{end_year})"
        else:
            base_df = base_df[base_df["gmd"] == selected_gmd]
            map_title = f"Avg Water Usage - GMD {selected_gmd} ({start_year}-{end_year})"

        if not base_df.empty:
            map_center = {"lat": float(base_df["lat_nad83"].mean()),
                          "lon": float(base_df["long_nad83"].mean())}
            map_zoom = 7.5

    # If no base data, return empties immediately
    if base_df.empty:
        map_fig = empty_figure("No data for selected county/GMD filters", height=600)
        trend_fig = empty_figure("No data for selected county/GMD filters", height=400)
        return map_fig, [], trend_fig, html.Div("No data available for the selected county/GMD filters.")

    # -----------------------
    # TREND (computed from base_df; NOT affected by avg-volume filter)
    # -----------------------
    trend_series = base_df[selected_year_cols].sum(axis=0, skipna=True)
    trend_data = trend_series.reset_index()
    trend_data.columns = ["Year", "Total_Pumping"]
    trend_data["Year_Int"] = trend_data["Year"].str.extract(r"(\d+)").astype(int)

    x = trend_data["Year_Int"].to_numpy()
    y = trend_data["Total_Pumping"].to_numpy()

    if len(x) > 1:
        slope, intercept = calculate_sens_slope(x, y)
        trend_line = slope * x + intercept

        if HAS_MK:
            mk_res = mk.original_test(y)
            p_value = float(mk_res.p)
            if mk_res.trend == "increasing":
                trend_direction = "Increasing"
            elif mk_res.trend == "decreasing":
                trend_direction = "Decreasing"
            else:
                trend_direction = "Stable"
            test_label = "Mann–Kendall p"
        else:
            _, p_value = stats.kendalltau(x, y)
            test_label = "Kendall tau p"
            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"

        significance = "Significant" if p_value < 0.05 else "Not Significant"
    else:
        slope, intercept = 0.0, 0.0
        trend_line = y
        p_value = 1.0
        test_label = "p"
        significance = "N/A"
        trend_direction = "N/A"

    trend_fig = px.line(
        trend_data, x="Year_Int", y="Total_Pumping", markers=True,
        labels={"Total_Pumping": "Total Pumping (AF)", "Year_Int": "Year"}
    )

    if len(x) > 1:
        trend_fig.add_trace(go.Scatter(
            x=x, y=trend_line, mode="lines", name="Sen's Slope",
            line=dict(color="black", width=3, dash="dash")
        ))

    trend_fig.update_layout(
        height=400,
        width=900,
        margin={"r": 0, "t": 10, "l": 0, "b": 0}
    )
    add_watermark(trend_fig, "Panthi HydroSystems Lab")

    # -----------------------
    # MAP (computed from map_df; includes avg-volume filter)
    # -----------------------
    map_df = base_df.copy()
    map_df["Period_Avg"] = map_df[selected_year_cols].replace(0, np.nan).mean(axis=1)

    # Remove NaN Period_Avg (e.g., all zeros over selected years)
    map_df = map_df.dropna(subset=["Period_Avg"])

    # Apply avg-volume filter safely
    min_v = 0.0 if min_volume is None else float(min_volume)
    max_v = float(map_df["Period_Avg"].max()) if max_volume is None else float(max_volume)

    map_df = map_df[(map_df["Period_Avg"] >= min_v) & (map_df["Period_Avg"] <= max_v)]

    # If map becomes empty after avg-volume filter
    if map_df.empty:
        map_fig = empty_figure("No wells match the avg-volume range filter", height=600)
        legend_items = []
        summary = html.Div([
            dbc.Row([
                dbc.Col(html.P(f"Trend: {trend_direction} ({significance})", className="fw-bold text-primary")),
                dbc.Col(html.P(f"Sen's Slope: {slope:.2f} AF/year")),
                dbc.Col(html.P(f"{test_label}: {p_value:.4f}")),
                dbc.Col(html.P(f"Total Volume (trend base): {base_df[selected_year_cols].sum().sum():,.0f} AF")),
                dbc.Col(html.P(f"Active Wells (trend base): {len(base_df)}")),
            ])
        ])
        return map_fig, legend_items, trend_fig, summary

    # Log scale for color
    map_df["Log_Avg"] = np.log10(map_df["Period_Avg"] + 1)

    map_fig = px.scatter_mapbox(
        map_df,
        lat="lat_nad83",
        lon="long_nad83",
        color="Log_Avg",
        zoom=map_zoom,
        center=map_center,
        mapbox_style="open-street-map",
        title=map_title,
        color_continuous_scale="Jet",
        size_max=15,
        hover_data={
            "county_name": True,
            "Period_Avg": ":,.2f",
            "Log_Avg": False,
            "lat_nad83": False,
            "long_nad83": False
        }
    )

    # --- Add shapefile boundaries
    if gdf is not None:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if geom.geom_type == "Polygon":
                x_coords, y_coords = geom.exterior.xy
                xs, ys = [list(x_coords)], [list(y_coords)]
            elif geom.geom_type == "MultiPolygon":
                xs, ys = [], []
                for part in geom.geoms:
                    x_coords, y_coords = part.exterior.xy
                    xs.append(list(x_coords))
                    ys.append(list(y_coords))
            else:
                continue

            gmd_id_val = str(row.get("GMD_ID", row.get("GMD_", "")))
            gmd_number = "".join(filter(str.isdigit, gmd_id_val))
            current_color = gmd_colors.get(gmd_number, default_color)

            # Highlight selection safely (no substring match issues)
            is_selected = (selected_gmd not in ["all", "outside"]) and (str(selected_gmd) == str(gmd_number))
            line_width = 6 if is_selected else 4

            for x_part, y_part in zip(xs, ys):
                map_fig.add_trace(go.Scattermapbox(
                    lat=y_part,
                    lon=x_part,
                    mode="lines",
                    line=dict(width=line_width, color=current_color),
                    name=f"GMD {gmd_number}",
                    showlegend=False,
                    hoverinfo="text",
                    text=f"GMD {gmd_number} Boundary"
                ))

    # Colorbar ticks that match Log_Avg = log10(value+1)
    vals = [1, 10, 100, 1000, 10000, 100000, 1000000]
    tickvals = [float(np.log10(v + 1)) for v in vals]
    ticktext = ["1", "10", "100", "1k", "10k", "100k", "1M"]

    map_fig.update_layout(
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title_font_size=20,
        coloraxis_colorbar=dict(
            title="Avg Pumping (AF/yr)",
            tickvals=tickvals,
            ticktext=ticktext,
            tickmode="array"
        )
    )
    add_watermark(map_fig, "Panthi Hydrology Lab")

    # --- Legend (static)
    legend_items = []
    for gmd_num, color in gmd_colors.items():
        legend_items.append(
            html.Div([
                html.Span(style={
                    "display": "inline-block",
                    "width": "15px",
                    "height": "15px",
                    "backgroundColor": color,
                    "borderRadius": "50%",
                    "marginRight": "5px",
                    "verticalAlign": "middle"
                }),
                html.Span(f"GMD {gmd_num}", style={"marginRight": "15px", "fontSize": "14px"})
            ], className="d-inline-flex align-items-center")
        )

    summary = html.Div([
        dbc.Row([
            dbc.Col(html.P(f"Trend: {trend_direction} ({significance})", className="fw-bold text-primary")),
            dbc.Col(html.P(f"Sen's Slope: {slope:.2f} AF/year")),
            dbc.Col(html.P(f"{test_label}: {p_value:.4f}")),
            dbc.Col(html.P(f"Total Volume (trend base): {base_df[selected_year_cols].sum().sum():,.0f} AF")),
            dbc.Col(html.P(f"Active Wells (trend base): {len(base_df)}")),
        ]),
        html.Small(
            "Note: Map points may be fewer than 'Active Wells (trend base)' due to the Avg Volume Range filter.",
            className="text-muted"
        )
    ])

    return map_fig, legend_items, trend_fig, summary


if __name__ == "__main__":
    app.run(debug=True, port=8052)
