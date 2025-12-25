import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats

# Helper Function for Sen's Slope Estimator
def calculate_sens_slope(x, y):
    if len(x) < 2:
        return 0, 0
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    slope = np.median(slopes)
    intercept = np.median(y - slope * x)
    return slope, intercept

# Loading Data
df = pd.read_csv('data_set.csv', encoding='latin1')

# Extract years
year_columns = [col for col in df.columns if col.startswith('AF_USED_') and col[-4:].isdigit()]
years = sorted([int(col.split('_')[-1]) for col in year_columns])

# Extract Counties
counties = sorted(df['county_abrev'].dropna().unique())

# Extract GMDs (Handling the <Null> values)
gmd_values = sorted([x for x in df['gmd'].unique() if str(x) != '<Null>'])
gmd_options = [{"label": "All GMDs", "value": "all"}] + \
              [{"label": f"GMD {g}", "value": g} for g in gmd_values] + \
              [{"label": "Outside GMD", "value": "outside"}]

# Calculate max possible total for the default range filter
max_possible_total = round(df[year_columns].sum(axis=1).max(), 2)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Needed for deployment on Render

app.layout = dbc.Container([
    dbc.Row([
        # --- LEFT SIDEBAR ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Filter Options")),
                dbc.CardBody([
                    html.Label("Select Year Range:", className="fw-bold mt-2"),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=min(years), max=max(years),
                        value=[min(years), max(years)],
                        marks={y: str(y) for y in range(min(years), max(years)+1, 5)},
                        step=1, tooltip={"always_visible": True}
                    ),
                    
                    html.Label("Select County:", className="fw-bold mt-4"),
                    dcc.Dropdown(
                        id="county-dropdown",
                        options=[{"label": "All Counties", "value": "all"}] + 
                                [{"label": c, "value": c} for c in counties],
                        value="all"
                    ),

                    html.Label("Select GMD:", className="fw-bold mt-4"),
                    dcc.Dropdown(
                        id="gmd-dropdown",
                        options=gmd_options,
                        value="all",
                        clearable=False
                    ),

                    html.Label("Water Volume Range (AF):", className="fw-bold mt-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Min", className="small text-muted"),
                            dcc.Input(id="min-pumping-volume", type="number", value=0, className="form-control")
                        ]),
                        dbc.Col([
                            html.Label("Max", className="small text-muted"),
                            dcc.Input(id="max-pumping-volume", type="number", value=max_possible_total, className="form-control")
                        ]),
                    ])
                ])
            ])
        ], width=3),

        # --- RIGHT CHARTS ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Geospatial Visualization")),
                dbc.CardBody(dcc.Graph(id="map-graph"))
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader(html.H5(f"Annual Water Pumping Trend ({min(years)}-{max(years)})")),
                dbc.CardBody(dcc.Graph(id="trend-graph"))
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader(html.H5("Trend & Summary Statistics")),
                dbc.CardBody(id="summary_stats")
            ]),
        ], width=9)
    ])
], fluid=True)

# Callback to disable GMD dropdown if a specific county is selected
@app.callback(
    [Output("gmd-dropdown", "disabled"),
     Output("gmd-dropdown", "value")],
    Input("county-dropdown", "value")
)
def toggle_gmd_options(selected_county):
    if selected_county != "all":
        return True, "all"
    return False, no_update

# MAIN VISUALIZATION CALLBACK
@app.callback(
    [Output("map-graph", "figure"),
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

    filtered_df = df.copy()
    
    # Defaults
    map_center = {"lat": 38.5, "lon": -98.0}
    map_zoom = 6.5
    map_title = f"Water Usage - All Counties ({start_year}-{end_year})"

    # Filter Logic
    if selected_county != "all":
        filtered_df = filtered_df[filtered_df["county_abrev"] == selected_county]
        if not filtered_df.empty:
            map_center = {"lat": filtered_df["lat_nad83"].mean(), "lon": filtered_df["long_nad83"].mean()}
            map_zoom = 9 
            full_name = filtered_df["county_name"].iloc[0] if "county_name" in filtered_df.columns else selected_county
            map_title = f"Water Usage in {full_name} County ({start_year}-{end_year})"
    elif selected_gmd != "all":
        if selected_gmd == "outside":
            filtered_df = filtered_df[filtered_df["gmd"] == "<Null>"]
            map_title = f"Water Usage - Outside GMD ({start_year}-{end_year})"
        else:
            filtered_df = filtered_df[filtered_df["gmd"] == selected_gmd]
            map_title = f"Water Usage - GMD {selected_gmd} ({start_year}-{end_year})"
            
        if not filtered_df.empty:
            map_center = {"lat": filtered_df["lat_nad83"].mean(), "lon": filtered_df["long_nad83"].mean()}
            map_zoom = 7.5

    # Calculate Total & Volume Filter
    filtered_df['Period_Total'] = filtered_df[selected_year_cols].sum(axis=1)
    min_v = min_volume if min_volume is not None else 0
    max_v = max_volume if max_volume is not None else filtered_df['Period_Total'].max()
    filtered_df = filtered_df[(filtered_df['Period_Total'] >= min_v) & (filtered_df['Period_Total'] <= max_v)]

    # Log Scale for colors
    filtered_df['Log_Total'] = np.log10(filtered_df['Period_Total'] + 1)

    # Map Figure
    map_fig = px.scatter_mapbox(
        filtered_df, 
        lat="lat_nad83", 
        lon="long_nad83", 
        color="Log_Total",
        zoom=map_zoom,
        center=map_center,
        mapbox_style="open-street-map", 
        title=map_title,
        color_continuous_scale="Jet", 
        size_max=15,
        hover_data={
            "county_name": True,
            "Period_Total": ":,.0f",
            "Log_Total": False,
            "lat_nad83": False,
            "long_nad83": False
        }
    )
    
    map_fig.update_layout(
        height=600, margin={"r":0, "t":50, "l":0, "b":0},
        title_font_size=20,
        coloraxis_colorbar=dict(
            title="Total Pumping (AF)",
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=["1", "10", "100", "1k", "10k", "100k", "1M"],
            tickmode="array"
        )
    )

    # Trend Analysis
    trend_data = filtered_df[selected_year_cols].sum().reset_index()
    trend_data.columns = ["Year", "Total_Pumping"]
    trend_data["Year_Int"] = trend_data["Year"].str.extract('(\d+)').astype(int)
    
    x = trend_data["Year_Int"].values
    y = trend_data["Total_Pumping"].values

    if len(x) > 1:
        slope, intercept = calculate_sens_slope(x, y)
        trend_line = slope * x + intercept
        tau, p_value = stats.kendalltau(x, y)
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
    else:
        slope, intercept, p_value, significance, trend_direction = 0, 0, 1.0, "N/A", "N/A"
        trend_line = y

    # Trend Graph with Title
    trend_fig = px.line(
        trend_data, x="Year_Int", y="Total_Pumping", markers=True, labels={'Total_Pumping':"Total Pumping", 'Year_Int':"Year"}
     
    )
    if len(x) > 1:
        trend_fig.add_trace(go.Scatter(
            x=x, y=trend_line, mode='lines', name="Sen's Slope",
            line=dict(color='black', width=3, dash='dash')
            
        ))

    # Summary Stats
    summary = html.Div([
        html.P(f"Trend Analysis: {trend_direction} ({significance})", className="fw-bold mb-1"),
        html.Ul([
            html.Li(f"Sen's Slope: {slope:.2f} AF/year"),
            html.Li(f"Mann-Kendall p-value: {p_value:.4f}"),
            html.Li(f"Total Pumping (Selected Period): {filtered_df['Period_Total'].sum():,.2f} AF"),
            html.Li(f"Active Pumping Points: {len(filtered_df)}")
        ])
    ])

    return map_fig, trend_fig, summary

if __name__ == '__main__':
    app.run(debug=True, port=8052)
