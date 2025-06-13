

import fastapi
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import json
import plotly.express as px
import plotly.io as pio
from train_and_save_model import create_dataset # Import to get data for graphs

# --- App Initialization ---
app = fastapi.FastAPI()

# Mount static files (for CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# --- Load Model and Metadata at Startup ---
try:
    model = joblib.load("model/solar_production_model.joblib")
    with open("model/model_metadata.json", 'r') as f:
        metadata = json.load(f)
    print("Model and metadata loaded successfully.")
except FileNotFoundError:
    print("Error: Model or metadata not found. Please run 'train_and_save_model.py' first.")
    model = None
    metadata = {"categorical_options": {}} # Default to avoid crash

# --- Generate EDA Graphs ---
# We use the same data generation function to create sample data for visualizations
df_for_graphs = create_dataset()

# Correlation Heatmap
corr_matrix = df_for_graphs.corr(numeric_only=True)
fig_corr = px.imshow(corr_matrix, 
                     text_auto=True, 
                     aspect="auto", 
                     color_continuous_scale='YlOrRd',
                     title="Feature Correlation Heatmap")
fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
correlation_plot_div = pio.to_html(fig_corr, full_html=False, include_plotlyjs='cdn')

# Production by Region Box Plot
fig_region = px.box(df_for_graphs, 
                    x='Region', 
                    y='Annual_Energy_Production_GWh', 
                    color='Region',
                    title="Energy Production Distribution by Region")
fig_region.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
region_plot_div = pio.to_html(fig_region, full_html=False, include_plotlyjs='cdn')


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serves the main page with the form and graphs."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metadata": metadata,
        "prediction_gwh": None,
        "correlation_plot_div": correlation_plot_div,
        "region_plot_div": region_plot_div
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    System_Size_MW: float = Form(...),
    Solar_Irradiance_kWh_m2_year: float = Form(...),
    Panel_Efficiency_pct: float = Form(...),
    Developer: str = Form(...),
    Region: str = Form(...),
    Equipment_Type: str = Form(...)
):
    """Handles form submission, makes a prediction, and re-renders the page with the result."""
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request, "metadata": metadata, "prediction_gwh": "Model not loaded!"
        })

    # Create a DataFrame from the form data
    # Use average values for features not in the form
    input_data = {
        'System_Size_MW': System_Size_MW,
        'Solar_Irradiance_kWh_m2_year': Solar_Irradiance_kWh_m2_year,
        'Panel_Efficiency_pct': Panel_Efficiency_pct,
        'Shading_Factor': 0.975, # Average of 0.95 and 1.0
        'Average_Temperature_C': 20, # Average of 10 and 30
        'Developer': Developer,
        'Region': Region,
        'Equipment_Type': Equipment_Type
    }
    input_df = pd.DataFrame([input_data])
    
    # Ensure column order matches the training data
    input_df = input_df[metadata['columns']]

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Re-render the page with the prediction result
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metadata": metadata,
        "prediction_gwh": prediction,
        "correlation_plot_div": correlation_plot_div,
        "region_plot_div": region_plot_div
    })

