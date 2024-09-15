import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

# Load the Home Credit Default Risk dataset (assuming it's stored as a CSV)
df = pd.read_csv('application_train.csv')  # Update with your actual dataset path

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Credit Score Dashboard", style={'text-align': 'center', 'color': '#2c3e50'}),

    # Overall statistics
    html.Div(id='overall-statistics', style={'backgroundColor': '#ecf0f1', 'padding': '20px'}),

    # Client-specific prediction section
    html.Div([
        html.H3("Client-Specific Prediction", style={'color': '#2980b9'}),
        html.Label("Client ID"),
        dcc.Input(id='sk_id_curr', type='number', placeholder="Enter Client ID", value=None),
        html.Br(),

        html.Button('Predict', id='predict-button', n_clicks=0, style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 20px'}),
        html.Div(id='prediction-result'),
        html.Div(id='client-overview')
    ], style={'backgroundColor': '#f5f6fa', 'padding': '20px', 'margin': '20px 0'})
])

# Overall Statistics Callback
@app.callback(
    Output('overall-statistics', 'children'),
    [Input('sk_id_curr', 'value')]
)
def update_overall_statistics(sk_id_curr):
    # Display overall statistics
    loan_distribution = dcc.Graph(figure=create_loan_distribution(), style={'border': '1px solid #bdc3c7', 'borderRadius': '10px'})
    default_rate = dcc.Graph(figure=create_default_rate(), style={'border': '1px solid #bdc3c7', 'borderRadius': '10px'})
    gender_distribution = dcc.Graph(figure=create_gender_distribution(), style={'border': '1px solid #bdc3c7', 'borderRadius': '10px'})
    income_distribution = dcc.Graph(figure=create_income_distribution(), style={'border': '1px solid #bdc3c7', 'borderRadius': '10px'})

    # Layout of overall statistics
    overall_statistics = html.Div([
        html.H3("Overall Dataset Statistics", style={'color': '#16a085'}),
        html.Div([
            html.Div([loan_distribution], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([default_rate], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'}),
        html.Div([
            html.Div([gender_distribution], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([income_distribution], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'})
    ])

    return overall_statistics

# Client-Specific Prediction Callback
@app.callback(
    [Output('client-overview', 'children'),
     Output('prediction-result', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('sk_id_curr', 'value')]
)
def update_client_prediction(n_clicks, sk_id_curr):
    if n_clicks > 0 and sk_id_curr:
        # Create payload for API request
        payload = {'SK_ID_CURR': sk_id_curr}

        # Make request to Flask API
        try:
            response = requests.post('http://127.0.0.1:5000/predict', json=payload)
            response.raise_for_status()
            data = response.json()
            prediction = data.get('prediction', 'Error')
            prob = data.get('probability', 'N/A')

            # Display client-specific overview
            client_overview = html.Div([
                html.H3(f"Client Overview: {sk_id_curr}"),
                html.P(f"Prediction Probability: {prob}")
            ])
            return client_overview, f"Prediction: {prediction}"
        except requests.exceptions.RequestException as e:
            return [html.Div(f'Error: {str(e)}')], []

    return [html.Div(), html.Div()]

# Helper Functions to Create Figures
# Loan Amount Distribution
def create_loan_distribution():
    fig = px.histogram(df, x='AMT_CREDIT', title='Loan Amount Distribution', color_discrete_sequence=['#e67e22'])
    fig.update_layout(template='plotly_dark', title_font=dict(size=20, color='#e74c3c'))
    return fig

# Default Rate Pie Chart
def create_default_rate():
    default_rate = df['TARGET'].value_counts(normalize=True)
    fig = px.pie(values=default_rate.values, names=['No Default', 'Default'], title='Default Rate', color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(template='plotly_dark', title_font=dict(size=20, color='#27ae60'))
    return fig

# Gender Distribution Pie Chart
def create_gender_distribution():
    fig = px.pie(df, names='CODE_GENDER', title='Gender Distribution', color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(template='plotly_dark', title_font=dict(size=20, color='#3498db'))
    return fig

# Income Type Distribution
def create_income_distribution():
    fig = px.histogram(df, x='AMT_INCOME_TOTAL', title='Income Distribution', nbins=50, color_discrete_sequence=['#9b59b6'])
    fig.update_layout(template='plotly_dark', title_font=dict(size=20, color='#f1c40f'))
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
