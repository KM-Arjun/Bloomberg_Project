import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn, optim
from torch import distributions as D
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestRegressor
import pandas_datareader.data as web
import datetime as dt

class ConditionalNet(torch.nn.Module):
    def __init__(self, lenX, layerCounts, hidden_size):
        super().__init__()
        self.lenX = lenX
        self.layerCounts = layerCounts
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.lenX+self.layerCounts, hidden_size),# we increased this to account for the embedding output
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        return self.layers(x)

class ConditionalModel(pl.LightningModule):
    def __init__(self, condNet, embedLayers, margin):
        super().__init__()
        self.condNet = condNet
        self.embed_layers = embedLayers
        self.margin = margin

    def training_step(self, batch, batch_idx):
        x, categorical_map, y = batch
        cond_emb = torch.stack([self.embed_layers[k](v) for k, v in categorical_map.items()],-1).sum(-1).squeeze(-2)
        full_cond = torch.concat((x.squeeze(),cond_emb),-1)
        dist_params = self.condNet(full_cond)
        dist_params = dist_params.chunk(2, -1)
        _mus = dist_params[0]
        _stds = torch.nn.Softplus()(dist_params[1]) + self.margin
        return -D.Normal(_mus, _stds).log_prob(y).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001) ## hyperparameterize
        return optimizer




df = pd.read_csv("Industry.csv")
df = df.iloc[: , 1:]
modelFile = open('modelNN.pickle', 'rb')
stateDict = pickle.load(modelFile)
indDictFile = open('indDict.pickle', 'rb')
indDict = pickle.load(indDictFile)


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "21.5rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


sidebar = html.Div(
    [
        html.H2("Input Variables", className="lead"),
        html.Hr(),

        dbc.Nav(
            [
                html.P("Revenue", className="text-left text-body mb-0" ),
                dcc.Input(id='username1', placeholder='Enter a value...', value=10000, type='number'),
                html.Br(),
                html.P("Industry", className="text-left text-body mb-0" ),
                dcc.Dropdown(id='username2', placeholder='Pick an industry', value='Waste Management',
                             options=[{'label': x, 'value': x}
                                      for x in sorted(df['Industry'].unique())]),
                html.Br(),
                html.P("Net Income", className="text-left text-body mb-0" ),
                dcc.Input(id='username3', placeholder='Enter a value...', value=10000, type='number'),
                html.Br(),
                html.P("Market Cap", className="text-left text-body mb-0" ),
                dcc.Input(id='username4', placeholder='Enter a value...', value=10000, type='number'),
                html.Br(),
                html.P("Cash Flow", className="text-left text-body mb-0" ),
                dcc.Input(id='username5', placeholder='Enter a value...', value=10000, type='number'),
                html.Br(),
                html.P("Emission", className="text-left text-body mb-0" ),
                dcc.Input(id='username6', placeholder='Enter a value...', value=10000, type='number'),
                html.Br(),



            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = html.Div(children = [
                dbc.Row([
                    dbc.Col(),

                    dbc.Col(html.H1('Log Intensity Distribution'),width = 9, style = {'margin-left':'7px','margin-top':'7px'})
                    ]),
                dbc.Row(
                    [dbc.Col(sidebar),
                    dbc.Col(dcc.Graph(id='line-fig2', figure={}), width = 9, style = {'margin-left':'15px', 'margin-top':'7px', 'margin-right':'15px'})
                    ])
    ]
)

@app.callback(
    # Output('line-fig', 'figure'),
    Output('line-fig2', 'figure'),
    # Input('my-dpdn', 'value'),
    Input('username1', 'value'),
    Input('username2', 'value'),
    Input('username3', 'value'),
    Input('username4', 'value'),
    Input('username5', 'value'),
    Input('username6', 'value')
)

def update_graph(input_value1, input_value2, input_value3, input_value4, input_value5, input_value6):
    given = True
    # User Input
    NetIncome = float(input_value3)
    Revenue = float(input_value1)
    MarketCap = float(input_value4)
    CashFlow = float(input_value5)
    Emission = float(input_value6)
    Industry = str(input_value2)

    ## calculate as part of the interface code
    E2R = np.log(100 * (Emission / Revenue))
    NI2R = 100 * (NetIncome / Revenue)
    MC2R = 100 * (MarketCap / Revenue)
    CF2R = 100 * (CashFlow / Revenue)
    CatX = np.array(Industry)
    ContX = np.array([NI2R, MC2R, CF2R, E2R])
    margin = 1e-2
    embed_layers = nn.ModuleDict({
        "Industry": nn.Embedding(max(indDict.values()) + 1, 16)
    })

    model = ConditionalModel(ConditionalNet(4, 16, 128), embed_layers, 1e-2)

    model.load_state_dict(stateDict.state_dict())
    model.eval()

    with torch.no_grad():
        x = torch.Tensor(ContX)
        categorical_map = {
            'Industry': torch.Tensor(np.array([indDict[Industry]])).long()  # .squeeze()
        }
        cond_emb = torch.stack([model.embed_layers[k](v) for k, v in categorical_map.items()], -1).sum(-1).squeeze(-2)
        # print(cond_emb.shape)
        full_cond = torch.concat((x.squeeze(), cond_emb), -1)
        # print(full_cond.shape)
        dist_params = model.condNet(full_cond)
        dist_params = dist_params.chunk(2, -1)
        _mus = dist_params[0]
        _stds = torch.nn.Softplus()(dist_params[1]) + margin



    if given:
        mu = _mus
        sigma = _stds
        #sim2 = list(np.random.normal(mu, sigma, 10000))
        #sim2 = pd.DataFrame(sim2, columns=["Emission Intensity"])
        sim2 = np.random.normal(mu, sigma, 10000)
        group_labels = ["  "]
        hist_data = [sim2]
        figln2 = ff.create_distplot(hist_data, group_labels,  show_hist=False) #px.histogram(sim2, x="Emission Intensity",  nbins=150)



    return  figln2


if __name__=='__main__':
    app.run_server(debug=True)