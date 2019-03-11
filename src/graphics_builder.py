# ------------------------------------------------------------------------------
# Loading the main libraries:
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly

plotly.tools.set_credentials_file(username='lia_amaral', api_key='yymuh9KYQM2uJphpnkSc')
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
import plotly.tools as tls
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------------------------------


class GraphicsBuilder:

    def __init__(self, IN_CSV_LIST=None,
                 OUT_CSV_LIST=None,
                 RAIN_CSV=None,
                 NORAIN_CSV=None,
                 COLUMN_TYPES=None,
                 THRESHOLD_RAIN=None):
        self.IN_CSV_LIST = IN_CSV_LIST
        self.RAIN_CSV = RAIN_CSV
        self.NORAIN_CSV = NORAIN_CSV
        self.COLUMN_TYPES = COLUMN_TYPES
        self.THRESHOLD_RAIN = THRESHOLD_RAIN
    """
    Create the plots and graphics to help the exploratory analysis.
    """

    @staticmethod
    def scatter_plotter(dataframe: pd.DataFrame, xvalue: str, xtitle: str, yvalue: str, ytitle: str,
                             chart_title: str, output_file_name: str) -> None:
        #N = 370392
        trace = go.Scattergl(
            x=dataframe[xvalue],
            y=dataframe[yvalue],
            mode='markers',
            marker=dict(
                line=dict(
                    width=1,
                    color='aquamarine')
            )
        )
        layout = go.Layout(
            width=1200,
            height=800,
            title=chart_title,
            hovermode='closest',
            xaxis=dict(
                title=xtitle,
                ticklen=5,
                zeroline=False,
                gridwidth=2,
            ),
            yaxis=dict(
                title=ytitle,
                ticklen=5,
                gridwidth=2,
            ),
            showlegend=False
        )
        data = [trace]
        fig = go.Figure(data=data, layout=layout)
        #plotly.offline.plot(fig, image='png', image_filename=output_file_name)
        plotly.offline.plot(fig, filename=output_file_name)
        return None

    def box_plotter(self, df_rain: pd.DataFrame, df_norain: pd.DataFrame, yvalue1: str, ytitle1: str,
                    yvalue2: str, ytitle2: str, size_norain: str, size_rain: str, chart_title: str, output_file_name: str) -> None:

        trace0 = go.Box(
            y=yvalue1,
            name=ytitle1+' ('+size_norain+') ',
            marker=dict(
                    color='purple',
                    #color='rgb(0, 128, 128)',
            )
        )
        trace1 = go.Box(
            y=yvalue2,
            name=ytitle2+' ('+size_rain+') ',
            marker=dict(
                    color='darksalmon',
                   #color='rgb(10, 140, 208)',
            )
        )
        layout = go.Layout(
            title=chart_title
        )
        data = [trace0, trace1]
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename=output_file_name)
        return None




