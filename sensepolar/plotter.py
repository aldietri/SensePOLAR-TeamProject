import plotly.graph_objs as go
import numpy as np
from collections import defaultdict
import math
import plotly.colors as colors
from ipywidgets import interact
from plotly.subplots import make_subplots

class PolarityPlotter:
    """
    A class to plot word polarity.

    Attributes:
        antonym_dict (defaultdict): A dictionary to store the antonym polarities.
        word_colors (dict): A dictionary to store the colors for each word.

    Methods:
        create_antonym_dict: Creates a dictionary of antonym polarities.
        generate_color_list: Generates a list of colors.
        plot_word_polarity: Plots the word polarity using lines and markers.
        plot_word_polarity_polar_fig: Plots the word polarity using polar coordinates.
        plot_word_polarity_2d: Plots the word polarity in a 2D scatter plot.
    """

    def __init__(self):
        self.antonym_dict = None
        self.word_colors = {}

    def generate_color_list(self, n):
        """
        Generate a list of colors using the Viridis color scale.

        Args:
            n (int): Number of colors to generate.

        Returns:
            list: List of colors.
        """
        color_palette = colors.sample_colorscale("Viridis", n)
        return color_palette

    def create_antonym_dict(self, words, polar_dimension):
        """
        Creates a dictionary of antonym polarities.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        antonym_dict = defaultdict(list)
        colors = self.generate_color_list(len(words))
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                antonym_dict[(antonym1, antonym2)].append((words[w_i], value))
            self.word_colors[words[w_i]] = colors[w_i]
        self.antonym_dict = antonym_dict

    def plot_word_polarity(self, words, polar_dimension):
        """
        Plots the word polarity for a given set of words and polar dimensions.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        fig = go.Figure()

        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=-1, y0=0, x1=-1, y1=len(self.antonym_dict) / 10, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=1, y0=0, x1=1, y1=len(self.antonym_dict) / 10, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=len(self.antonym_dict) / 10,
                      line=dict(color='black', width=1, dash='dash'))

        counter = 0.01
        offset = 0.005
        for i, (antonyms, polars) in enumerate(self.antonym_dict.items()):
            show_legend = True if i == 0 else False
            x_coords = []
            y_coords = []
            for polar in polars:
                x_coords.append(polar[1])
                y_coords.append(counter)
                fig.add_shape(type="line", x0=polar[1], y0=counter, x1=0, y1=counter,
                              line=dict(width=2, color=self.word_colors[polar[0]]), visible=show_legend)
                fig.add_trace(go.Scatter(x=[polar[1]], y=[counter], mode='markers',
                                         marker=dict(symbol='square', size=10, color=self.word_colors[polar[0]]),
                                         name=polar[0], showlegend=show_legend))
            fig.add_annotation(x=-1.1, y=counter, text=antonyms[0], font=dict(size=18), showarrow=False, xanchor='right')
            fig.add_annotation(x=1.1, y=counter, text=antonyms[1], font=dict(size=18), showarrow=False, xanchor='left')
            counter += offset + 0.1

        for word, polar in zip(words, polar_dimension):
            x_coords = [p[1] for p in polar]
            y_coords = [counter] * len(polar)
            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',
                                     line=dict(width=1, color=self.word_colors[word]),
                                     showlegend=False, visible='legendonly'))

        fig.update_layout(
            xaxis_title="Polarity",
            yaxis_title="Words",
            xaxis_range=[-1, 1],
            xaxis_autorange=True, yaxis_autorange=True
        )

        fig.show()

    def plot_word_polarity_polar_fig(self, words, polar_dimension):
        """
        Plots the word polarity using polar coordinates.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        word_dict = {word: None for word in words}
        for idx, item in enumerate(polar_dimension):
            antonym_dict = {}
            for antonym1, antonym2, value in item:
                if value > 0:
                    antonym_dict[antonym1] = value
                else:
                    antonym_dict[antonym2] = abs(value)
            word_dict[words[idx]] = antonym_dict

        num_cols = 2
        num_rows = math.ceil(len(polar_dimension) / num_cols)
        specs = [[{"type": "polar"} for _ in range(num_cols)] for _ in range(num_rows)]

        fig = make_subplots(rows=num_rows, cols=num_cols, specs=specs)

        for idx, word in enumerate(word_dict):
            fig.add_trace(
                go.Scatterpolar(
                    r=list(word_dict[word].values()),
                    theta=list(word_dict[word].keys()),
                    name=word,
                    line=dict(width=1, color=self.word_colors[word]),
                ), int(idx / 2 + 1), (idx % 2 + 1)
            )

        fig.update_traces(fill="toself")

        fig.show()

    def plot_word_polarity_2d(self, words, polar_dimension):
        """
        Plots the word polarity in a 2D scatter plot.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        antonym_dict = defaultdict(list)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                antonym_dict[(antonym1, antonym2)].append((words[w_i], value))

        word_dict = defaultdict(list)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                word_dict[words[w_i]].append(value)

        fig = go.Figure()

        antonyms = list(antonym_dict.keys())

        fig.update_layout(
            xaxis_range=(-1, 1),
            yaxis_range=(-1, 1)
        )

        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color='black', width=1))

        colors = np.linspace(0, 1, len(words))

        for i, word in enumerate(word_dict):
            x, y = word_dict[word][:2]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(color=self.word_colors[word], size=10),
                                     name=word))

        fig.add_annotation(x=-1, y=0, text=antonyms[0][0], showarrow=False, xshift=-15)
        fig.add_annotation(x=1, y=0, text=antonyms[0][1], showarrow=False, xshift=15)
        fig.add_annotation(x=0, y=-1, text=antonyms[1][0], showarrow=False, yshift=-15)
        fig.add_annotation(x=0, y=1, text=antonyms[1][1], showarrow=False, yshift=15)

        fig.show()

    def plot_word_polarity_interactive(self, words, polar_dimension):
        """
        Plots the word polarity in an interactive plot.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        antonym_dict = defaultdict(list)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                antonym_dict[(antonym1, antonym2)].append((words[w_i], value))

        fig = go.Figure()

        antonyms = list(antonym_dict.keys())

        fig.update_layout(
            xaxis_range=(-1, 1),
            yaxis_range=(-1, 1)
        )

        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color='black', width=1))

        colors = np.linspace(0, 1, len(words))

        for i, word in enumerate(words):
            x_coords = [p[1] for p in polar_dimension[i]]
            y_coords = [i] * len(polar_dimension[i])
            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',
                                    line=dict(width=1, color=self.word_colors[word]),
                                    showlegend=False, visible='legendonly'))

        scatter_trace = go.Scatter(x=[], y=[], mode='markers',
                                marker=dict(symbol='square', size=10), showlegend=False)

        fig.add_trace(scatter_trace)

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=word,
                            method="update",
                            args=[{"visible": [False] * (len(words) + 1)}, {"title": word}],
                        ) for word in words
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1,
                    xanchor="right",
                    y=1,
                    yanchor="top"
                ),
            ]
        )

        fig.data[0].visible = True
        fig.data[-1].visible = True

        def update_trace(trace, points, selector):
            scatter_trace = fig.data[-1]
            scatter_trace.x = []
            scatter_trace.y = []
            scatter_trace.marker.color = []

            word = trace['name']
            idx = words.index(word)

            scatter_trace.x += [p[1] for p in polar_dimension[idx]]
            scatter_trace.y += [idx] * len(polar_dimension[idx])
            scatter_trace.marker.color += [self.word_colors[word]] * len(polar_dimension[idx])

            scatter_trace.visible = True

            # Check if any word is selected in any antonym pair
            any_word_selected = any(fig.data[i + 1].visible for i in range(len(words)))

            if any_word_selected:
                fig.update_layout(
                    xaxis_range=(-1, 1),
                    yaxis_range=(-1, len(words))
                )
            else:
                fig.update_layout(
                    xaxis_range=None,
                    yaxis_range=None
                )

        for i, word in enumerate(words):
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(color=self.word_colors[word]),
                    showlegend=True,
                    legendgroup=word,
                    name=word,
                    visible=False
                )
            )

            fig.data[i + 1].on_click(update_trace)

        fig.show()
