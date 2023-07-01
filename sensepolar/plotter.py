from sys import displayhook
import plotly.graph_objs as go
import numpy as np
from collections import defaultdict
import math
import plotly.colors as colors
from ipywidgets import interact
from plotly.subplots import make_subplots
from ipywidgets import widgets, Layout

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
        fig = make_subplots(rows=1, cols=1)
        counter = 0.01
        offset = 0.005
        min_polar = np.inf
        max_polar = -np.inf
        antonym_traces = []
        word_traces = []  # Track word traces for each antonym pair

        # Calculate min and max polar values
        for antonyms, polars in self.antonym_dict.items():
            for polar in polars:
                min_polar = min(min_polar, polar[1])
                max_polar = max(max_polar, polar[1])

        x_range = [min_polar - 0.1, max_polar + 0.1]
        scale = max(abs(x_range[0]), abs(x_range[1]))

        fig.add_shape(type="line", x0=-scale, y0=0, x1=scale, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=-scale, y0=0, x1=-scale, y1=len(self.antonym_dict) / 10,
                      line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=scale, y0=0, x1=scale, y1=len(self.antonym_dict) / 10,
                      line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=len(self.antonym_dict) / 10,
                      line=dict(color='black', width=1, dash='dash'))

        for i, (antonyms, polars) in enumerate(self.antonym_dict.items()):
            show_legend = True if i == 0 else False
            x_coords = []
            y_coords = []
            antonym_traces.append([])
            for polar in polars:
                x_coords.append(polar[1])
                y_coords.append(counter)
                trace = go.Scatter(x=[polar[1]], y=[counter], mode='markers',
                                   marker=dict(symbol='square', size=20, color=self.word_colors[polar[0]]),
                                   name=polar[0], showlegend=show_legend, legendgroup=polar[0])
                antonym_traces[-1].append(trace)
                word_trace = go.Line(x=[polar[1], 0], y=[counter, counter], mode='lines',
                                line=dict(width=2, color=self.word_colors[polar[0]]), showlegend=False,
                                legendgroup=polar[0])
                word_traces.append(trace)
                fig.add_trace(trace)
                fig.add_trace(word_trace)
            fig.add_annotation(xref="x", yref="y", x=-scale-0.1, y=counter, text=antonyms[0],
                               font=dict(size=18), showarrow=False, xanchor='right')
            fig.add_annotation(xref="x", yref="y", x=scale+0.1, y=counter, text=antonyms[1],
                               font=dict(size=18), showarrow=False, xanchor='left')
            counter += offset + 0.1

        fig.update_layout(
            xaxis_title="Polarity",
            yaxis_title="Words",
            xaxis_range=x_range,
            xaxis_autorange=True,
            yaxis_autorange=True
        )
        fig.show()

    def plot_word_polarity_polar(self, words, polar_dimension):
        """
        Plots the word polarity using Scatterpolar plots.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        fig = go.Figure()

        for word, polar_dim in zip(words, polar_dimension):
            r = [abs(value) for _, _, value in polar_dim]
            theta = [antonym2 if value > 0 else antonym1 for antonym1, antonym2, value in polar_dim]

            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                name=word
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max([abs(value) for _, _, value in dim]) for dim in polar_dimension])]
                )
            ),
            showlegend=True
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

    def plot_word_polarity_2d(self, words, polar_dimension, x_axis=None, y_axis=None):
        """
        Plots the word polarity in a 2D scatter plot.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        word_dict = defaultdict(list)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                word_dict[words[w_i]].append(value)

        fig = go.Figure()

        antonym_dict = self.antonym_dict

        antonyms = list(antonym_dict.keys())

        max_value = max(max([abs(val) for val in word_dict[word][:2]]) for word in word_dict) + 0.5

        fig.update_layout(
            xaxis_range=(-max_value, max_value),
            yaxis_range=(-max_value, max_value),
            height=800,
            width=800,
        )

        fig.add_shape(type="line", x0=-max_value, y0=0, x1=max_value, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-max_value, x1=0, y1=max_value, line=dict(color='black', width=1))

        colors = np.linspace(0, 1, len(words))

        for i, word in enumerate(word_dict):
            x, y = word_dict[word][:2]
            if x_axis is not None and y_axis is not None:
                x, y = word_dict[word][x_axis], word_dict[word][y_axis]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(color=self.word_colors[word], size=18),
                                    name=word))

        fig.add_annotation(x=-max_value, y=0, text=antonyms[0][0], showarrow=False, xshift=-15, font=dict(size=18),)
        fig.add_annotation(x=max_value, y=0, text=antonyms[0][1], showarrow=False, xshift=15, font=dict(size=18),)
        fig.add_annotation(x=0, y=-max_value, text=antonyms[1][0], showarrow=False, yshift=-15, font=dict(size=18),)
        fig.add_annotation(x=0, y=max_value, text=antonyms[1][1], showarrow=False, yshift=15, font=dict(size=18),)

        fig.show()


    def get_most_descriptive_antonym_pairs(self, words, polar_dimensions, inspect_words, n):
        """
        Retrieves the common antonym pairs that best describe the inspected words.

        Args:
            words (list): A list of all words.
            polar_dimensions (list): A list of polar dimensions.
            inspect_words (list): A subset of words to inspect.
            n (int): Number of antonym pairs to retrieve.

        Returns:
            list: The common antonym pairs that best describe the inspected words,
                along with the polarity values.

        Note:
            This implementation assumes that each word has only one antonym pair associated with it.
        """
        word_dict = {word: None for word in words}
        for idx, item in enumerate(polar_dimensions):
            antonym_dict = {}
            for antonym1, antonym2, value in item:
                antonym_dict[(antonym1, antonym2)] = value
            word_dict[words[idx]] = antonym_dict

        common_antonyms = set(word_dict[inspect_words[0]].keys())
        for word in inspect_words[1:]:
            common_antonyms.intersection_update(word_dict[word].keys())

        descriptive_pairs = []
        scores = []

        for antonym in common_antonyms:
            score = sum(abs(word_dict[word][antonym]) for word in inspect_words)
            descriptive_pairs.append(antonym)
            scores.append(score)

        sorted_pairs = [pair for _, pair in sorted(zip(scores, descriptive_pairs), reverse=True)]

        if len(sorted_pairs) > n:
            sorted_pairs = sorted_pairs[:n]

        result = []
        for pair in sorted_pairs:
            polarities = [word_dict[word][pair] for word in inspect_words]
            result.append((pair, polarities))

        return result[:n]


    def plot_descriptive_antonym_pairs(self, words, polar_dimensions, inspect_words, n):
        """
        Plots each word against the descriptive antonym pairs using a horizontal bar plot.

        Args:
            words (list): A list of words.
            descriptive_pairs (list): The descriptive antonym pairs, along with the polarity values.

        Returns:
            None
        """
        descriptive_pairs = self.get_most_descriptive_antonym_pairs( words, polar_dimensions, inspect_words, n)
        fig = make_subplots(rows=len(descriptive_pairs), cols=1, shared_xaxes=True, subplot_titles=[f"{pair[0][0]}-{pair[0][1]}" for pair in descriptive_pairs])

        #take max absolute value from all polarity values
        scale = max(max([abs(n) for n in polars]) for _, polars in descriptive_pairs) + 0.5
        min_polarity = -scale
        max_polarity = scale

        color_sequence = colors.qualitative.Plotly

        for i, (antonym_pair, polarity_values) in enumerate(descriptive_pairs):
            fig_idx = i + 1

            color_map = {word: color_sequence[j % len(color_sequence)] for j, word in enumerate(inspect_words)}
            legend_names = [] 

            for j, word in enumerate(inspect_words):
                if word not in legend_names:
                    legend_names.append(word)
                    fig.add_trace(go.Bar(
                        y=[word],
                        x=[polarity_values[j]],
                        name=word,
                        marker=dict(color=color_map[word]), 
                        orientation='h',
                        showlegend=False,
                        offsetgroup=f"Pair {fig_idx}"
                    ), row=fig_idx, col=1)

        fig.update_layout(
            title="Word Polarity for Descriptive Antonym Pairs",
            xaxis_title="Polarity",
            barmode="group",
            legend_traceorder="reversed",
            xaxis=dict(range=[min_polarity, max_polarity]) 
        )

        fig.update_yaxes(showticklabels=False)

        fig.show()



    def plot_word_polarity_2d_interactive(self, words, polar_dimension):
        self.create_antonym_dict(words, polar_dimension)
        word_dict = defaultdict(dict)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                word_dict[words[w_i]][(antonym1, antonym2)] = value

        antonym_dict = self.antonym_dict
        colors = np.linspace(0, 1, len(words))
        antonym_pairs = list(antonym_dict.keys())
        antonyms_x = [' vs '.join(pair) for pair in antonym_pairs]
        antonyms_y = [' vs '.join(pair) for pair in antonym_pairs]

        max_value = max(max([abs(val) for val in word_dict[word].values()][:2]) for word in word_dict) + 0.5

        default_x = antonyms_x[0]
        default_y = antonyms_y[0]

        word_data = []  # List to store word data

        for i, word in enumerate(words):
            if antonym_pairs[0] in word_dict[word].keys() and antonym_pairs[0] in word_dict[word].keys():
                x, y = word_dict[word][antonym_pairs[0]], word_dict[word][antonym_pairs[0]]
                color = colors[i]
                word_data.append({'name': word, 'x': x, 'y': y, 'color': color})  # Add word data to the list

        fig = go.Figure()

        fig.update_layout(
            xaxis_range=(-max_value, max_value),
            yaxis_range=(-max_value, max_value),
            height=800,
            width=800,
            xaxis=dict(title=default_x),
            yaxis=dict(title=default_y),
        )

        fig.add_shape(type="line", x0=-max_value, y0=0, x1=max_value, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-max_value, x1=0, y1=max_value, line=dict(color='black', width=1))

        scatter_traces = []
        for word_data_entry in word_data:
            scatter_trace = go.Scatter(
                x=[word_data_entry['x']],
                y=[word_data_entry['y']],
                mode='markers',
                marker=dict(color=word_data_entry['color'], size=18),
                name=word_data_entry['name']
            )
            scatter_traces.append(scatter_trace)

        fig.add_traces(scatter_traces)

        dropdown_x = go.layout.Updatemenu(
            buttons=list([
                dict(
                    label=antonyms_x[i],
                    method="update",
                    args=[{
                        "x": [[word_dict[word][(selected_x[0], selected_x[1])] for word in words if
                            (selected_x[0], selected_x[1]) in word_dict[word].keys()] for word in words],
                        "xaxis": {"title": antonyms_x[i]}
                    }],
                    args2=[{
                        "x": [[word_data_entry['x'] for word_data_entry in word_data] for _ in words],
                        "y": [[word_data_entry['y'] for word_data_entry in word_data] for _ in words]
                    }]
                )
                for i, selected_x in enumerate(antonym_pairs)
            ]),
            direction="down",
            active=0,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top",
        )

        dropdown_y = go.layout.Updatemenu(
            buttons=list([
                dict(
                    label=antonyms_y[i],
                    method="update",
                    args=[{
                        "y": [[word_dict[word][(selected_y[0], selected_y[1])] for word in words if
                            (selected_y[0], selected_y[1]) in word_dict[word].keys()] for word in words],
                        "yaxis": {"title": antonyms_y[i]}
                    }],
                    args2=[{
                        "x": [[word_data_entry['x'] for word_data_entry in word_data] for _ in words],
                        "y": [[word_data_entry['y'] for word_data_entry in word_data] for _ in words]
                    }]
                )
                for i, selected_y in enumerate(antonym_pairs)
            ]),
            direction="down",
            active=0,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top",
        )

        fig.update_layout(updatemenus=[dropdown_x, dropdown_y])

        fig.show()




    def plot_word_polarity_2d_interactive_(self, words, polar_dimension):
        self.create_antonym_dict(words, polar_dimension)
        word_dict = defaultdict(dict)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                word_dict[words[w_i]][(antonym1, antonym2)] = value

        antonym_dict = self.antonym_dict
        colors = np.linspace(0, 1, len(words))
        antonym_pairs = list(antonym_dict.keys())
        antonyms_x = [' vs '.join(pair) for pair in antonym_pairs]
        antonyms_y = [' vs '.join(pair) for pair in antonym_pairs]

        max_value = max(max([abs(val) for val in word_dict[word].values()][:2]) for word in word_dict) + 0.5

        default_x = antonyms_x[0]
        default_y = antonyms_y[0]

        word_data = []  # List to store word data

        for i, word in enumerate(words):
            if antonym_pairs[0] in word_dict[word].keys() and antonym_pairs[0] in word_dict[word].keys():
                x, y = word_dict[word][antonym_pairs[0]], word_dict[word][antonym_pairs[0]]
                color = colors[i]
                word_data.append({'name': word, 'x': x, 'y': y, 'color': color})  # Add word data to the list

        fig = go.Figure()

        fig.update_layout(
            xaxis_range=(-max_value, max_value),
            yaxis_range=(-max_value, max_value),
            height=800,
            width=800,
            xaxis=dict(title=default_x),
            yaxis=dict(title=default_y),
        )

        fig.add_shape(type="line", x0=-max_value, y0=0, x1=max_value, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-max_value, x1=0, y1=max_value, line=dict(color='black', width=1))

        scatter_traces = []
        for word_data_entry in word_data:
            scatter_trace = go.Scatter(
                x=[word_data_entry['x']],
                y=[word_data_entry['y']],
                mode='markers',
                marker=dict(color=word_data_entry['color'], size=18),
                name=word_data_entry['name']
            )
            scatter_traces.append(scatter_trace)

        fig.add_traces(scatter_traces)

        dropdown_x = widgets.Dropdown(
            options=antonyms_x,
            value=default_x,
            description='X-axis:'
        )

        dropdown_y = widgets.Dropdown(
            options=antonyms_y,
            value=default_y,
            description='Y-axis:'
        )

        def update_plot(change):
            selected_x = dropdown_x.value
            selected_y = dropdown_y.value

            for i, word_data_entry in enumerate(word_data):
                word = word_data_entry['name']
                x = word_dict[word].get((selected_x.split(' vs ')[0], selected_x.split(' vs ')[1]), 0)
                y = word_dict[word].get((selected_y.split(' vs ')[0], selected_y.split(' vs ')[1]), 0)

                fig.data[i].x = [x]
                fig.data[i].y = [y]

        dropdown_x.observe(update_plot, 'value')
        dropdown_y.observe(update_plot, 'value')

        displayhook(widgets.VBox([dropdown_x, dropdown_y]))
        fig.show()