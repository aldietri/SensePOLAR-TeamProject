import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import math
import plotly.colors as colors
from plotly.subplots import make_subplots


class PolarityPlotter:
    """
    A class to plot word polarity.

    Attributes:
        antonym_dict (defaultdict): A dictionary to store the antonym polarities.
        word_colors (dict): A dictionary to store the colors for each word.
        sort_by (str): Sort order for antonym pairs (default: None).
        order_by (str): Order of sorting (default: 'asec').

    Methods:
        generate_color_list: Generates a list of colors.
        create_antonym_dict: Creates a dictionary of antonym polarities.
        plot_word_polarity: Plots the word polarity using lines and markers.
        plot_word_polarity_polar: Plots the word polarity using Scatterpolar plots.
        plot_word_polarity_polar_fig: Plots the word polarity using polar coordinates.
        plot_word_polarity_2d: Plots the word polarity in a 2D scatter plot.
        get_most_descriptive_antonym_pairs: Retrieves the common antonym pairs that best describe the inspected words.
        plot_descriptive_antonym_pairs: Plots each word against the descriptive antonym pairs using a horizontal bar plot.
        plot_word_polarity_2d_interactive: Plots the word polarity in a 2D scatter plot with interactive dropdowns.
    """

    def __init__(self, sort_by=None, order_by='asec'):
        """
        Initialize the PolarityPlotter object.

        Args:
            sort_by (str): Sort order for antonym pairs (default: None).
            order_by (str): Order of sorting (default: 'asec').
        """
        self.antonym_dict = None
        self.word_colors = {}
        self.sort_by = sort_by
        self.order_by = order_by

    def generate_color_list(self, n):
        """
        Generate a list of colors using the Viridis color scale.

        Args:
            n (int): Number of colors to generate.

        Returns:
            list: List of colors.
        """
        color_palette = colors.sample_colorscale("Viridis", n+1)
        return color_palette

    def create_antonym_dict(self, words, contexts, polar_dimension):
        """
        Creates a dictionary of antonym polarities.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        if self.sort_by == 'descriptive':
            polar_dimension = self.get_most_descriptive_antonym_pairs(words, polar_dimension, words, len(polar_dimension))
            polar_dimension = polar_dimension if self.order_by == 'desc' else polar_dimension[::-1]
            converted_data = []
            for i in range(len(polar_dimension[0][1])):
                temp = []
                for pair, values in polar_dimension:
                    temp.append((*pair, values[i]))
                converted_data.append(temp)
            polar_dimension = converted_data
        antonym_dict = defaultdict(list)
        colors = self.generate_color_list(len(words))
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                antonym_dict[(antonym1, antonym2)].append(((words[w_i], contexts[w_i]), value))
            self.word_colors[words[w_i]] = colors[w_i]
        self.antonym_dict = antonym_dict

    def plot_word_polarity(self, words, contexts, polar_dimension):
        """
        Plots the word polarity for a given set of words and polar dimensions.

        Args:
            words (list): A list of words.
            contexts (list): A list of contexts (definitions) for each word.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        self.create_antonym_dict(words, contexts, polar_dimension)
        fig = make_subplots(rows=1, cols=1)
        counter = 0.05
        offset = 0.05
        min_polar = np.inf
        max_polar = -np.inf
        antonym_traces = []
        word_traces = []  # Track word traces for each antonym pair

        # Calculate min and max polar values
        for antonyms, polars in self.antonym_dict.items():
            for polar in polars:
                min_polar = min(min_polar, polar[1])
                max_polar = max(max_polar, polar[1])

        x_range = [min_polar - 0.5, max_polar + 0.5]
        scale = max(abs(x_range[0]), abs(x_range[1]))

        fig.add_shape(type="line", x0=-scale, y0=0, x1=scale, y1=0, line=dict(color='gray', width=1))
        fig.add_shape(type="line", x0=-scale, y0=0, x1=-scale, y1=len(self.antonym_dict) / 10,
                      line=dict(color='gray', width=1))
        fig.add_shape(type="line", x0=scale, y0=0, x1=scale, y1=len(self.antonym_dict) / 10,
                      line=dict(color='gray', width=1))
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=len(self.antonym_dict) / 10,
                      line=dict(color='gray', width=1, dash='dash'))
        
        for y_ref in np.arange(0, (len(self.antonym_dict) / 10), 0.05):
            fig.add_shape(type="line", x0=-scale, y0=y_ref, x1=scale, y1=y_ref,
                          line=dict(color='gray', width=1, dash='dot'))

        for i, (antonyms, polars) in enumerate(self.antonym_dict.items()):
            show_legend = True if i == 0 else False
            x_coords = []
            y_coords = []
            antonym_traces.append([])
            for polar in polars:
                antonym_label = polar[0][0]
                hover_text = f"Word: {polar[0][0]}<br>Definition: {contexts[words.index(polar[0][0])]}<br>Polarity: {polar[1]}"
                x_coords.append(polar[1])
                y_coords.append(counter)
                trace = go.Scatter(x=[polar[1]], y=[counter], mode='markers',
                                   marker=dict(symbol='square', size=20, color=self.word_colors[polar[0][0]]),
                                   name=antonym_label, showlegend=show_legend, legendgroup=polar[0][0],
                                   hoverinfo="text", text=hover_text)
                antonym_traces[-1].append(trace)
                word_trace = go.Line(x=[polar[1], 0], y=[counter, counter], mode='lines',
                                line=dict(width=2, color=self.word_colors[polar[0][0]]), showlegend=False,
                                legendgroup=polar[0][0])
                word_traces.append(trace)
                fig.add_trace(trace)
                fig.add_trace(word_trace)
            fig.add_annotation(xref="x", yref="y", x=-scale-0.1, y=counter, text=antonyms[0][0],
                               font=dict(size=18), showarrow=False, xanchor='right', hovertext=f"Definition: {antonyms[0][1]}")
            fig.add_annotation(xref="x", yref="y", x=scale+0.1, y=counter, text=antonyms[1][0],
                               font=dict(size=18), showarrow=False, xanchor='left', hovertext=f"Definition: {antonyms[1][1]}")
        
            counter += offset

        fig.update_layout(
            xaxis_title="Polarity",
            yaxis_title="Words",
            xaxis_range=x_range,
            xaxis_autorange=True,
            yaxis_autorange=True,
            yaxis=dict(
                showticklabels=False
            )
        )

        return fig



    def plot_word_polarity_polar(self, words, contexts, polar_dimension):
        """
        Plots the word polarity using Scatterpolar plots.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.

        Returns:
            None
        """
        fig = go.Figure()
        colors = self.generate_color_list(len(words))
        for word, context, polar_dim in zip(words, contexts, polar_dimension):
            r = [abs(value) for _, _, value in polar_dim]
            theta = [antonym2[0] if value > 0 else antonym1[0] for antonym1, antonym2, value in polar_dim]
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                name= f"Word: {word}<br>Definition:{context}",
                marker=dict(color=self.word_colors[word])
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

        return fig

   

    def plot_word_polarity_2d(self, words, polar_dimension, x_axis=None, y_axis=None):
        """
        Plots the word polarity in a 2D scatter plot.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.
            x_axis (int): Index of the X-axis antonym pair (default: None).
            y_axis (int): Index of the Y-axis antonym pair (default: None).

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

        colors = self.generate_color_list(len(words))

        for i, word in enumerate(word_dict):
            x, y = word_dict[word][:2]
            if x_axis is not None and y_axis is not None:
                x, y = word_dict[word][x_axis], word_dict[word][y_axis]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(color=self.word_colors[word], size=18),
                                    name=word))

        fig.add_annotation(x=-max_value, y=0, text=antonyms[0][0][0], showarrow=False, xshift=-15, font=dict(size=18),)
        fig.add_annotation(x=max_value, y=0, text=antonyms[0][1][0], showarrow=False, xshift=15, font=dict(size=18),)
        fig.add_annotation(x=0, y=-max_value, text=antonyms[1][0][0], showarrow=False, yshift=-15, font=dict(size=18),)
        fig.add_annotation(x=0, y=max_value, text=antonyms[1][1][0], showarrow=False, yshift=15, font=dict(size=18),)

        return fig

    def get_most_descriptive_antonym_pairs(self, words, polar_dimensions, inspect_words, n=-1):
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
        """
        if n == -1:
            n = len(inspect_words)
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

        if self.order_by == 'asec':
            sorted_pairs = [pair for _, pair in sorted(zip(scores, descriptive_pairs), reverse=False)]
        else:
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
            polar_dimensions (list): A list of polar dimensions.
            inspect_words (list): A subset of words to inspect.
            n (int): Number of antonym pairs to retrieve.

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimensions)
        descriptive_pairs = self.get_most_descriptive_antonym_pairs(words, polar_dimensions, inspect_words, n)
        fig = make_subplots(
            rows=len(descriptive_pairs),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{pair[0][0]}-{pair[0][1]}" for pair in descriptive_pairs]
        )

        scale = max(max([abs(n) for n in polars]) for _, polars in descriptive_pairs) + 0.5
        min_polarity = -scale
        max_polarity = scale

        for i, (antonym_pair, polarity_values) in enumerate(descriptive_pairs):
            fig_idx = i + 1
            legend_names = []
            for j, word in enumerate(inspect_words):
                if word not in legend_names:
                    legend_names.append(word)
                    if i == 0:
                        fig.add_trace(
                            go.Bar(
                                y=[word],
                                x=[polarity_values[j]],
                                name=word,
                                marker=dict(color=self.word_colors[word]),
                                orientation='h',
                                showlegend=True,
                                offsetgroup=f"Pair {fig_idx}"
                            ),
                            row=fig_idx,
                            col=1
                        )
                    else:
                        fig.add_trace(
                            go.Bar(
                                y=[word],
                                x=[polarity_values[j]],
                                name=word,
                                marker=dict(color=self.word_colors[word]),
                                orientation='h',
                                showlegend=False,
                                offsetgroup=f"Pair {fig_idx}"
                            ),
                            row=fig_idx,
                            col=1
                        )


        fig.update_layout(
            title="Word Polarity for Descriptive Antonym Pairs",
            barmode="group",
            legend_traceorder="reversed",
            xaxis=dict(range=[min_polarity, max_polarity]),
            height=200 * len(descriptive_pairs),
            legend_itemclick=False 
        )

        fig.update_yaxes(showticklabels=False)

        return fig




    def plot_word_polarity_2d_interactive(self, words, polar_dimension, x_antonym_pair=None, y_antonym_pair=None):
        """
        Plots the word polarity in a 2D scatter plot with interactive dropdowns.

        Args:
            words (list): A list of words.
            polar_dimension (list): A list of polar dimensions.
            x_antonym_pair (tuple): The initial antonym pair for the X-axis (default: None).
            y_antonym_pair (tuple): The initial antonym pair for the Y-axis (default: None).

        Returns:
            None
        """
        self.create_antonym_dict(words, polar_dimension)
        word_dict = defaultdict(dict)
        for w_i in range(len(words)):
            for antonym1, antonym2, value in polar_dimension[w_i]:
                word_dict[words[w_i]][(antonym1, antonym2)] = value

        antonym_dict = self.antonym_dict
        colors = self.generate_color_list(len(words))
        antonym_pairs = list(antonym_dict.keys())

        max_value = max(max([abs(val) for val in word_dict[word].values()][:2]) for word in word_dict) + 0.5

        if x_antonym_pair is None:
            x_antonym_pair = antonym_pairs[0]
        if y_antonym_pair is None:
            y_antonym_pair = antonym_pairs[1]

        default_x = ' vs '.join(x_antonym_pair)
        default_y = ' vs '.join(y_antonym_pair)

        word_data = []  # List to store word data

        for i, word in enumerate(words):
            word_entry = {'name': word, 'color': colors[i]}
            for antonym_pair in antonym_pairs:
                if antonym_pair in word_dict[word]:
                    word_entry[(antonym_pair, 'x')] = word_dict[word][antonym_pair]
                    word_entry[(antonym_pair, 'y')] = word_dict[word][antonym_pair]
                else:
                    word_entry[(antonym_pair, 'x')] = 0
                    word_entry[(antonym_pair, 'y')] = 0

            word_data.append(word_entry)

        fig = go.Figure()

        fig.update_layout(
            xaxis_range=(-max_value, max_value),
            yaxis_range=(-max_value, max_value),
            height=800,
            width=800,
            xaxis=dict(
                title=f"{default_x}",
                tickmode='array',
                tickvals=[0, -max_value, max_value],
                ticktext=["0", x_antonym_pair[0], x_antonym_pair[1]],
                side='bottom',
                showline=True,
                showticklabels=True,
                ticks='outside',
            ),
            yaxis=dict(
                title=f"{default_y}",
                tickmode='array',
                tickvals=[0, -max_value, max_value],
                ticktext=["0", y_antonym_pair[0], y_antonym_pair[1]],
                side='left',
                showline=True,
                showticklabels=True,
                ticks='outside',
            ),
        )

        fig.add_shape(type="line", x0=-max_value, y0=0, x1=max_value, y1=0, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=0, y0=-max_value, x1=0, y1=max_value, line=dict(color='black', width=1))

        scatter_traces = []
        for word_entry in word_data:
            x_value = word_entry[(x_antonym_pair, 'x')]
            y_value = word_entry[(y_antonym_pair, 'y')]

            scatter_trace = go.Scatter(
                x=[x_value],
                y=[y_value],
                mode='markers',
                marker=dict(color=word_entry['color'], size=18),
                name=word_entry['name']
            )
            scatter_traces.append(scatter_trace)

        fig.add_traces(scatter_traces)

        return fig
