import streamlit as st

from sensepolar.plotter import PolarityPlotter
import streamlit.components.v1 as components

import pandas as pd

st.set_page_config(layout="centered", page_title="SensePOLAR", page_icon="🌊")
st.elements.utils._shown_default_value_warning=True

# Changes block color and other attributes
block_color = """
<style>
[data-testid="stHorizontalBlock"]:nth-of-type(1){
    background-color: #7895CB;
    padding: 4px;
    border-radius: 10px;
    color: #FFFFFF
}
</style>
"""
st.markdown(block_color, unsafe_allow_html=True) 

# TODO: May need to be changed or removed, dependent on future adjustments of the site
# Changes color of text input labels
label_color = """
<style>
.stTextInput > label{
    color: #FFFFFF
}
</style>
"""
st.markdown(label_color, unsafe_allow_html=True)

# Changes sidebar color
sidebar_color = """
<style>
[data-testid=stSidebar]{
    background-color: #7895CB;
}
</style>
"""
st.markdown(sidebar_color, unsafe_allow_html=True) 

# Changes block color and other attributes
upload_color = """
<style>
[data-testid="stFileUploader"]{
    background-color: #7895CB;
    padding: 4px;
    border-radius: 10px;
    color: #FFFFFF
}
</style>
"""
st.markdown(upload_color, unsafe_allow_html=True) 

# TODO: May need to be changed or removed, dependent on future adjustments of the site
# Changes color of text input labels
upload_label_color = """
<style>
[data-testid="stFileUploader"] > label{
    color: #FFFFFF
}
</style>
"""
st.markdown(upload_label_color, unsafe_allow_html=True) 


# Changes alignment of text - centers headerContainer text to buttons
text_alignment = """
<style>
[data-testid="stText"] {
    padding-top: 11px
}
</style>
"""
st.markdown(text_alignment, unsafe_allow_html=True) 

# Changes color of widget text with certain text
def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# Headline
st.title("	📷 Visualizer", anchor=False)

# Subheadline with short explaination
with st.expander("Intro", expanded=True):
    st.write("""
        Welcome to our visualization page! 📊 Here, we bring the results extracted from the SensePOLAR Framework to 
        life through captivating visualizations that make understanding complex NLP tasks a breeze. 
        Our user-friendly interface allows both beginners and experts to explore and interact with the data effortlessly.
        Embrace the visualization page and let the data speak for itself as you embark on a transformative NLP adventure. 🌟
    """)

@st.cache_data
def load_dataframe(uploaded_file, template_path):
    """
    # Load pandas dataframe from specified path.

    Parameters:
    -----------
    uploaded_file : UploadedFile class of streamlit
        An uploaded file.
    template_path: String
        A string of the template path.

    Returns:
    -----------
    dataframe/template : pandas.DataFrame
       A pandas dataframe.
    """

    # Load template
    template = pd.read_csv(template_path)
    
    # Check if file was uploaded
    if uploaded_file:
        # Excel
        if ".xlsx" in uploaded_file.name:
            dataframe = pd.read_excel(uploaded_file, header=0)
        # CSV
        else:
            dataframe = pd.read_csv(uploaded_file)

        # Check if uploaded file contains necessary columns of the template layout
        if set(template.columns).issubset(set(dataframe.columns)):
            return dataframe[template.columns]

        # Print warning if uplaoded file does not equal the template layout
        st.warning("Please check whether the uploaded file is formatted in the required way.", icon="⚠️")
    return template

# File upload for polar antoym pairs
uploaded_file_results = st.file_uploader("Already prepared a file?", key="Result_Upload")

# Load dataframe
dataframe_results = load_dataframe(uploaded_file_results, "excel_files_for_experts/Empty_Dataframe_Result.csv")

# Show dataframe
st.dataframe(dataframe_results, use_container_width=True)

# Save data to session state
st.session_state["df_results"] = dataframe_results


@st.cache_data
def create_visualisations(options, dataframe, k, x_axis, y_axis, ordering):
    """
    # Creates visualizations for the word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    options : list
        A list containing the specified plots that are to be returned.
    words : list
        A list containing the analyzed words.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    k: int
        An integer to indicate how many antonym pairs to consider when selecting the most discriminative antonym pairs
    x_axis: list
        A list containing the x_axis values that are to be displayed for a 2d plot
    y_axis: list
        A list containing the y_axis values that are to be displayed for a 2d plot
    """
    
    plotter = PolarityPlotter()

    # Get all unique words
    words = dataframe["word"].unique()

    # Extract antonym pairs and polar values and bring them into the needed format for visualization
    polar_dimensions = []
    for word in words:
        polar_dimensions.append(dataframe[dataframe["word"] == word][["antonym_1", "antonym_2", "value"]].values.tolist())


    tabs = st.tabs(options)

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, polar_dimensions)
        tabs[options.index("Standard")].plotly_chart(fig, use_container_width=True)

    if "2d" in options:
        fig = plotter.plot_word_polarity_2d_interactive(words, polar_dimensions, x_antonym_pair=tuple(x_axis), y_antonym_pair=tuple(y_axis))
        tabs[options.index("2d")].plotly_chart(fig, use_container_width=True)

    if "Polar" in options:
        fig = plotter.plot_word_polarity_polar_fig(words, polar_dimensions)
        tabs[options.index("Polar")].plotly_chart(fig, use_container_width=True)

    if "Most discriminative" in options:
        # TODO: Use selected ordering
        fig = plotter.plot_descriptive_antonym_pairs(words, polar_dimensions, words, k)
        tabs[options.index("Most discriminative")].plotly_chart(fig, use_container_width=True)


# Sidebar

with st.sidebar:
    st.markdown("# Visualization")
    if len(st.session_state["df_results"]) < 2:
        given_options = ["Standard", "Polar", "Most discriminative"]
    else:
        given_options = ["Standard", "2d", "Polar", "Most discriminative"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)

    # Axes choice for 2d plot
    x_axis = []
    y_axis = []
    if "2d" in selected_options:
        st.markdown("## 2D")
        axes_column = st.columns(2)
        antonym_count = int(st.session_state["df_results"].shape[0] / st.session_state["df_results"]["word"].nunique())
        axis_values = list(zip(st.session_state["df_results"]["antonym_1"][:antonym_count], st.session_state["df_results"]["antonym_2"][:antonym_count]))
        x_axis = axes_column[0].selectbox("x-axis", axis_values, format_func=lambda x: ", ".join(x))
        y_axis = axes_column[1].selectbox("y-axis", axis_values, format_func=lambda x: ", ".join(x))

    # Number Input for Most discriminative plot
    k = 3
    if "Most discriminative" in selected_options:
        st.write("## Most Discriminative")
        k = st.number_input("Please select the amount of most discriminative antonym pairs to consider ", min_value=1, max_value=antonym_count)

        # Ascending or Descending ordering of Most descriminative antonym pairs
        selected_ordering = st.selectbox("Please select the ordering of the most descriminative antonym pairs", options=["Ascending", "Descending"])

def check_inputs(df):
    """
    # Checks whether all relevant input fields are populated in a proper manner.

    Parameters:
    -----------
    df : pandas.DataFrame
        A dataframe containing words, antonym pairs, and polar values.


    Returns:
    -----------
    boolean
        A boolean indicating whether all relevant input fields are populated in a proper manner.
    """

    # Check antonyms
    # If dataframe contains any null or empty string values parse warning and return false
    if df.isnull().values.any() or df.eq("").values.any():
        st.warning("Please check whether all necessary antonym fields have been populated before executing", icon="⚠️")
        return False
    
    # If everything is okay return true
    return True


# Execute button - Execute SensePOLAR calculation and visualization
if st.button(label="Execute", key="execute"):
    # Checks whether all relevant input fields are populated in a proper manner and then execute calculation and visualization
    if check_inputs(st.session_state["df_results"]):
        # Check whether visualization options have been selected
        if not selected_options:
            st.warning("Please select some visualization options", icon="⚠️")
        else:
            try:
                create_visualisations(selected_options, st.session_state["df_results"], k, x_axis, y_axis, selected_ordering)
            except:
                st.warning("An error has occured. Please check your selected Inputs", icon="⚠️")