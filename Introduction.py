import streamlit as st
from sensepolar.embed.bertEmbed import BERTWordEmbeddings
# from sensepolar.oracle.dictionaryapi import Dictionary

st.set_page_config(layout="centered", page_title="SensePOLAR", page_icon="🌊")

# Changes sidebar color
sidebar_color = """
<style>
[data-testid=stSidebar]{
    background-color: #7895CB;
}
</style>
"""
st.markdown(sidebar_color, unsafe_allow_html=True) 

st.write("# Welcome to SensePOLAR! 🌊")

with st.expander("Intro", expanded=True):
    st.write("""
        Introducing SensePOLAR: Explore the power of interpretability in text analysis 🔍! Our innovative framework enhances the interpretability of contextual word embeddings like BERT 
        by rating words based on user-defined senses, such as correctness or direction 🎯. Unlike previous methods, SensePOLAR is designed to handle polysemy, effortlessly distinguishing 
        between different word senses. By transforming pre-trained contextual word embeddings into an interpretable space, SensePOLAR empowers users to delve into word semantics on scales like 
        "good" vs. "bad" 👍👎. These interpretable word embeddings excel across diverse NLP tasks, revolutionizing your text analysis. 
        Immerse yourself in the future of word embeddings with SensePOLAR's new user-friendly frontend 🌐!
"""
)
    
st.markdown("""
     ##### SensePOLAR's frontend is composed of three sections:

    🐣 Beginner - Perfect for newcomers, it simplifies the SensePOLAR Framework for easy use, guiding you through an intuitive application of SensePOLAR.

    🦅 Expert - For those seeking customization and advanced capabilities, this section empowers you to creatively tweak your input, change between different language models, and analyse large quantities of data.

    📷 Visualizer - Discover captivating visualizations that bring SensePOLAR's results to life, making complex NLP tasks a breeze for users of all levels.
""")

# Beginner Page

st.write("## Beginner")
with st.expander("Description", expanded=False):

    st.image("media/beginner_overview.png", caption="Fig.1 Overview Beginner Page")

    st.markdown("""The Beginner Page is initially structured into three different compartments and will change
                dependent on user input, calculation and execution. It is primarily made up of the settings bar 🔴, 
                the antonyms selection 🟢, and the subject choice 🔵 (See Fig.1). An examplary workflow could look as follows:
    """)

    st.markdown("""
                1. Select Dictionary
                2. Declare Antonyms
                3. (Optional) Download Antonyms in CSV-Format
                4. Provide Subject Words
                5. Adjust Visualization Settings
                6. Execute Analysis
                7. (Optional) Save Plots
                8. (Optional) Download Analysis Results in CSV-Format   
                
    """)

    st.write("### Dictionaries")
    st.markdown("""On the Beginner Page, the settings menu includes a unique option for selecting different dictionaries. 
                These dictionaries include WordNet, which served as the foundation for the SensePOLAR framework, as well as two additional experimental dictionaries: 
                Wordnik and DictionaryAPI. It's worth noting that utilizing Wordnik requires an API key, and it's important to be aware that the use of Wordnik and DictionaryAPI is considered experimental. 
                This experimental status means that their performance may be unpredictable, and exceptions could potentially occur.        
    """)

    st.write("### Antonyms")
    st.markdown("""The antonyms section functions as an input mechanism for creating various dimensions against which subject words will be assessed later. 
                    On the Beginner Page, users can easily select antonym pairs. They have the option to define their own antonyms and 
                    then select from among the top 5 most common definitions from their previously selected dictionary to provide context for their choice. 
                    Furthermore, users can dynamically add or remove antonym pairs and minimize the antonym pair panels to enhance visual clarity.
                    Additionally, users have the flexibility to download the selected antonym pairs and subsequently upload and process them on the Expert Page,
                    allowing for more advanced and comprehensive analyses when needed.
    """)
    
    st.video("media/beginner_antonyms.webm")

    st.write("### Subjects")
    st.markdown("""The subjects section enables users to input subject words for analysis, which are evaluated based on the antonym pairs defined earlier. 
                To do this, users specify their chosen subject words and provide context in which these subject words are used. 
                This context helps the framework derive meaning. Again, users have the flexibility to add or remove subjects dynamically and can minimize 
                the subjects panels to improve visual clarity.


    """)

    st.video("media/beginner_subjects.webm")

    st.write("### Visualization")
    st.markdown("""In the visualization section, you have the option to select one of two transformation methods for controlling how SensePOLAR calculates its dimensions. 
                Additionally, you can choose the output plots and configure their related settings. SensePOLAR offers two distinct approaches for transforming its embedding space: 
                a 'base change' method and a 'projection-based' method. To help you visualize this transformation, there are currently four different plot variations available:
    """)

    st.write("##### Standard Plot")
    st.markdown("""The standard plot visually represents the polarity of each subject word with respect to the specified axis formed by antonym pairs. 
                In this plot, you can modify the visualization to your preferences through the customization options provided in the settings. 
                This allows you to determine the order of elements within the plot according to your specific requirements.
    """)
    st.image("media/standard_plot.png", caption="Fig.2 Standard Plot")

    st.write("##### 2D Plot")
    st.markdown("""The 2D plot offers a graphical representation of the polarity of subject words along two distinct axes, 
                with each axis defined by selected antonym pairs. In this plot, you have the flexibility to customize your x-axis and y-axis by choosing from the available antonym pairs. 
                This feature empowers you to define and explore the specific dimensions and relationships you want to analyze within the 2D plot, 
                providing a more tailored and insightful visual representation of the data.
    """)
    st.image("media/2d_plot.png", caption="Fig.3 2D Plot")

    st.write("##### Polar Plot")
    st.markdown("""The polar plot offers a distinctive way of visualizing subject word polarity in relation to antonym pairs, resembling a circular or radial chart. 
                It allows for specific settings related to the display of antonym pairs on its axes. You can select your desired antonym pairs and then choose between two display modes:
    """)

    st.markdown("**1. Solo Display**: Here, antonym pairs are shown individually on each axis, allowing for a detailed analysis of subject word polarity in relation to each antonym pair.")
    st.image("media/polar_plot_solo.png", caption="Fig.4 Polar Plot - Solo")

    st.markdown("**2. Grouped Display**: In this mode, antonym pairs are grouped on each axis. This provides a broader perspective on subject word polarity, considering its relationship to the antonym pair as a whole, without distinguishing individual antonyms.")
    st.image("media/polar_plot_grouped.png", caption="Fig.5 Polar Plot - Grouped")


    st.write("##### Most Discriminative Plot")
    st.markdown("""The most discriminative plot is a valuable tool for visualizing the intricate connections between subject words and antonym pairs. 
                Within this plot, you can readily discern the subject words that have particularly strong associations with specific antonyms and the extent of these associations. 
                This visualization empowers you to pinpoint which words showcase the most pronounced or descriptive relationships with the selected antonym pairs, 
                providing insights that can be instrumental in your analytical endeavors. Moreover, the settings offer the flexibility to customize the plot further by reordering elements and focusing on the top-k most discriminative antonym pairs, 
                allowing for a more tailored and insightful analysis.
    """)
    st.image("media/md_plot.png", caption="Fig. 6 Most Discriminative Plot")

# Expert Page

st.write("## Expert")
with st.expander("Description", expanded=False):

    st.image("media/expert_overview_2.png", caption="Fig.7 Overview Expert Page")

    st.markdown("""The Expert Page is initially structured into three different compartments and will change
                dependent on user input, calculation and execution. It is primarily made up of the settings bar 🔴, 
                the antonyms selection 🟢, and the subject choice 🔵 (See Fig.7). An examplary workflow could look as follows:
    """)

    st.markdown("""
                1. (Optional) Download Antonyms Template
                2. Upload Antonyms File
                4. Provide Subject Words (Simple) or Upload Subject Words File (Advanced)
                5. Select Language Model
                6. Adjust Visualization Settings
                7. Execute Analysis
                8. (Optional) Save Plots
                9. (Optional) Download Analysis Results in CSV-Format   
                
    """)

    st.write("### Language Models")
    st.markdown("""On the Expert Page, you'll find a specialized setting in the menu that allows you to choose from various language models. 
                These models include "bert-base-uncased," "gpt2," and "roberta-base." Each of these models offers distinct capabilities and nuances for language processing tasks. 
                It's important to consider your specific needs and objectives when selecting a language model. 
                Keep in mind that the choice of language model can significantly impact the results and performance of your text analysis tasks. 
                Therefore, it's essential to make an informed decision based on your project requirements and desired outcomes.
    """)

    st.write("### Antonyms")
    st.markdown("""On the Expert Page, the antonyms section serves as a powerful input mechanism designed to facilitate the creation of multiple dimensions for evaluating subject words in later analyses. 
                Here, users benefit from advanced capabilities, including the option to upload a file containing a substantial number of antonym pairs for analysis. 
                This feature streamlines the process of working with large datasets of antonyms. 
                Users can then delve deeper into their antonym pairs by accessing a specialized data editor that allows for comprehensive display and editing. 
                Furthermore, to facilitate this process, a downloadable template is available, ensuring that the uploaded file adheres to the required format for seamless integration into the system.
    """)

    st.warning("""Please note that the data editor is a newly added feature in Streamlit and may occasionally result in errors, especially for larger editing tasks. 
                For extensive editing, we recommend using the downloaded template provided for a more seamless experience.
    """)
    
    st.video("media/expert_antonyms.webm")

    st.write("### Subjects")
    st.markdown("""The subjects section on the Expert Page provides users with a versatile input mechanism for subject words intended for in-depth analysis, 
                which is evaluated based on the previously defined antonym pairs. Users can opt for the familiar and straightforward subject word input method, 
                akin to the one available on the Beginner Page, where they specify their chosen subject words and provide contextual information to assist in deriving meaning within the framework.
    """)

    st.markdown("""In addition to this, an advanced option allows for a more intricate subject word input process, including the ability to upload subject word datasets through file upload. 
                Users can further fine-tune and manage their subject word data using the integrated data editor, ensuring precision and control over their analysis. 
                To facilitate this process, a downloadable template is also available, ensuring that the uploaded file adheres to the required format for seamless integration into the system.
    """)

    st.warning("""Please note that the data editor is a newly added feature in Streamlit and may occasionally result in errors, especially for larger editing tasks. 
                For extensive editing, we recommend using the downloaded template provided for a more seamless experience.
    """)

    st.video("media/expert_subjects.webm")

    st.write("### Visualization")
    st.markdown("""In the visualization section, you have the option to select one of two transformation methods for controlling how SensePOLAR calculates its dimensions. 
                Additionally, you can choose the output plots and configure their related settings. SensePOLAR offers two distinct approaches for transforming its embedding space: 
                a 'base change' method and a 'projection-based' method. To help you visualize this transformation, there are currently four different plot variations available:
    """)

    st.write("##### Standard Plot")
    st.markdown("""The standard plot visually represents the polarity of each subject word with respect to the specified axis formed by antonym pairs. 
                In this plot, you can modify the visualization to your preferences through the customization options provided in the settings. 
                This allows you to determine the order of elements within the plot according to your specific requirements.
    """)
    st.image("media/standard_plot.png", caption="Fig.2 Standard Plot")

    st.write("##### 2D Plot")
    st.markdown("""The 2D plot offers a graphical representation of the polarity of subject words along two distinct axes, 
                with each axis defined by selected antonym pairs. In this plot, you have the flexibility to customize your x-axis and y-axis by choosing from the available antonym pairs. 
                This feature empowers you to define and explore the specific dimensions and relationships you want to analyze within the 2D plot, 
                providing a more tailored and insightful visual representation of the data.
    """)
    st.image("media/2d_plot.png", caption="Fig.3 2D Plot")

    st.write("##### Polar Plot")
    st.markdown("""The polar plot offers a distinctive way of visualizing subject word polarity in relation to antonym pairs, resembling a circular or radial chart. 
                It allows for specific settings related to the display of antonym pairs on its axes. You can select your desired antonym pairs and then choose between two display modes:
    """)

    st.markdown("**1. Solo Display**: Here, antonym pairs are shown individually on each axis, allowing for a detailed analysis of subject word polarity in relation to each antonym pair.")
    st.image("media/polar_plot_solo.png", caption="Fig.4 Polar Plot - Solo")

    st.markdown("**2. Grouped Display**: In this mode, antonym pairs are grouped on each axis. This provides a broader perspective on subject word polarity, considering its relationship to the antonym pair as a whole, without distinguishing individual antonyms.")
    st.image("media/polar_plot_grouped.png", caption="Fig.5 Polar Plot - Grouped")


    st.write("##### Most Discriminative Plot")
    st.markdown("""The most discriminative plot is a valuable tool for visualizing the intricate connections between subject words and antonym pairs. 
                Within this plot, you can readily discern the subject words that have particularly strong associations with specific antonyms and the extent of these associations. 
                This visualization empowers you to pinpoint which words showcase the most pronounced or descriptive relationships with the selected antonym pairs, 
                providing insights that can be instrumental in your analytical endeavors. Moreover, the settings offer the flexibility to customize the plot further by reordering elements and focusing on the top-k most discriminative antonym pairs, 
                allowing for a more tailored and insightful analysis.
    """)
    st.image("media/md_plot.png", caption="Fig.6 Most Discriminative Plot")

# Visualizer Page

st.write("## Visualizer")
with st.expander("Description", expanded=False):

    st.image("media/visualizer_overview.png", caption="Fig.8 Overview Visualizer Page")

    st.markdown("""The Visualizer Page is structured into two different compartments and will change
                dependent on user input, calculation and execution. It is primarily made up of the settings bar 🔴 and 
                the result upload 🟢 (See Fig.9). This page serves as a means to quickly upload and recapitulate previous results obtained from the Beginner or Expert Page without repetitive data entry and calculation. 
                This streamlined approach enhances efficiency and ensures that users can easily visualize, explore, and build upon prior analyses.
    """)

    st.warning("""Please note that the data editor is a newly added feature in Streamlit and may occasionally result in errors, especially for larger editing tasks. 
                For extensive editing, we recommend using the downloaded template provided for a more seamless experience.
    """)

    st.write("### Visualization")
    st.markdown("""In the visualization section, you have the option to select one of two transformation methods for controlling how SensePOLAR calculates its dimensions. 
                Additionally, you can choose the output plots and configure their related settings. SensePOLAR offers two distinct approaches for transforming its embedding space: 
                a 'base change' method and a 'projection-based' method. To help you visualize this transformation, there are currently four different plot variations available:
    """)

    st.write("##### Standard Plot")
    st.markdown("""The standard plot visually represents the polarity of each subject word with respect to the specified axis formed by antonym pairs. 
                In this plot, you can tailor the visualization to your preferences through the customization options provided in the settings. 
                This allows you to determine the order of elements within the plot according to your specific requirements.
    """)
    st.image("media/standard_plot.png", caption="Fig.2 Standard Plot")

    st.write("##### 2D Plot")
    st.markdown("""The 2D plot offers a graphical representation of the polarity of subject words along two distinct axes, 
                with each axis defined by selected antonym pairs. In this plot, you have the flexibility to customize your x-axis and y-axis by choosing from the available antonym pairs. 
                This feature empowers you to define and explore the specific dimensions and relationships you want to analyze within the 2D plot, 
                providing a more tailored and insightful visual representation of the data.
    """)
    st.image("media/2d_plot.png", caption="Fig.3 2D Plot")

    st.write("##### Polar Plot")
    st.markdown("""The polar plot offers a distinctive way of visualizing subject word polarity in relation to antonym pairs, resembling a circular or radial chart. 
                It allows for specific settings related to the display of antonym pairs on its axes. You can select your desired antonym pairs and then choose between two display modes:
    """)

    st.markdown("**1. Solo Display**: Here, antonym pairs are shown individually on each axis, allowing for a detailed analysis of subject word polarity in relation to each antonym pair.")
    st.image("media/polar_plot_solo.png", caption="Fig.4 Polar Plot - Solo")

    st.markdown("**2. Grouped Display**: In this mode, antonym pairs are grouped on each axis. This provides a broader perspective on subject word polarity, considering its relationship to the antonym pair as a whole, without distinguishing individual antonyms.")
    st.image("media/polar_plot_grouped.png", caption="Fig.5 Polar Plot - Grouped")

    st.write("##### Most Discriminative Plot")
    st.markdown("""The most discriminative plot is a valuable tool for visualizing the intricate connections between subject words and antonym pairs. 
                Within this plot, you can readily discern the subject words that have particularly strong associations with specific antonyms and the extent of these associations. 
                This visualization empowers you to pinpoint which words showcase the most pronounced or descriptive relationships with the selected antonym pairs, 
                providing insights that can be instrumental in your analytical endeavors. Moreover, the settings offer the flexibility to customize the plot further by reordering elements and focusing on the top-k most discriminative antonym pairs, 
                allowing for a more tailored and insightful analysis.
    """)
    st.image("media/md_plot.png", caption="Fig.6 Most Discriminative Plot")

@st.cache_resource
def load_bert_model(model_name='bert-base-uncased'):
    """
    # Load Bert model.
    """
    return BERTWordEmbeddings(model_name=model_name)

# Load bert model into cache so beginner and expert page are more performant
model = load_bert_model()

