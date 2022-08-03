import streamlit as st
from transformers import pipeline

# function to load model from wherever you need. Sample uses 
@st.cache(allow_output_mutation=True, show_spinner=False)
def use_model():
    model_name = "oliverguhr/german-sentiment-bert"
    sentiment_analysis_model = pipeline(model=model_name, tokenizer=model_name)
    return sentiment_analysis_model

# main page
st.title("Titel")
explanation = st.expander("Wie funktioniert es?")
explanation.write("Erklärung wie es funktioniert")
what_else = st.expander("Was kann man noch machen?")
what_else.write("Dinge die man noch machen kann")

# change sample input as needed
sample_input = ("Beispielsatz")

# and change whatever input values and methods are needed
input_text = st.text_area(
    label=(
        "Schreibe einen deutschen Satz, oder versuche es mit dem vorgegebenen"
        " Text"
    ),
    value=sample_input,
    height=80,
    max_chars=128,
)
# button returns a boolean value
compute = st.button("Analysiere Text")
if compute:
    with st.spinner("Lade Modell und berechne..."):
        # call model
        model = use_model()
        output = model(input_text)
        # transform output here as needed
        st.subheader(
            f"Ergebnis: {output}"
        )

# remove menu for production
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)