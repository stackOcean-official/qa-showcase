from zmq import NULL
import streamlit as st
from transformers import pipeline

# function to load model from wherever you need. Sample uses
@st.cache(allow_output_mutation=True, show_spinner=False)
def use_model():
    model_name = "deepset/gelectra-base-germanquad"
    question_answer_model = pipeline(
        model=model_name, tokenizer=model_name, task="question-answering"
    )
    return question_answer_model


# change sample input as needed
sample_question = (
    "In welchem Quartal haben wir am meisten Umsatz gemacht?",
    "Wie hat sich unser Gewinn entwickelt?",
    "Wann haben wir am wenigsten Gewinne gemacht?",
)
sample_context = (
    "Der Bericht zeigt, dass das Unternehmen auf einem guten Kurs ist. Unsere"
    " Gewinne sind um 20% gestiegen. Allerdings haben im 3. Quartal letzten"
    " Jahres sehr viel mehr Umsatz gemacht haben, als in den restlichen"
    " Quartalen. Leider haben wir im 1. Quartal so gut wie keinen Umsatz"
    " erzielt."
)
input_option = "In welchem Quartal haben wir am meisten Umsatz gemacht?"

input_question = st.text_input(
    "Frage selbst etwas oder versuche es mit der vorgegebenen Frage:",
    value=input_option,
    max_chars=128,
)
st.write("Andere mögliche Fragen könnten sein:")
st.code("Wie hat sich unser Gewinn entwickelt?")
input_context = st.text_area(
    label=(
        "Schreibe hier den Text aus dem die Antwort gefunden werden soll oder"
        " nimm unseren Beispieltext:"
    ),
    value=sample_context,
    height=200,
    max_chars=10000,
)

# button returns a boolean value
compute = st.button("Beantworte meine Frage")
if compute:
    with st.spinner("Lade Modell und berechne..."):
        # call model
        model = use_model()
        output = model(question=input_question, context=input_context)
        st.subheader(f"Antwort: {output['answer']}")
        st.write(f"Zuversicht: {round(output['score'] * 100, 2)}%")

# remove menu for production
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stCodeBlock div {padding-top: 2px; padding-bottom: 2px;}
.stCodeBlock pre {padding-top: 2px; padding-bottom: 2px;}
.stCodeBlock button {max-height: 20px;}
p {margin-bottom: 4px; font-size: 14px}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
