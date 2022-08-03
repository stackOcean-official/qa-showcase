import streamlit as st
from transformers import pipeline

# function to load model from wherever you need. Sample uses 
@st.cache(allow_output_mutation=True, show_spinner=False)
def use_model():
    model_name = "deepset/gelectra-base-germanquad"
    question_answer_model = pipeline(model=model_name, tokenizer=model_name ,task="question-answering")
    return question_answer_model


# change sample input as needed
sample_question = "Wo wohne ich?"
sample_context = "Mein Name ist Klaus und ich lebe in Berlin"

input_question = st.text_input("Die zu beantwortende Frage:", value=sample_question, max_chars=128)
input_context = st.text_area(label="Der Kontext aus dem die Antwort gefunden werden soll", value=sample_context, height = 200, max_chars=10000)

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
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)