from urllib.parse import unquote

import arabic_reshaper
import streamlit as st
from bidi.algorithm import get_display

from html_utils import ga
from utils import annotate_answer, get_results, shorten_text

ga()

# ~~CHANGE~~ rename app
st.set_page_config(
    page_title="MentalHealth QA App",
    page_icon="📖",
    initial_sidebar_state="expanded"
    # layout="wide"
)
# ~~CHANGE~~ remove footer
# footer()


rtl = lambda w: get_display(f"{arabic_reshaper.reshape(w)}")


_, col1, _ = st.beta_columns(3)

with col1:
    # ~~CHANGE~~ remove logo
    # st.image("is2alni_logo.png", width=200)
    st.title("إسألني أي شيء")

st.markdown(
    """
<style>
p, div, input, label {
  text-align: right;
}
</style>
    """,
    unsafe_allow_html=True,
)

# ~~CHANGE~~ rename app and change logo on sidebar
st.sidebar.header("MentalHealth QA App")
st.sidebar.image("logo.png", width=150)
st.sidebar.write("Powered by [AraELECTRA](https://github.com/aub-mind/arabert)")
st.sidebar.write("\n")
n_answers = st.sidebar.slider(
    "عدد الأجوبة", min_value=1, max_value=10, value=2, step=1
)

question = st.text_input("", value="")
if "؟" not in question:
    question += "؟"

run_query = st.button("أجب")
if run_query:
    # https://discuss.streamlit.io/t/showing-a-gif-while-st-spinner-runs/5084
    with st.spinner("... جاري البحث "):
        results_dict = get_results(question)

    if len(results_dict) > 0:
        st.write("## :الأجابات هي")
        for result in results_dict["results"][:n_answers]:
            annotate_answer(result)
            # f"[**المصدر**](<{result['link']}>)"
    else:
        st.write("## 😞 ليس لدي جواب")
