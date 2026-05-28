import streamlit as st

def process():
    with st.chat_message("user"):
        st.write("hello")

cols = st.columns(2)
with cols[0]:
    if st.button("click"):
        process()
