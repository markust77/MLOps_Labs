import streamlit as st

st.title("Hallo Streamlit 👋")
st.write("Das ist mein erstes Streamlit-Skript.")

name = st.text_input("Wie heißt du?")
if name:
    st.write(f"Hallo {name}!")

x = st.slider("Wähle eine Zahl", 0, 100, 25)
st.write(x, "im Quadrat ist", x * x)
    
if st.button("Klick mich"):
    st.balloons()
