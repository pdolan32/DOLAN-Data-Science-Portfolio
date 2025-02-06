import streamlit as st

st.title("This is my first Streamlit app. Hooray!")

if st.button("Click me!"):
    st.write(" You clicked the button! Nice work! ")
else:
    st.write("Click the button to see what happens...")

color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")