import streamlit as st
import pandas as pd
# ================================
# Step 1: Displaying a Simple DataFrame in Streamlit
# ================================
st.subheader("Now, let's look at some data!")
# Creating a simple DataFrame manually
# This helps students understand how to display tabular data in Streamlit.
df = pd.DataFrame({
'Name': ['Alice', 'Bob', 'Charlie', 'David'],
'Age': [25, 30, 35, 40],
'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})
# Displaying the table in Streamlit
# st.dataframe() makes it interactive (sortable, scrollable)
st.write("Here's a simple table:")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique())

filtered_df = df[df["City"] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)

df2 = pd.read_csv("data/sample_data.csv")
st.dataframe(df2)