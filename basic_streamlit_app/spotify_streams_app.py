# This code imports the necessary libraries
import streamlit as st
import pandas as pd

# This code reads in my data csv file and creates a dataframe with the imported data
df = pd.read_csv("basic_streamlit_app/data/spotify_data.csv")
df.index = range(1, len(df) + 1)

# This code defaults the dataframe to display daily streams
columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Daily Streams']  

st.title("The Most Streamed Spotify Songs for 2/6/25 🎶")

st.subheader("This app showcases the top ten most-streamed songs on Spotify for February 6th, 2025.")
st.write("The below dataframe displays the songs' titles, the leading and featured artists, as well as the songs' daily streams. To see the songs' total streams, click the 'Toggle Total Streams Button.")

# This code creates a container for the buttons to group them together
with st.container():
    col1, col2 = st.columns([1, 3])  
    with col1:
        if st.button("Toggle Daily Streams"):
            columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Daily Streams']
    with col2:
        if st.button("Toggle Total Streams"):
            columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Total Streams']


# This code creates a dataframe with particular columns
stream_data = df[columns_to_show]

# This code displays the dataframe created above
st.dataframe(stream_data)

st.write("The below dropdown menu allows you to isolate the streaming data for a particular song in the top ten.")

# This code creates a dropdown menu that allows the user to select an individual song to display the data for.
song = st.selectbox("Select a song", df["Song Title"].unique())
filtered_df = df[df["Song Title"] == song]
st.write(f"Streaming Data for {song}:")
st.dataframe(filtered_df)

# To open the app, run the following: streamlit run basic_streamlit_app/spotify_streams_app.py