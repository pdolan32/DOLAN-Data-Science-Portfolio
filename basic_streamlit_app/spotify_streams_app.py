import streamlit as st
import pandas as pd

df = pd.read_csv("basic_streamlit_app/data/spotify_data.csv")
df.index = range(1, len(df) + 1)

columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Daily Streams']  # Default to daily streams

st.title("The Most Streamed Spotify Songs for 2/6/25")

st.subheader("This app showcases the top ten most-streamed songs on Spotify for February 6th, 2025.")
st.write("The below dataframe displays the songs' titles, the leading and featured artists, as well as the songs' daily streams. To see the songs' total streams, click the 'Toggle Total Streams Button.")


if st.button("Toggle Daily Streams"):
    columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Daily Streams']
else:
    columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Total Streams']

if st.button("Toggle Total Streams"):
    columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Total Streams']
else:
    columns_to_show = ['Song Title', 'Lead Artist', 'Featured Artist(s)', 'Daily Streams']

stream_data = df[columns_to_show]

st.dataframe(stream_data)

st.write("The below dropdown menu allows you to isolate the streaming data for a particular song in the top ten.")

song = st.selectbox("Select a song", df["Song Title"].unique())
filtered_df = df[df["Song Title"] == song]
st.write(f"Streaming Data for {song}:")
st.dataframe(filtered_df)
# To open the app, run the following: streamlit run basic_streamlit_app/spotify_streams_app.py
