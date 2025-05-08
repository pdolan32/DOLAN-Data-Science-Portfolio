# Spotify Streams App

## Overview
The **Spotify Streams App** is a simple Streamlit application that displays the top ten most-streamed songs on Spotify for February 6th, 2025. The app provides an interactive way to explore streaming data, allowing users to view key details about each song and customize their display preferences.

Some notable features of the application include:
- **Data Display**: a dataframe showcasing the top ten most-streamed songs, including: song title, main artist(s), featured artist(s) (if applicable), daily stream count, and total stream count.
- **Interactivity**: the user is able to toggle between displaying daily stream count and total stream count using buttons and select a specific song from a drop-down menu to view its details.

## Instructions

### Prerequisites
Ensure you have the following installed:
- Python (3.12.7 recommended)
- pip (Python package manager)
- Streamlit (if not installed, see below)

### Running the Application

1. **Install Dependencies**:

    Make sure you have Python 3.12.7 installed, then run:

    ```bash
    pip install streamlit pandas
    ```

2. **Get the Data**:  
Download the `spotify_data.csv` dataset from my data folder and place it in the `data` folder within your project directory.


## Running the App
To launch the application, use the following command in your terminal:

```bash
streamlit run basic_streamlit_app/spotify_streams_app.py
```

The app should open automatically in your default web browser.

