# Tidy Data Project

## Project Overview

In this project, I work through the process of cleaning and tidying data, using various functions to reshape and transform a dataset into a tidier format. Specifically, I focus on a dataset of medalists from the 2008 Summer Olympics. Once the dataset is tidied, I conduct basic exploratory data analysis.

The goal of this project is to transform a messy dataset into a tidy one using functions learned in class, such as .melt() and .str.split()., and glean insights from the tidy dataset.

The principles of a tidy dataset are as follows: 
- each variable is represented by a column: each column corresponds to a specific variable or feature, such as 'age' or 'height'

- each observation is represented by a row: each row corresponds to a single data point or observation, such as an individual athlete's height
  
- each observational unit is represented by a table: different units of analysis, such as individual athletes, events, or teams, are organized into distinct table.

These tidy dataset principles provides a structured and standardized way of organizing data, making datasets easier to clean, analyze, and visualize.

## Instructions
### Prerequisites
Ensure you have the following installed or downloaded:
- Python (3.12.7 recommended)
- pip (Python package manager)
- Jupyter Notebook (version 7.3.2 recommended)
- olympics_08_medalists.csv (see Dataset Description below for information and instructions)

### Running the Application

1. **Install Libraries**:

    Make sure you have Python 3.12.7 installed, then run the following in an empty Jupyter Notebook file:

    ```bash
    !pip install pandas matplotlib seaborn
    ```
2. **Download the Jupyter Notebook**:

    From the TidyData-Project repository folder, download the Jupyter Notebook file titled DOLAN_TidyData-Project.ipynb.        Once this notebook is downloaded, upload this notebook to your personal Jupyter Notebook from the homepage.

3. **Open the Jupyter Notebook**:

   In your Jupyter Notebook homepage, navigate to the DOLAN_TidyData-Project.ipynb file and open it.

4. **Run the Code**:

At the top right corner of your Jupyter Notebook screen, you should see multiple notebook headers. Click on the 'Run'header, and a dropdown menu should appear. Within this dropdown menu, click 'Run All Cells,' and the project output should be available to view and read. Alternatively, you may run each cell individually by first clicking on the cell and then clicking 'Run Selected Cell' in the same dropdown menu as earlier.

## Dataset Description

The dataset used in this project, 'olympics_08_medalists.csv,' is a large dataset, with the columns consisting of each Olympic event at the 2008 Summer Olympic Games (both male and female, if applicable), and the rows consisting of each medalist. The data outlines the 'type' of medal each medalist achieved in his or her respective event: bronze, silver, or gold. This particular dataset is sourced from [GitHub](https://edjnet.github.io/OlympicsGoNUTS/2008/), with the original data coming from the European Data Journalism Network.

Before running the Tidy Data Project program, ensure that you have the dataset is downloaded and uploaded to your Jupyter Notebook. Click [HERE](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/TidyData-Project/olympics_08_medalists.csv) to access the dataset and download it.
 
## References for Additional Reading

To better understand some of the tidy-data functions applied in this project, consider reading the [Pandas Cheat Sheet](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/TidyData-Project/Pandas_Cheat_Sheet.pdf).

To better understand the principles of tidy datasets, consider reading the [Tidy Data article](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/TidyData-Project/tidy-data.pdf).

## Visualizations

#### A tidy dataset enables in-depth analysis, making it possible to uncover insights, such as those in the visuals below!

<img width="970" alt="TidyData_BarChart" src="https://github.com/user-attachments/assets/80742f0a-81aa-45c2-bf63-d77e5619e418" />
<img width="500" alt="Screenshot 2025-03-13 at 7 57 01 PM" src="https://github.com/user-attachments/assets/cdec4969-f288-47b4-8512-c3c548d0aaeb" />
<img width="465" alt="Screenshot 2025-03-13 at 7 57 13 PM" src="https://github.com/user-attachments/assets/db026cf1-fef2-4cea-a791-6408773ce4fe" />
