# Tidy Data Project 🧼

## Overview

In this project, I work through the process of cleaning and tidying data, using various functions to reshape and transform a dataset into a tidier format. Specifically, I focus on a dataset of medalists from the 2008 Summer Olympics. Once the dataset is tidied, I conduct basic exploratory data analysis.

The goal of this project is to transform a messy dataset into a tidy one using functions learned in class, such as .melt() and .str.split()., and glean insights from the tidy dataset.

The principles of a tidy dataset are as follows: 
- Each variable is represented by a column: each column corresponds to a specific variable or feature, such as 'age' or 'height'

- Each observation is represented by a row: each row corresponds to a single data point or observation, such as an individual athlete's height
  
- Each observational unit is represented by a table: different units of analysis, such as individual athletes, events, or teams, are organized into distinct table.

## Instructions
### Prerequisites
Ensure you have the following installed:
- Python (3.12.7 recommended)
- pip (Python package manager)

### Installation

1. **Install Libraries**:

    Make sure you have Python 3.12.7 installed, then run:

    ```bash
    !pip install pandas matplotlib seaborn
    ```

## Dataset Description

The dataset used in this project, 'olympics_08_medalists.csv,' is a large dataset, with the columns consisting of each Olympic event at the 2008 Summer Olympic Games (both male and female, if applicable), and the rows consisting of each medalist. The data outlines the 'type' of medal each medalist achieved in his or her respective event: bronze, silver, or gold. This particular dataset is sourced from [GitHub](https://edjnet.github.io/OlympicsGoNUTS/2008/), with the original data coming from the European Data Journalism Network.

Before running the Tidy Data Project program, ensure that you have the dataset is downloaded and uploaded to your Jupyter Notebook.

## References
  
