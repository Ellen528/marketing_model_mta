# Multi-Touch Attribution (MTA) Analysis Tool

This repository contains a Streamlit application designed to perform Multi-Touch Attribution (MTA) analysis on marketing data. The application supports various attribution models including Linear, Position-Based, Last Touch, Time Decay, and Markov Chain models.

This tool is based on real-life scenarios. For a detailed wiki, please check the following link:

## Features

- **Data Upload**: Users can upload their own CSV files to analyze conversion paths. Please refer to the snapshot image for the required file format.
- **Attribution Models**: Supports multiple attribution models to analyze the impact of different marketing channels.
- **Interactive Visualizations**: Provides bar charts to compare the effectiveness of different channels based on the selected attribution model.
- **Downloadable Results**: Users can download the results of their analyses as CSV files.

## Exploring the Tool

There are two ways to explore the functionalities of this tool:

### 1. Visit the Deployed Application

For immediate access without the need for local setup, visit the deployed version of the application at [this link](https://marketingmodelmta-qgcagtaspybepuf2942jw5.streamlit.app/).

### 2. Run Locally

To run this application locally, you'll need to have Python installed on your machine. Follow these steps to set up and run the application:

1. **Clone the Repository**

2. **Set Up a Virtual Environment** (Optional but recommended)

3. **Install Dependencies**
pip install -r requirements.txt

4. **Run the Application**
streamlit run mta_app.py

   This will start the Streamlit server and open the application in your default web browser.

## Usage

After launching the application, follow these steps to analyze your data:

1. **Upload Data**: Use the sidebar to upload a CSV file containing your marketing data.
2. **Configure Parameters**: Adjust the parameters for different attribution models as needed.
3. **Run Analysis**: Click the "Run Analysis" button to process the data and view the results.
4. **View and Download Results**: Explore the visualizations and download the results as a CSV file.

## Future Plans

We are continuously working to enhance the capabilities of this tool. Future updates will include the integration of more sophisticated models such as the Additive Hazard model from survival analysis. 

## Contributing

Contributions to this project are welcome! 
