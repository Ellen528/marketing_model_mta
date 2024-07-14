import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mta_functions import * 


# ---------- steamlit flow ---------- 
st.markdown("""
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #8B4513;
            text-align: center;
            font-family: 'Courier New', Courier, monospace;
        }
        .sub-title {
            font-size: 24px;
            font-weight: bold;
            color: #A0522D;
            margin-top: 20px;
            font-family: 'Courier New', Courier, monospace;
        }
        .content-block {
            background-color: #FFF8DC;
            color: #000000;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 2px 2px 5px #A0522D;
            font-family: 'Courier New', Courier, monospace;
        }
        .banner-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        .text {
            font-family: 'Courier New', Courier, monospace;
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

# Title of the app
st.title("Channel Attribution Tools")

# Description of the page with styled HTML for an old-fashioned game style
# Introduction Block
st.markdown("<div class='sub-title'>Introduction</div>", unsafe_allow_html=True)
st.markdown("""
    <div class='content-block text'>
        Welcome to the Channel Attribution Tools application.
        This tool allows you to calculate marketing channel attribution for conversions using four different methods: 
        position-based, last-touch, time decay, and Markov chain.
    </div>
""", unsafe_allow_html=True)

# Sidebar for data loading
with st.sidebar:
    st.header("Data Upload")
    # File uploader for user's CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    demo_file_path = './data/data.csv.gz'
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        path_column = st.selectbox('Select the column for paths:', data.columns)
        total_conversions_column = st.selectbox('Select the column for total conversions:', data.columns)
        total_null_column = st.selectbox('Select the column for total nulls:', data.columns)
        total_conversion_value_column = st.selectbox('Select the column for total conversion value:', data.columns)
        # Rename the columns to standard names expected by other functions
        data.rename(columns={
            path_column: 'path',
            total_conversions_column: 'total_conversions',
            total_null_column: 'total_null',
            total_conversion_value_column: 'total_conversion_value'
        }, inplace=True)
        st.session_state['data'] = data  # Store data in session state
    elif st.button("Load Demo Data"):
        data = pd.read_csv(demo_file_path)
        st.session_state['data'] = data  # Store demo data in session state

# Check if data is loaded and process it
st.header("Preview of Processed Data:")
if 'data' in st.session_state:
    multi_channel_data, conv_multi_channel_data = process_data(st.session_state['data'])
    st.write("Multi-Channel Data:")
    st.dataframe(multi_channel_data)  # Adjust width to match other content # Adjust width to match other content

# Method Parameters
st.header("Method Parameters")

st.subheader("Position-Based Attribution")
first_touch_pct = st.slider("First Touch Percentage", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
last_touch_pct = st.slider("Last Touch Percentage", min_value=0.0, max_value=1.0, value=0.4, step=0.1)

st.subheader("Time Decay Attribution")
time_decay_lambda = st.slider("Lambda for Time Decay", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.subheader("Markov Chain Attribution")
markov_order = st.selectbox("Markov Model Order", ('1','2','3','4'))

# Run main function and display results
if st.button("Run Analysis"):
    if 'data' in st.session_state:
        data = st.session_state['data']
    combined_df = calculate_attribution_table(data, first_touch_pct, last_touch_pct, time_decay_lambda)  # Ensure this function uses the parameters and returns the DataFrame
    for col in combined_df.columns:
        if col != 'Channel':
            combined_df[col] = combined_df[col].apply(lambda x: round(x,1))
    # Transform the DataFrame to long format
    combine_long_df = combined_df.melt(id_vars=['Channel'], 
                                       var_name='Method', 
                                       value_name='Attribution')

    st.write("Attribution Analysis Results:")
    st.dataframe(combined_df, width=700)  # Adjust width to match other content
    # Set the aesthetic style of the plots
    plt.style.use('ggplot')
    # Plotting with Seaborn with improved style and color palette
    plt.figure(figsize=(12, 6))  # Adjust the figure size to your preference
    bar_plot = sns.barplot(data=combine_long_df, x='Channel', y='Attribution', hue='Method', palette='Set2')

    # Customizing the plot to match the provided style
    bar_plot.set_title('Total Conversion Value', fontsize=20)
    bar_plot.set_xlabel('Channel', fontsize=18)
    bar_plot.set_ylabel('Attribution Value', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Attribution Method', fontsize=12, title_fontsize='13')

    st.pyplot(plt)  # Display the plot

    # Download button for the DataFrame
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='combined_attribution_data.csv',
        mime='text/csv',
    )