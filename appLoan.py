import streamlit as st
import pandas as pd
import altair as alt
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load the saved pipeline
with open('model_pipeline1.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Load the saved pipeline
# model_pipeline = joblib.load('model_pipeline.pkl')

st.title('Loan Approval Prediction')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('loan.csv')
  df

  st.write('**X**')
  X_raw = df.drop('Loan_Approved', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Loan_Approved
  y_raw

#   Client_ID,Credit_Score,Income,DTI_Ratio,Collateral,Loan_Amount,Loan_Purpose,Employment_Years,Loan_Term_Years,Down_Payment,Existing_Liabilities,Citizenship,Loan_Approved

with st.expander('Data visualization'):
    st.write("Select X and Y axes for visualization")

    # Dropdowns for X and Y axis selection
    x_axis = st.selectbox("Select X-axis", df.columns, index=df.columns.get_loc('Credit_Score'))
    y_axis = st.selectbox("Select Y-axis", df.columns, index=df.columns.get_loc('Income'))

    # Create a Plotly scatter plot
    scatter_chart = px.scatter(
        df, 
        x=x_axis, 
        y=y_axis, 
        color='Loan_Approved',
        hover_data=[x_axis, y_axis, 'Loan_Approved']
    )

    # Enable click events in Plotly
    scatter_chart.update_layout(clickmode='event+select')

    # Display the Plotly chart in Streamlit
    st.plotly_chart(scatter_chart, use_container_width=True)

    # Get clicked data
    selected_point = st.session_state.get('clicked_data')

    if selected_point:
        st.write(f"Clicked point data: {selected_point}")

# This function will handle click events (you can extend it for more complex behavior)
def handle_click(trace, points, selector):
    st.session_state['clicked_data'] = points.customdata

# Adding click handler for the points
scatter_chart.data[0].on_click(handle_click)

with st.sidebar:
    # Title of the app
    st.title("Loan Application Form")

    # Sidebar for user input
    st.sidebar.header("Input Features")

    # Numerical input fields
    client_id = st.sidebar.number_input("Client ID", min_value=1)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850)
    income = st.sidebar.number_input("Income", min_value=0.0, format="%.2f")
    dti_ratio = st.sidebar.number_input("DTI Ratio", min_value=0.0, format="%.2f")
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, format="%.2f")
    employment_years = st.sidebar.number_input("Employment Years", min_value=0, max_value=50)
    loan_term_years = st.sidebar.number_input("Loan Term (Years)", min_value=1, max_value=30)
    down_payment = st.sidebar.number_input("Down Payment", min_value=0.0, format="%.2f")

    # Categorical input fields
    existing_liabilities = st.sidebar.selectbox(
        "Existing Liabilities",
        options=['Personal Loan', 'Credit Card', 'Mortgage', 'None']
    )

    # Option to add a new category
    new_liability = st.sidebar.text_input("Add New Liability Category (if any)")

    # Citizenship input
    citizenship = st.sidebar.selectbox("Citizenship", options=["Yes", "No"])

    # Loan Purpose input
    loan_purpose = st.sidebar.selectbox(
        "Loan Purpose",
        options=['Medical', 'Vacation', 'Home Improvement', 'Business', 'Car']
    )

    # Option to add a new category
    new_purpose = st.sidebar.text_input("Add New Loan Purpose Category (if any)")

    # Collateral input
    collateral = st.sidebar.selectbox(
        "Collateral",
        options=['None', 'House', 'Car']
    )

    # Option to add a new collateral category
    new_collateral = st.sidebar.text_input("Add New Collateral Category (if any)")

        # Button to save the data
    if st.sidebar.button("Submit"):
        # Load existing data if it exists, otherwise create a new DataFrame
        try:
            df = pd.read_csv('loan_data.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=[
                'Client_ID', 'Credit_Score', 'Income', 'DTI_Ratio', 'Loan_Amount',
                'Employment_Years', 'Loan_Term_Years', 'Down_Payment',
                'Existing_Liabilities', 'Citizenship', 'Loan_Purpose', 'Collateral'
            ])

        # Add new categories if provided
        if new_liability:
            existing_liabilities += f", {new_liability}"
        if new_purpose:
            loan_purpose += f", {new_purpose}"
        if new_collateral:
            collateral += f", {new_collateral}"

    # Create a new record and append it to the DataFrame
    new_record = {
        'Client_ID': client_id,
        'Credit_Score': credit_score,
        'Income': income,
        'DTI_Ratio': dti_ratio,
        'Loan_Amount': loan_amount,
        'Employment_Years': employment_years,
        'Loan_Term_Years': loan_term_years,
        'Down_Payment': down_payment,
        'Existing_Liabilities': existing_liabilities,
        'Citizenship': citizenship,
        'Loan_Purpose': loan_purpose,
        'Collateral': collateral
    }


    input_df = pd.DataFrame(new_record, index=[0])
    input_features = pd.concat([input_df, X_raw], axis=0)


    try:
        # Make predictions
        prediction = model_pipeline.predict(input_df)
        # Convert numerical prediction to Yes/No
        prediction_label = "Yes" if prediction[0] == 1 else "No"
        st.success(f"Loan Approval: {prediction_label}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")










    # # # Save the updated DataFrame to the CSV file
    # df.to_csv('loan_data.csv', index=False)

    # st.sidebar.success("Data submitted successfully!")

with st.expander('Input features'):
  st.write('**Input Client Details**')
  input_df
  st.write('**Combined Clients data**')
  input_features

# # Display the current DataFrame
# if st.checkbox("Show current data"):
#     st.write(input_features)



# # Use it to make predictions
# predictions = model_pipeline.predict(input_df)


