import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from millify import millify



st.set_page_config(
    page_title="Car Price Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="centered")

st.title(" Car Sales Dashboard & Price Predictor")
st.markdown("""
Welcome to the **Car Sales Analytics Dashboard**  

This Dashboard lets you:
- Explore second-hand car market trends  
- Filter cars by year, price, manufacturer, and fuel type  
- Visualize insights like price trends, brand positioning, and depreciation  
- Predict the resale value of a specific car  

Use the **sidebar filters** to customize the analysis.
""") 

tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predictor"])



df = pd.read_csv("car_sales_data.csv")  

# rebuild category codes
df['Manufacturer_num'] = df['Manufacturer'].astype('category').cat.codes
df['Model_num']        = df['Model'].astype('category').cat.codes
df['Fuel_Type_num']    = df['Fuel type'].astype('category').cat.codes

# loading trained model
with open("car_price_model_WithOutliers.pkl", "rb") as f:
    final_model = pickle.load(f)

#  The same predict_price function in the notebook

def predict_price(engine_size, year, mileage, fuel_type, manufacturer, model):
    # make dictionaries that convert category text --> numeric codes
    fuel_map = {cat: code for code, cat in enumerate(df['Fuel type'].astype('category').cat.categories)}
    manu_map = {cat: code for code, cat in enumerate(df['Manufacturer'].astype('category').cat.categories)}
    model_map = {cat: code for code, cat in enumerate(df['Model'].astype('category').cat.categories)}

    # change user input (text) into same numbers we used in training
    fuel_val = fuel_map.get(fuel_type, 0)      # example: Petrol --> 2
    manu_val = manu_map.get(manufacturer, 0)
    model_val = model_map.get(model, 0)

    # put features together into one row (1.0 > intercept)
    x_new = np.array([[1.0, engine_size, year, mileage, fuel_val, manu_val, model_val]])

    # run prediction
    pred = final_model.predict(x_new)[0]

    return pred

with tab2:
    st.subheader("Car Price Prediction")

    manufacturer = st.selectbox("Brand of Car", sorted(df["Manufacturer"].unique()))
    models_filtered = df.loc[df["Manufacturer"]==manufacturer, "Model"].unique()
    if len(models_filtered) == 0:
        models_filtered = df["Model"].unique()
    model = st.selectbox("Model", sorted(models_filtered))
    fuel_type = st.selectbox("Fuel type", sorted(df["Fuel type"].unique()))

    c1, c2 = st.columns(2)
    with c1:
        engine_size = st.number_input("Engine size ", min_value=0.5, max_value=8.0, value=1.6, step=0.1)
        mileage     = st.number_input("Mileage", min_value=0, max_value=1_000_000, value=60000, step=500)
    with c2:
        year        = st.number_input("Year of manufacture", min_value=1950, max_value=2025, value=2018, step=1)

    if st.button("Predict Price", use_container_width=True):
        pred = predict_price(engine_size, year, mileage, fuel_type, manufacturer, model)
        st.success(f"Estimated price: **${pred:,.2f}**")

with tab1:
   # Sidebar Filters 
   st.sidebar.header("  Filters")

   # Min & Max from data
   y_min = int(df["Year of manufacture"].min())
   y_max = int(df["Year of manufacture"].max())
   p_min = int(df["Price"].min())
   p_max = int(df["Price"].max())

   # Year of Manufacture
   year_range = st.sidebar.slider(
       "Year of Manufacture",
       min_value=y_min, max_value=y_max,
       value=(y_min, y_max),  
       step=1
       )

   # Price Range
   price_range = st.sidebar.slider(
       "Price Range",
       min_value=p_min, max_value=p_max,
       value=(p_min, p_max),  # default 
       step=500  
       )

   # Manufacturer + Fuel filters
   manufacturer_filter = st.sidebar.multiselect("Manufacturer", df["Manufacturer"].unique())
   fuel_filter = st.sidebar.multiselect("Fuel Type", df["Fuel type"].unique())

   # Apply filters 
   filtered_df = df[
        df["Year of manufacture"].between(year_range[0], year_range[1]) &
        df["Price"].between(price_range[0], price_range[1])
        ]

   if manufacturer_filter:
        filtered_df = filtered_df[filtered_df["Manufacturer"].isin(manufacturer_filter)]
   if fuel_filter:
        filtered_df = filtered_df[filtered_df["Fuel type"].isin(fuel_filter)]


# Dashboard Visuals
   st.subheader("Data Insights")

   # Some info (Might added in KPIs)
   avg_price = filtered_df['Price'].mean()
   median_mileage = filtered_df["Mileage"].median()
   brands_count = filtered_df["Manufacturer"].nunique()
   max_Engine = filtered_df['Engine size'].max()

   #  KPIs 
   k1, k2, k3, k4 = st.columns(4)

   # Cars Count
   k1.metric("Cars", f"{len(filtered_df):,}")

   # Total Value
   total_value = filtered_df["Price"].sum()
   k2.metric("Total Price", f"£{millify(total_value, precision=2)}")
   
   # Models count
   models_count = filtered_df["Model"].nunique()
   k3.metric("Models Count", f"{models_count:,}")

   # The Top brand
   top_brand = filtered_df["Manufacturer"].value_counts().idxmax() # The id of top brand
   top_brand_count = filtered_df["Manufacturer"].value_counts().max() # Take the num of counts
   k4.metric("Top Brand", top_brand, f"{top_brand_count}")


   # -------------------------------------------------------------------- 
   # Plots 

   # -------------------------------------------------------------------- 
   # Sort Manufacturers by avg prices
   st.markdown("### Manufacturers by Avg Price")
   manu_price = filtered_df.groupby("Manufacturer")["Price"].mean().nlargest(5).reset_index()
   fig = px.bar(manu_price, x="Price", y="Manufacturer", orientation="h", text_auto=".0f",
                     color="Price", color_continuous_scale="Blues")
   st.plotly_chart(fig, use_container_width=True)

   # -------------------------------------------------------------------- 
   # Price vs Year of manufacture
   st.markdown("### Price Trend by Year ")
   year_price = filtered_df.groupby("Year of manufacture")["Price"].mean().reset_index()
   fig = px.line(year_price, x="Year of manufacture", y="Price", markers=True)
   st.plotly_chart(fig, use_container_width=True)

   # -------------------------------------------------------------------- 
   st.markdown("### Mileage vs Price **(Fuel Highlighted)**")
   fig = px.scatter(filtered_df, x="Mileage", y="Price", color="Fuel type",
                         hover_data=["Manufacturer", "Model"], opacity=0.6)
   st.plotly_chart(fig, use_container_width=True)

   st.markdown("### Engine Size vs Price **(Trendline)**")
   fig = px.scatter(filtered_df, x="Engine size", y="Price", color="Manufacturer",
                         hover_data=["Model"], trendline="ols", opacity=0.6)
   st.plotly_chart(fig, use_container_width=True)

   # -------------------------------------------------------------------- 
   # PIE CHART
   st.markdown("### Fuel Type over Total Price")
   #fuel_share = filtered_df["Fuel type"].value_counts().reset_index() # Count num of each fuel type
   fuel_share = filtered_df.groupby("Fuel type")["Price"].sum().reset_index() # Count the sum of prices for each fuel type
   fuel_share.columns = ["Fuel type", "Total Value"] # Put them in df

   fig = px.pie(
    fuel_share, names="Fuel type", values="Total Value",
    color="Fuel type", hole=0
    )
   # Showing the percent & label in the pie chart
   fig.update_traces(textinfo="percent+label", pull=[0.05]*len(fuel_share))
   fig.update_layout(margin=dict(l=10, r=20, t=50, b=10))
   st.plotly_chart(fig, use_container_width=True)


   # -------------------------------------------------------------------- 
   # Total Price over years

   st.markdown("### Total Price Over Years")
   
   # Group by Year of manufacture and sum prices
   price_by_year = (filtered_df.groupby("Year of manufacture")["Price"].sum().reset_index())
   fig = px.line(price_by_year, x="Year of manufacture", y="Price", markers=True,
    labels={"Price": "Total Price", "Year of manufacture": "Year"},)
   fig.update_traces(line=dict(width=3, color="orange"))
   fig.update_layout( yaxis_tickformat="~s",   # format numbers as 100M, 600M
    margin=dict(l=20, r=20, t=50, b=20),)
   st.plotly_chart(fig, use_container_width=True)

   ""
   ""
   
   """
   ## Raw data
   """

   df

# Data Source

   st.markdown("---")  # horizontal line for separation
   st.markdown(
     "📂 **Data Source:** [Mock Dataset of Second-Hand Car Sales](https://www.kaggle.com/datasets/msnbehdani/mock-dataset-of-second-hand-car-sales)"
   )

