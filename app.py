import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------- Load Mappings ----------
with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

# Mappings: use human-readable names for dropdowns
crop_mapping = mappings["Crop"]            # e.g., {"Maize": 1, "Tomatoes": 2, ...}
county_mapping = mappings["County"]          # e.g., {"Nairobi": 47, "Garissa": 52, ...}
market_reverse_mapping = mappings["Market_Reverse"]  # e.g., {40: "Garissa Main Market", ...}

# ---------- Load Prediction Data ----------
wholesale_df = pd.read_csv("wholesale_price_predictions.csv")
retail_df = pd.read_csv("retail_price_predictions.csv")
market_df = pd.read_csv("market_recommendations.csv")

# ---------- Create Lookup Dictionaries ----------
wholesale_dict = wholesale_df.set_index(["Crop_ID", "County_ID"])["Wholesale_Prediction"].to_dict()
retail_dict = retail_df.set_index(["Crop_ID", "County_ID"])["Retail_Prediction"].to_dict()
market_dict = market_df.groupby("County_ID")["Market_Recommendation"].apply(list).to_dict()

# ---------- Prepare Dropdown Options ----------
crop_names = sorted(crop_mapping.keys())
county_names = sorted(county_mapping.keys())
price_types = ["Wholesale", "Retail"]

# ---------- Streamlit App Layout ----------
st.set_page_config(page_title="🌾 Crop Price Prediction & 🛒 Market Recommendation", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌾 Crop Price Prediction & 🛒 Market Recommendation</h1>", unsafe_allow_html=True)

# ---------- User Selections ----------
st.markdown("### Please select your options:")
col1, col2 = st.columns(2)
with col1:
    selected_crop = st.selectbox("🌾 Select Crop", crop_names)
    selected_county = st.selectbox("📍 Select County", county_names)
with col2:
    selected_price_type = st.radio("💰 Select Price Type", price_types)

# Convert selected names to their encoded IDs using mappings
crop_id = crop_mapping.get(selected_crop)
county_id = county_mapping.get(selected_county)

# ---------- Get Prediction ----------
predicted_price = None
if st.button("🔍 Get Prediction"):
    with st.spinner("Fetching prediction..."):
        if crop_id is None or county_id is None:
            st.error("❌ Invalid selection. Please try again.")
        else:
            # Choose the proper dictionary based on price type
            price_dict = wholesale_dict if selected_price_type.lower() == "wholesale" else retail_dict
            predicted_price = price_dict.get((crop_id, county_id))
            
            if predicted_price is None:
                st.error("❌ No data found. Please check your inputs.")
            else:
                rounded_price = round(predicted_price, 2)
                st.success(f"💰 Predicted {selected_price_type} Price for **{selected_crop}** in **{selected_county}**: **{rounded_price} KES per Kg**")

# ---------- Market Recommendations ----------
st.markdown("### 🛒 Market Recommendations for Selected County")
if county_id is None:
    st.error("❌ Invalid county selection.")
else:
    recommended_market_ids = market_dict.get(county_id, [])
    if isinstance(recommended_market_ids, int):
        recommended_market_ids = [recommended_market_ids]
    unique_market_ids = list(dict.fromkeys(recommended_market_ids))
    top_market_ids = unique_market_ids[:3]
    top_markets = [mappings["Market_Reverse"].get(mid, f"⚠️ Unknown Market ({mid})") for mid in top_market_ids]
    
    if top_markets:
        st.info(f"🏆 Top Recommended Markets: {', '.join(top_markets)}")
    else:
        st.warning("⚠️ No market recommendations available for this county.")

# ---------- Download Prediction Results as CSV ----------
if predicted_price is not None:
    rounded_price = round(predicted_price, 2)
    # Create a DataFrame with the prediction results
    result_df = pd.DataFrame({
        "Crop": [selected_crop],
        "County": [selected_county],
        "Price Type": [selected_price_type],
        "Predicted Price (KES per Kg)": [rounded_price],
        "Recommended Markets": [", ".join(top_markets)]
    })
    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prediction Results as CSV",
        data=csv_data,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

# ---------- Visualization: Top 5 Counties by Predicted Price ----------
st.markdown("## 📊 Price Distribution Across Top 5 Counties")

def plot_price_distribution(selected_crop_id, selected_price_type):
    # Use the correct dataset for visualization
    df = wholesale_df if selected_price_type.lower() == "wholesale" else retail_df
    # Filter dataset for the selected crop
    crop_data = df[df["Crop_ID"] == selected_crop_id]
    if crop_data.empty:
        return None
    # Determine the price column to use
    price_column = "Wholesale_Prediction" if selected_price_type.lower() == "wholesale" else "Retail_Prediction"
    # Group by County_ID and compute average predicted price
    county_prices = crop_data.groupby("County_ID")[price_column].mean().reset_index()
    # Map County_ID back to human-readable names using County_Reverse mapping
    county_prices["County"] = county_prices["County_ID"].map(lambda cid: mappings["County_Reverse"].get(cid, f"Unknown ({cid})"))
    # Sort by predicted price and take the top 5 (lowest prices for clarity)
    top5 = county_prices.nsmallest(5, price_column)
    return top5

price_distribution = plot_price_distribution(crop_id, selected_price_type)
if price_distribution is not None and not price_distribution.empty:
    y_col = "Wholesale_Prediction" if selected_price_type.lower() == "wholesale" else "Retail_Prediction"
    fig = px.bar(price_distribution, x="County", y=y_col, color="County",
                 title=f"📊 Top 5 Counties by Predicted {selected_price_type} Price for {selected_crop}",
                 labels={y_col: f"Predicted {selected_price_type} Price (KES per Kg)"})
    st.plotly_chart(fig)
else:
    st.warning("⚠️ No price distribution data available for the selected crop.")
