from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle

app = FastAPI()

# Load datasets with predictions
wholesale_df = pd.read_csv("wholesale_price_predictions.csv")
retail_df = pd.read_csv("retail_price_predictions.csv")
market_df = pd.read_csv("market_recommendations.csv")

# Load mappings
with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

# Extract mappings
crop_mapping = mappings.get("Crop", {})
reverse_crop_mapping = mappings.get("Crop_Reverse", {})
county_mapping = mappings.get("County", {})
reverse_county_mapping = mappings.get("County_Reverse", {})
market_reverse_mapping = mappings.get("Market_Reverse", {})

def get_price_prediction(crop: str, county: str, price_type: str):
    """Retrieve predicted price and round to 2 decimals."""
    crop_id = crop_mapping.get(crop)
    county_id = county_mapping.get(county)

    if crop_id is None or county_id is None:
        return None  # Invalid crop or county

    df = wholesale_df if price_type == "wholesale" else retail_df
    result = df[(df['Crop_ID'] == crop_id) & (df['County_ID'] == county_id)]
    
    if result.empty:
        return None
    
    price = result.iloc[0]['Wholesale_Prediction'] if price_type == "wholesale" else result.iloc[0]['Retail_Prediction']
    return round(price, 2)  # Round to 2 decimals

def get_top_markets(county: str):
    """Retrieve top 3 recommended markets from saved predictions."""
    county_id = county_mapping.get(county)

    if county_id is None:
        return None  # Invalid county

    result = market_df[market_df['County_ID'] == county_id]
    if result.empty:
        return None

    # Decode top 3 market names
    top_markets = result['Market_Recommendation'].head(3).tolist()
    return [market_reverse_mapping.get(m, "Unknown Market") for m in top_markets]

@app.get("/predict/")
def predict_price(crop: str, county: str, price_type: str):
    """API endpoint to return price prediction and recommended markets."""
    if price_type not in ["wholesale", "retail"]:
        raise HTTPException(status_code=400, detail="⚠️ Invalid price_type. Choose 'wholesale' or 'retail'.")

    price = get_price_prediction(crop, county, price_type)
    if price is None:
        raise HTTPException(status_code=404, detail="❌ No price prediction available for the given crop and county.")
    
    markets = get_top_markets(county)
    if not markets:
        markets = ["⚠️ No recommended markets found"]

    return {
        "🌾 Crop": crop,
        "📍 County": county,
        "💰 Price Type": price_type.capitalize(),
        "💵 Predicted Price": f"{price} KES",
        "🛒 Recommended Markets": markets
    }
