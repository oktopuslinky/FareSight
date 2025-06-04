'''
HOW TO USE
have 2 terminals:
    uvicorn main:app --reload --workers 1
    python -m http.server 8080

open http://localhost:8080 in browser

TO CHECK IF REQUESTS ARE WORKING
    http://127.0.0.1:8000/docs#/default/predict_zones_predict_post
    ^^ insert requests and execute
'''

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from typing import List
import pickle

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture
class RidePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(RidePredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Global variables for preprocessors and model
onehot = None
scaler = None
y_scaler = None
pulocation_ids = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def load_preprocessors_and_model():
    global onehot, scaler, y_scaler, pulocation_ids, model
    try:
        with open('onehot.pkl', 'rb') as f:
            onehot = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('y_scaler.pkl', 'rb') as f:
            y_scaler = pickle.load(f)
        with open('pulocation_ids.pkl', 'rb') as f:
            pulocation_ids = pickle.load(f)
        
        # Load model
        input_size = len(pulocation_ids) + 4  # One-hot encoded PULocationID + 4 numerical features
        model = RidePredictionModel(input_size).to(device)
        model.load_state_dict(torch.load('ride_prediction_model.pth', map_location=device))
        model.eval()
        
        print("Preprocessors and model loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading preprocessors or model: {e}")
        raise HTTPException(status_code=500, detail="Preprocessor or model files not found")

# Function to predict rides for all zones
def predict_rides_all_zones(hour, day_of_week, month, year):
    global onehot, scaler, y_scaler, pulocation_ids, model
    
    # Create input data
    input_data = pd.DataFrame({
        'PULocationID': pulocation_ids,
        'hour': [hour] * len(pulocation_ids),
        'day_of_week': [day_of_week] * len(pulocation_ids),
        'month': [month] * len(pulocation_ids),
        'year': [year] * len(pulocation_ids)
    })
    
    # Preprocess inputs
    pulocation_enc = onehot.transform(input_data[['PULocationID']])
    numerical = scaler.transform(input_data[['hour', 'day_of_week', 'month', 'year']])
    X = np.hstack([pulocation_enc, numerical])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Inverse transform predictions
    predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Create results
    results = pd.DataFrame({
        'PULocationID': pulocation_ids,
        'Predicted_Rides': predictions
    })
    results = results.sort_values(by='Predicted_Rides', ascending=False).reset_index(drop=True)
    
    return results

# Request and response models
class PredictionRequest(BaseModel):
    time: str  # e.g., "12:00"
    service: str  # "Uber" or "Lyft"
    year: int  # e.g., 2022
    month: int  # 1-12
    day_of_week: int  # 0=Monday, 6=Sunday

class ZonePrediction(BaseModel):
    zoneId: int
    zoneName: str
    predictedRides: float

# Complete zone names dictionary
zone_names = {
    1: "Newark Airport",
    2: "Jamaica Bay",
    3: "Allerton/Pelham Gardens",
    4: "Alphabet City",
    5: "Arden Heights",
    6: "Arrochar/Fort Wadsworth",
    7: "Astoria",
    8: "Astoria Park",
    9: "Auburndale",
    10: "Baisley Park",
    11: "Bath Beach",
    12: "Battery Park",
    13: "Battery Park City",
    14: "Bay Ridge",
    15: "Bay Terrace/Fort Totten",
    16: "Bayside",
    17: "Bedford",
    18: "Bedford Park",
    19: "Bellerose",
    20: "Belmont",
    21: "Bensonhurst East",
    22: "Bensonhurst West",
    23: "Bethpage",
    24: "Bloomfield/Emerson Hill",
    25: "Bloomingdale",
    26: "Boerum Hill",
    27: "Borough Park",
    28: "Breezy Point/Fort Tilden/Riis Beach",
    29: "Brighton Beach",
    30: "Broad Channel",
    31: "Bronx Park",
    32: "Bronxdale",
    33: "Brooklyn Heights",
    34: "Brooklyn Navy Yard",
    35: "Brownsville",
    36: "Bushwick North",
    37: "Bushwick South",
    38: "Cambria Heights",
    39: "Canarsie",
    40: "Carroll Gardens",
    41: "Castle Hill",
    42: "Central Harlem",
    43: "Central Harlem North",
    44: "Central Park",
    45: "Charleston/Tottenville",
    46: "Chinatown",
    47: "City Island",
    48: "Claremont/Bathgate",
    49: "Clinton East",
    50: "Clinton Hill",
    51: "Clinton West",
    52: "Co-Op City",
    53: "Cobble Hill",
    54: "Columbia Street",
    55: "Coney Island",
    56: "Corona",
    57: "Country Club",
    58: "Crotona Park",
    59: "Crotona Park East",
    60: "Crown Heights North",
    61: "Crown Heights South",
    62: "Cypress Hills",
    63: "Douglaston",
    64: "Downtown Brooklyn/MetroTech",
    65: "Dumbo/Vinegar Hill",
    66: "Dyker Heights",
    67: "East Chelsea",
    68: "East Concourse/Concourse Village",
    69: "East Elmhurst",
    70: "East Flatbush/Farragut",
    71: "East Flatbush/Remsen Village",
    72: "East Flushing",
    73: "East Harlem North",
    74: "East Harlem South",
    75: "East New York",
    76: "East New York/Pennsylvania Avenue",
    77: "East Tremont",
    78: "East Village",
    79: "East Williamsburg",
    80: "Eastchester",
    81: "Elmhurst",
    82: "Elmhurst/Maspeth",
    83: "Eltingville/Annadale/Prince's Bay",
    84: "Erasmus",
    85: "Far Rockaway",
    86: "Financial District North",
    87: "Financial District South",
    88: "Flatbush/Ditmas Park",
    89: "Flatiron",
    90: "Flatlands",
    91: "Flushing",
    92: "Flushing Meadows-Corona Park",
    93: "Fordham South",
    94: "Forest Hills",
    95: "Fort Greene",
    96: "Fresh Meadows",
    97: "Freshkills Park",
    98: "Garment District",
    99: "Glen Oaks",
    100: "Glendale",
    101: "Gowanus",
    102: "Gramercy",
    103: "Gravesend",
    104: "Great Kills",
    105: "Great Kills Park",
    106: "Greenpoint",
    107: "Greenwich Village North",
    108: "Greenwich Village South",
    109: "Grymes Hill/Clifton",
    110: "Hamilton Heights",
    111: "Hammels/Arverne",
    112: "Heartland Village/Todt Hill",
    113: "Highbridge",
    114: "Highbridge Park",
    115: "Hillcrest/Pomonok",
    116: "Hollis",
    117: "Homecrest",
    118: "Howard Beach",
    119: "Hudson Sq",
    120: "Hunts Point",
    121: "Inwood",
    122: "Inwood Hill Park",
    123: "Jackson Heights",
    124: "Jamaica",
    125: "Jamaica Estates",
    126: "JFK Airport",
    127: "Kensington",
    128: "Kew Gardens",
    129: "Kew Gardens Hills",
    130: "Kingsbridge Heights",
    131: "Kips Bay",
    132: "LaGuardia Airport",
    133: "Laurelton",
    134: "Lenox Hill East",
    135: "Lenox Hill West",
    136: "Lincoln Square East",
    137: "Lincoln Square West",
    138: "Little Italy/NoLiTa",
    139: "Long Island City/Hunters Point",
    140: "Long Island City/Queens Plaza",
    141: "Longwood",
    142: "Lower East Side",
    143: "Madison",
    144: "Manhattan Beach",
    145: "Manhattan Valley",
    146: "Manhattanville",
    147: "Marble Hill",
    148: "Marine Park/Floyd Bennett Field",
    149: "Marine Park/Mill Basin",
    150: "Mariners Harbor",
    151: "Maspeth",
    152: "Meatpacking/West Village West",
    153: "Melrose South",
    154: "Middle Village",
    155: "Midtown Center",
    156: "Midtown East",
    157: "Midtown North",
    158: "Midtown South",
    159: "Midwood",
    160: "Morningside Heights",
    161: "Morrisania/Melrose",
    162: "Mott Haven/Port Morris",
    163: "Mount Hope",
    164: "Murray Hill",
    165: "Murray Hill-Queens",
    166: "New Dorp/Midland Beach",
    167: "New Springville/Graniteville",
    168: "North Corona",
    169: "Norwood",
    170: "Oakland Gardens",
    171: "Oakwood",
    172: "Ocean Hill",
    173: "Ocean Parkway South",
    174: "Old Astoria",
    175: "Ozone Park",
    176: "Park Slope",
    177: "Parkchester",
    178: "Pelham Bay",
    179: "Pelham Bay Park",
    180: "Pelham Parkway",
    181: "Penn 1",
    182: "Port Richmond",
    183: "Prospect Heights",
    184: "Prospect Lefferts Gardens",
    185: "Prospect Park",
    186: "Queens Village",
    187: "Queensboro Hill",
    188: "Queensbridge/Ravenswood",
    189: "Randalls Island",
    190: "Red Hook",
    191: "Rego Park",
    192: "Richmond Hill",
    193: "Ridgewood",
    194: "Rikers Island",
    195: "Riverdale/North Riverdale/Fieldston",
    196: "Rockaway Park",
    197: "Roosevelt Island",
    198: "Rosebank",
    199: "Rossville/Woodrow",
    200: "Rugby",
    201: "Schuylerville/Edgewater Park",
    202: "Seagate",
    203: "Sheepshead Bay",
    204: "SoHo",
    205: "Soundview/Bruckner",
    206: "Soundview/Castle Hill",
    207: "South Beach/Dongan Hills",
    208: "South Jamaica",
    209: "South Ozone Park",
    210: "South Williamsburg",
    211: "Springfield Gardens North",
    212: "Springfield Gardens South",
    213: "Spuyten Duyvil/Kingsbridge",
    214: "St Albans",
    215: "St George/New Brighton",
    216: "Stapleton",
    217: "Starrett City",
    218: "Steinway",
    219: "Stuyvesant Heights",
    220: "Stuyvesant Town/Peter Cooper Village",
    221: "Sunset Park East",
    222: "Sunset Park West",
    223: "Sutton Place/Turtle Bay",
    224: "Times Sq/Theatre District",
    225: "TriBeCa/Civic Center",
    226: "Two Bridges/Seward Park",
    227: "Union Sq",
    228: "University Heights/Morris Heights",
    229: "Upper East Side North",
    230: "Upper East Side South",
    231: "Upper West Side North",
    232: "Upper West Side South",
    233: "Van Cortlandt Park",
    234: "Van Cortlandt Village",
    235: "Van Nest/Morris Park",
    236: "Vesey/Battery Park City",
    237: "Wakefield",
    238: "Washington Heights North",
    239: "Washington Heights South",
    240: "West Brighton",
    241: "West Chelsea/Hudson Yards",
    242: "West Concourse",
    243: "West Farms/Bronx River",
    244: "West Village",
    245: "Westchester Village/Unionport",
    246: "Westerleigh",
    247: "Whitestone",
    248: "Williamsbridge/Olinville",
    249: "Williamsburg (North Side)",
    250: "Williamsburg (South Side)",
    251: "Windsor Terrace",
    252: "Woodhaven",
    253: "Woodlawn/Wakefield",
    254: "Woodside",
    255: "World Trade Center",
    256: "Yorkville East",
    257: "Yorkville West",
    258: "NV",
    259: "Outside of NYC",
    260: "Outside of NYC",
    261: "Outside of NYC",
    262: "Outside of NYC",
    263: "Outside of NYC",
    264: "Unknown Zone 264",
    265: "Unknown Zone 265"
}

@app.post("/predict", response_model=List[ZonePrediction])
async def predict_zones(request: PredictionRequest):
    try:
        # Parse time
        time_obj = datetime.strptime(request.time, "%H:%M")
        hour = time_obj.hour
        month = request.month
        day_of_week = request.day_of_week
        year = request.year

        # Validate inputs
        if request.service not in ["Uber", "Lyft"]:
            raise HTTPException(status_code=400, detail="Service must be 'Uber' or 'Lyft'")
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
        if not (0 <= day_of_week <= 6):
            raise HTTPException(status_code=400, detail="Day of week must be between 0 (Monday) and 6 (Sunday)")

        # Get predictions
        results = predict_rides_all_zones(hour, day_of_week, month, year)
        
        # Format response (top 5 zones)
        response = []
        for _, row in results.head(5).iterrows():
            zone_id = int(row['PULocationID'])
            zone_name = zone_names.get(zone_id, f"Zone {zone_id}")
            response.append(ZonePrediction(
                zoneId=zone_id,
                zoneName=zone_name,
                predictedRides=float(row['Predicted_Rides'])
            ))
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))