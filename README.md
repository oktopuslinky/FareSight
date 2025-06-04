# FareSight - About
FareSight is an innovative application that leverages machine learning to predict high-demand taxi zones in New York City at any given time during the year, giving rideshare drivers a competitive edge to maximize their income. By analyzing historical ride data and its patterns, FareSight identifies the most lucrative zones for drivers at any given time, transforming guesswork into data-driven decisions.

# Tech Stack
<ul>
  <li>Machine Learning: Python and PyTorch for predictive modeling.</li>
  <li>Backend: FastAPI for high-performance, asynchronous API endpoints.</li>
  <li>Frontend: HTML/CSS/JavaScript served via Python’s HTTP server for lightweight zone visualization.</li>
  <li>Data Processing: Pandas/NumPy for handling and analyzing ride data.</li>
  <li>Environment: Python 3.8+, Uvicorn for ASGI server implementation.</li>
</ul>

# App Startup Instructions
  Have 2 terminals running simultaneously, and input these commands:
    ```uvicorn main:app --reload --workers 1
    python -m http.server 8080```
  
  Open http://localhost:8080 in a browser

# To Check if Requests are Working:
    http://127.0.0.1:8000/docs#/default/predict_zones_predict_post
    ^^ insert requests and execute

# Data Source Notes
The data used for training the neural network was from this dataset: https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data
