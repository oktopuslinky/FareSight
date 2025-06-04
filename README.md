# FareSight
Through the use of a trained machine learning model, determine what the best areas are for rideshare drivers to maximize their income.

# APP STARTUP INSTRUCTIONS
  Have 2 terminals running simultaneously, and input these commands:
      uvicorn main:app --reload --workers 1
      python -m http.server 8080
  
  Open http://localhost:8080 in a browser

TO CHECK IF REQUESTS ARE WORKING
    http://127.0.0.1:8000/docs#/default/predict_zones_predict_post
    ^^ insert requests and execute
