import requests
from datetime import datetime

def get_zone_predictions(time_str, service, year, month):
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "time": time_str,
        "service": service,
        "year": year,
        "month": month
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        zones = response.json()
        return zones
    except requests.exceptions.RequestException as e:
        print(f"Error fetching predictions: {e}")
        return None

def main():
    # Get user input
    time_str = input("Enter time (HH:MM, e.g., 12:00): ")
    service = input("Enter service (Uber or Lyft): ").capitalize()
    year = int(input("Enter year (e.g., 2022): "))
    month = int(input("Enter month (1-12): "))
    
    # Validate time format
    try:
        datetime.strptime(time_str, "%H:%M")
    except ValueError:
        print("Invalid time format. Please use HH:MM (e.g., 12:00).")
        return
    
    # Get predictions
    zones = get_zone_predictions(time_str, service, year, month)
    
    if zones:
        print(f"\nTop zones for {service} at {time_str} in {month}/{year}:")
        for i, zone in enumerate(zones, 1):
            print(f"{i}. {zone['zoneName']} (Zone ID: {zone['zoneId']}) - Predicted Rides: {zone['predictedRides']:.2f}")

if __name__ == "__main__":
    main()