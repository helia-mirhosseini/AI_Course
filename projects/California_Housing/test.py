import requests
payload = {
  "MedInc": 10.0, "HouseAge": 1, "AveRooms": 3, "AveBedrms": 1.0,
  "Population": 1000, "Latitude": 34.0, "Longitude": -115.0  # Removed AveOccup here
}
resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
print("Status code:", resp.status_code)
print("Raw text:\n", resp.text)
