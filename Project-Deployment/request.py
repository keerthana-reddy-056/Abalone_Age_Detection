import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Sex':2, 'Length':0.455, 'Diameter':0.365, 'Height':0.095, 'Whole weight':0.514, 'Shucked weight':0.2245, 'Viscera weight':0.101, 'Shell weight':0.15})

print(r.json())
