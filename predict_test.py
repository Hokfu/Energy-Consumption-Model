import requests

url = 'http://localhost:9696/predict'

energy_consumption = {
    "date" : "2018-01",
    "usage_kwh" : 3.17, 
    "lagging_current_reactive.power_kvarh" : 2.95,
    "leading_current_reactive_power_kvarh" : 0,
    "co2(tco2)" : 0,
    "lagging_current_power_factor" : 73.21, 
    "leading_current_power_factor" : 100,
    "nsm" : 900,
    "week_status" : "sunday"
}

response  = requests.post(url,json=energy_consumption).json()

print(response['load_type'])