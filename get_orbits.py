import urllib.request
import json
import re

url = "https://raw.githubusercontent.com/rickyreusser/periodic-planar-three-body-orbits/master/data/suvakov.json"
try:
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    for orbit in data[:5]:
        print(orbit['name'], orbit['v1'], orbit['v2'])
except Exception as e:
    print("Failed", e)
