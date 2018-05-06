from math import cos, sqrt
import sys
import json

audiosegs = json.loads(open("/home/marcello/Music/AI_Tomomi/python/tSNE_tests/json-data/example-audio-tSNE-onsets.json").read())
newpoint = [0.22157612715148592, 0.028174570174771243]

def distance(lon1, lat1, lon2, lat2): 
      x = (lon2-lon1) * cos(0.5*(lat2+lat1)) 
      y = (lat2-lat1) 
      return x*x + y*y
pointlist = sorted(audiosegs, key= lambda d: distance(d["point"][0], d["point"][1], newpoint[0], newpoint[1] ))
print("newpoint: %s" % newpoint )
print("closest points: %s" % pointlist[:5])
