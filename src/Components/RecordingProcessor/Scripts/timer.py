from pymongo import MongoClient
import math

mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
db_name = "EchoNet"
# Connect to the MongoDB client
client = MongoClient(mongo_uri)

# Select the database
db = client[db_name]


#######################################
#        TEST EVENTS COLLECTION       #
#######################################

eventsCollection = db["test-events"]

# record = {
#     "Timevalue": timestamp,
#     "SensorId": document_list[i]['sensorId'],
#     "Species": None,
#     "ClusterID": None,
#     "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# }

test_event = {
    'ClusterID':'1_453463463463', #Folder name of cluster to also be named as the CLUSTER ID, Cluster ID can be potentially be time in ms?
    'Species':'Determined by Engine', #Send one clip off from the cluster into the engine then have it populate this cell
    'Events': [              # The file names of the events that are considered to be within the cluster, (micID, followed by the timestamp)
        '1_23423582358235',
        '2_24248682469924',
       '3_34858345994528'
    ],
    
    # The calculated LLA of the animal as determined by the triangulation algo
    # if there are less than 3 events within the cluster, there will be a range of lla predictions
    'location':[             
        'lat',
        'lon',
        'alt'
    ],
    'confidence':'confidence', #confidence as determined by the engine
    'audioClip': "",           #The audio clip converted into text
    'sampleRate': 16000,       #The sample rate
}

#Insert into db and print
inserted_id = eventsCollection.insert_one(test_event).inserted_id


#######################################
#            MICROPHONES              #
#######################################
# collection = db["microphones-deakin"]
# documents = collection.find()

# print(documents)

# #We'll set the maximum neighbor range to 1 kilometer
# #In good conditions, a kookaburra can be heard from up to 1km away, and is considered amongst the loudest birds in Australia
# neighborRange = 1000

# Microphones = {}
# class Microphone:

#   def __init__(self, name, lla):
#         self.name = name
#         self.lla = lla

#   def setNeighbors(self, neighbors):
#     self.neighbors = neighbors

# document_list = [doc for doc in documents]

# def distance(coord1, coord2):
#     # Radius of the Earth in kilometers
#     R = 6371.0

#     # Coordinates and altitude in decimal degrees (latitude, longitude, altitude)
#     lat1, lon1, alt1 = coord1
#     lat2, lon2, alt2 = coord2

#     # Convert latitude and longitude from degrees to radians
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     delta_phi = math.radians(lat2 - lat1)
#     delta_lambda = math.radians(lon2 - lon1)

#     # Haversine formula for two-dimensional distance
#     a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     two_d_distance = R * c

#     # Convert altitude from meters to kilometers and calculate the three-dimensional distance
#     alt_diff = abs(alt1 - alt2) / 1000
#     total_distance = math.sqrt(two_d_distance**2 + alt_diff**2)

#     # Convert total distance from kilometers to meters
#     total_distance_meters = total_distance * 1000

#     return total_distance_meters

# for doc in document_list:
#     #print(doc)
#     #print("ok")
#     microphone = Microphone(doc['sensorId'],doc['microphoneLLA'])
#     Microphones[microphone.name] = microphone
#     neighbors = {}

#     #Create a com
#     for neighbor in document_list:
#         if neighbor != doc:

            
#             #Check to see if they can be considered as 'neighboring'
#             micDistance = distance(doc['microphoneLLA'],neighbor['microphoneLLA'])
#             if  micDistance < neighborRange:
                
#                 print(f"Neighbors Found for {doc['sensorId']}:  {neighbor['sensorId']}: {micDistance}")
#                 #If so, append the neighbor to an array, stashing both the sensorID and the distance between the two microphones
#                 neighbors[neighbor['sensorId']] = micDistance

#     Microphones[microphone.name].neighbors = neighbors

# print(Microphones)

# for m in Microphones:
#     print(m)