import os
import taglib
import requests
import time as tm
from pymongo import MongoClient
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import math
import subprocess
import base64
import asyncio
import aiohttp
from datetime import datetime, timezone


#URL responsible for passing event info to engine
engine_url = "http://ts-api-cont:9000/clustering/post_recording"
triangulation_url = "http://ts-api-cont:9000/triangulation/post_points"

#url = self.config['MODEL_SERVER']

mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
db_name = "EchoNet"
# Connect to the MongoDB client
client = MongoClient(mongo_uri)

# Select the database
db = client[db_name]

#Microphone dictionary and class
#Microphones class objects will make note of their distances to neighbors within a determined range
Microphones = {}
class Microphone:

    def __init__(self, name, lla):
            self.name = name
            self.lla = lla

    def setNeighbors(self, neighbors):
        self.neighbors = neighbors

#Function for calculating the distance between two co-ordinates, will be used for neighboring microphone distances
def distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Coordinates and altitude in decimal degrees (latitude, longitude, altitude)
    lat1, lon1, alt1 = coord1
    lat2, lon2, alt2 = coord2

    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula for two-dimensional distance
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    two_d_distance = R * c

    # Convert altitude from meters to kilometers and calculate the three-dimensional distance
    alt_diff = abs(alt1 - alt2) / 1000
    total_distance = math.sqrt(two_d_distance**2 + alt_diff**2)

    # Convert total distance from kilometers to meters
    total_distance_meters = total_distance * 1000

    return total_distance_meters

def fetchMicrophones():
    microphone_collection = db["microphones-deakin"]
    microphone_documents = microphone_collection.find()
    print(microphone_documents)

    #We'll set the maximum neighbor range to 1 kilometer
    #In good conditions, a kookaburra can be heard from up to 1km away, and is considered amongst the loudest birds in Australia
    neighborRange = 1000

    document_list = [doc for doc in microphone_documents]

    for doc in document_list:
        #print(doc)
        #print("ok")
        microphone = Microphone(doc['sensorId'],doc['microphoneLLA'])
        Microphones[microphone.name] = microphone
        neighbors = {}

        #Create a com
        for neighbor in document_list:
            if neighbor != doc:

                #Check to see if they can be considered as 'neighboring'
                micDistance = distance(doc['microphoneLLA'],neighbor['microphoneLLA'])

                if  micDistance < neighborRange:
                    
                    print(f"Neighbors Found for {doc['sensorId']}:  {neighbor['sensorId']}: {micDistance}")
                    #If so, append the neighbor to an array, stashing both the sensorID and the distance between the two microphones
                    neighbors[neighbor['sensorId']] = micDistance

        Microphones[microphone.name].neighbors = neighbors

    print(Microphones)

    for m in Microphones:
        print(m)

#Getting the events
events = db["test-events"]
startTime = 0

# Specify the directory path
directory_path = '/mnt/recordings/onset/audio'
watchDirectory = '/mnt/recordings/onset/audio'
export_directory = '/mnt/recordings/clusters/' 

# Specify the post URL for sending audio to engine.

# engineAPI = "http://localhost:9000/engine/upload_wav"

# vocalisation_event = {
#     "timestamp": data.timestamp.isoformat(),
#     "sensorId": data.sensorId,
#     "microphoneLLA": data.microphoneLLA,
#     "animalEstLLA": data.animalEstLLA,
#     "animalTrueLLA": data.animalTrueLLA,
#     "animalLLAUncertainty": 50.0,
#     "audioClip" : data.audioClip, 
#     "audioFile" : data.audioFile      
# } 

# response = requests.post(engineAPI, files=files, data = data)

async def post_request(url, json_data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_data) as response:
            return await response.text()

def clustering(times):
    clusteredEvents = []
    range = 500 #Max range in meters that a vocalization can travel and be detected on a mic
    timeband = (range/343) *1000 

#   #Timing error correction (in seconds)
    timingErrorCorrection = .01

    for event in times:
        event_mic = event[0]
        event_time = event[1]
        cluster = []
        

        if((int(tm.time())*1000) - event_time  >= 60000):
            print("Event passed age check.")
            cluster.append(event)

            for event2 in times:
                event_2_mic = event2[0]
                event_2_time = event2[1]

                #Go to next event if event 2 belongs to same microphone
                if (event_mic == event_2_mic):
                    continue

                if (event_2_time < event_time + timeband and event_2_time > event_time - timeband):
                    print("Potential event found")
                    
                    distanceNeighbor = Microphones[f"{event_mic}"].neighbors[f"{event_2_mic}"]

                    #Get the difference in time between the two events
                    timeDelta = abs(event_2_time - event_time)

                    if (timeDelta < (distanceNeighbor/343 + timingErrorCorrection)*1000):
                        print("Potential event found")
                        cluster.append(event2)
                        times.remove(event2)

            times.remove(event)

            clusteredEvents.append(cluster)
            print(clusteredEvents)

        else:
            return clusteredEvents
        
        return clusteredEvents

    return

#Sorts the files in the onset volume by vocalization time (ascending order)
def sortSecond(val):
    return val[1]

#Briefly compares audio clips within timeframes of interest to better gauge if they are coming from the same species
def euclideanCompare(clusteredEvents):
    print()

#Responsible for converting .wav to string
def file_to_base64(filepath):
    # Open the file in binary mode
    with open(filepath, 'rb') as file:
        # Read the file content
        file_content = file.read()
        # Encode the file content to base64
        base64_string = base64.b64encode(file_content).decode('utf-8')
    return base64_string

def count_files(directory):
    if not os.path.isdir(directory):
        raise Exception(f"Directory {directory} does not exist")
    
    file_names = []
    for root, dirs, files in os.walk(directory):
        file_names.append(files)

    return file_names

async def task():
    
    print("Counting Files...")
    files = count_files(directory_path)[0]

    #Will act as the sorted times
    times = []

    #Clusters array which will temporarily contain grouped events
    #Will be responsible for writing the sound events to new folder and also the db

    counter = len(files)
    if counter == 0:
        return

    #Append recording infomation and sort them based on their time.
    for file in files:

        parts = file.split('_')
        mic = parts[0]
        startTime = int(parts[1].split('.')[0])
        time_list = (mic, startTime, file)
        times.append(time_list)        
        times.sort(key=sortSecond)
        #print(times)
    
    #Perform the clustering algo on the times list
    #
    #   1. Ensure time within each event is past a certain age threshold
    #
    #       a.  (Edge case) If last event is just past certain age threshold
    #           ensure that the algorithm allows for events little behind the threshold
    #           as a *final bording call, that may belong to events that satisfy the age requirement
    #
    #   2. If an event passes the age check, check sequential events 

    clusteredEvents = clustering(times)

     #######################################################################
    #                         TO DO EUCLIDEAN COMPARE                       #
    #  Before creating the euclidean compare algorithm                      #
    #  need to work on the onset detection for multiple onsets in a clip    #
     #######################################################################

    #Process events again so check to get euclidean vector values of clips
    #If similar enough, we can then say they are part of the same cluster

    #clusteredEvents = euclideanCompare(clusteredEvents)
    
    eventsdb = db['test-events']

    if clusteredEvents != None:
        for cluster in clusteredEvents:
            
            names = []
            #Create new db entry
            folder_name = cluster[0][1]
            audio_to_send = cluster[0][2]

            for event in cluster:
                #eg. rename as the following /mnt/recordings/onset/audio/3/12135136136/3_131361371.wav
                print("Inserting file into new folder:  ", event)
                name = event[2].split('.')
                name = name[0]
                names.append(name)
                new_file_name = f"{len(cluster)}/{folder_name}/{event[2]}"
                new_file_destination = f"{len(cluster)}/{folder_name}/"

                new_file_path = os.path.join(export_directory, new_file_name)
                file_path = f"/mnt/recordings/onset/audio/{event[2]}"
                destination_dir = os.path.join(export_directory, new_file_destination)

                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)

                os.rename(file_path, new_file_path)
                print(f'Renamed "{file_path}" to "{new_file_name}"')



            clusteredEvent = {
                'timestamp': datetime.now(timezone.utc),
                'sensorId': None,   #PLACEHOLDER
                'clusterID':str(folder_name), #Folder name of cluster to also be named as the CLUSTER ID, Cluster ID can be potentially be time in ms?
                'species': None,
                'microphoneLLA':   [],#Placeholder  #Send one clip off from the cluster into the engine then have it populate this cell
                'events': names,           # The file names of the events that are considered to be within the cluster, (micID, followed by the timestamp)
                # The calculated LLA of the animal, if there are less than 3 events within the cluster, there will be a range of lla predictions
                # As determined by the triangulation algo
                "animalEstLLA": None,
                "animalTrueLLA": None,
                "animalLLAUncertainty": 50, #PLACEHOLDER
                'confidence': None, #confidence as determined by the engine
                'audioClip': None,
                'sampleRate': None,
            }

            try:
                x = eventsdb.insert_one(clusteredEvent)
                print("Succesfully inserted cluster into db")
                print("Sending file to engine for classification/ triangulation.")

            except Exception:
                print("Couldn't insert record into db:  ", Exception)

            try:
                #Send Event to Engine API
                file_name = f"{len(cluster)}/{folder_name}/{audio_to_send}"
                filepath = f"/mnt/recordings/clusters/{file_name}"

                clusterID = str(folder_name)
                audioClip = file_to_base64(filepath)
                audioFile = "Recording_Mode_V2"

                vocalizationEvent = {

                    "type": "clustering",
                    "clusterID": clusterID,
                    "audioClip": audioClip,
                    "audioFile": audioFile
                }

                vocalization_times = {
                    "clusterID": clusterID,
                    "times": names
                }
                
                    # Create tasks to run requests concurrently
                task1 = asyncio.create_task(post_request(engine_url, vocalizationEvent))
                task2 = asyncio.create_task(post_request(triangulation_url, vocalization_times))

                # Wait for both tasks to complete
                response1, response2 = await asyncio.gather(task1, task2)

                print("Response from engine_url:", response1)
                print("Response from triangulation_url:", response2)
                # x = requests.post(engine_url, json = vocalizationEvent)
                # y = requests.post(triangulation_url, json = vocalization_times)

                # print("Returned:    ",x)

            except Exception as e:
                print("Error sending file to engine:  ", e)

            #Send data to triangulation
            #Send audio to engine

class OnMyWatch:
 
    def __init__(self):
        self.observer = Observer()
 
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, watchDirectory, recursive = True)
        self.observer.start()
        try:
            while True:
                
                tm.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")
 
        self.observer.join()
 
class Handler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.created_files = set()

    def on_any_event(self, event):
        if event.is_directory:
            return None

        if event.event_type == 'created':

            if event.src_path not in self.created_files:
                self.created_files.add(event.src_path)
                print("Watchdog received created event - %s." % event.src_path)
                asyncio.run((task()))
                #clustering()            

        #print("Printing self created files: ",self.created_files)

if __name__ == '__main__':
    fetchMicrophones()
    startTime = tm.time()
    print(f"Watchdog starting... Observing  {watchDirectory}")
    watch = OnMyWatch()
    watch.run()