from pymongo import MongoClient
import math
import uuid
import threading
import time
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sympy import symbols, solve, sqrt
import numpy as np
from scipy.optimize import fsolve
from datetime import datetime, timezone

# Replace these with your connection details
mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
db_name = "EchoNet"

# Connect to the MongoDB client
client = MongoClient(mongo_uri)

# Select the database
db = client[db_name]

# Select the collection
collection = db["microphones-new"]
triangulatedEvents = db["triangulatedEvents2"]
# Retrieve all documents from the collection
documents = collection.find()

#We'll set the maximum neighbor range to 1 kilometer
#In good conditions, a kookaburra can be heard from up to 1km away, and is considered amongst the loudest birds in Australia
neighborRange = 1000

Microphones = {}
class Microphone:

  def __init__(self, name, lla):
        self.name = name
        self.lla = lla

  def setNeighbors(self, neighbors):
    self.neighbors = neighbors



#This function will calculate the distance between 2 mics using the Haversine formula
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

document_list = [doc for doc in documents]

# Print the documents
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
          #print("Neighbors Found")
          #If so, append the neighbor to an array, stashing both the sensorID and the distance between the two microphones
          neighbors[neighbor['sensorId']] = micDistance

    Microphones[microphone.name].neighbors = neighbors

def cluster():
  # Choose the events collection
  time.sleep(2)
  collectionEvents = db['dummyEvents5']

  documents = collectionEvents.find({"ClusterID": None})

  events = [doc for doc in documents]

  #Species dictionary that will contain each species and the distance of how far away they can be heard in meters.
  species = {}

  #Default range if no species within the dictionary has been found
  range = 100

  #Timing error correction (in seconds)
  timingErrorCorrection = 0.1

  for event in events:
    #print(event)
    if(event['ClusterID'] == None):
      #print("No ClusterID found for: {}".format(event['_id']))
      #A list of the clustered events
      clusteredEvents = []
      #We can add the first event of a cluster to the clustered events if it doesn't already belong to another cluster

      #clusteredEvents.append((event['SensorId'],event['Timevalue']))

      #generate a random UUID for the Cluster ID, convert it to a string, and set this events ClusterID to it
      random_id = uuid.uuid4()
      random_id_str = str(random_id)
      event['ClusterID'] = random_id_str
      collectionEvents.update_one({"_id": event['_id']}, {"$set": {"ClusterID": random_id_str}})
      print(f"Your random ID is: {random_id_str}")

      eventDictionary = {
                  "sensorId": event['SensorId'],
                  "microphoneLLA": Microphones[event['SensorId']].lla,
                  "ClusterId": random_id_str,
                  "timestamp": event['Timevalue'],
                  "Species": event['Species']
              }



      clusteredEvents.append(eventDictionary)
      #Check the detected species
      #Set the time and mic filter based on the how far the species call can be heard away from
      #TO DO - RESEARCH AND CREATE A STORE BASED ON HOW FAR AWAY EACH SPECIES CAN BE HEARD FROM
      #FOR NOW - Just default it to 100m or whatever the 'range' value is set to

      try:
        range = species[event['Species']['range']]
      except:
        print('Missing Species Information for : {}. \nMicrophone neighbor range will now default to {}'.format(event['Species'], range))


      #The timeband filter to apply to the rest of the events to narrow the search for events that could be considered clustered
      timeband = (range/343) *1000
      vocalizationTime = event['Timevalue']

      for event2 in events:

        #Only proceed if not collecting comparing the same event
        if event2!= event:
          if (event2['ClusterID'] != None):
            continue
          if ( event2['Timevalue'] > vocalizationTime - timeband and event2['Timevalue'] < vocalizationTime + timeband):
            #print("Potential Event cluster found.")
            #print("Event: {}\nMic:  {}\nNeighbor: {}".format(vocalizationTime, event['SensorId'], event2['SensorId']))
            #print(Microphones[event['SensorId']].neighbors[event2['SensorId']])

            #Get the distance seperating these two neighbors
            distanceNeighbor = Microphones[event['SensorId']].neighbors[event2['SensorId']]

            #Get the difference in time between the two events
            timeDelta = abs(event2['Timevalue'] - event['Timevalue'])


            if (timeDelta < (distanceNeighbor/343 + timingErrorCorrection)*1000):
              #print("---EVENT CLUSTER FOUND---")
              #print(" INIT MIC: {}\n Detecting Mic:  {}\n Distance Between: {} \n Time Delta: {}".format(event['SensorId'],event2['SensorId'], distanceNeighbor, timeDelta))
              event2['ClusterID'] = random_id_str
              collectionEvents.update_one({"_id": event2['_id']}, {"$set": {"ClusterID": random_id_str}})
              #print()
              #clusteredEvents.append((event2['SensorId'],event2['Timevalue']))
              eventDictionary = {
                  "sensorId": event2['SensorId'],
                  "microphoneLLA": Microphones[event2['SensorId']].lla,
                  "ClusterId": random_id_str,
                  "timestamp": event2['Timevalue'],
                  "Species": event2['Species']
              }
              clusteredEvents.append(eventDictionary)


      print("Cluster was found: {}".format(clusteredEvents))
      triangulate(clusteredEvents)
      #TO-DO SEND CLUSTERED EVENTS TO THE TRIANGULATION ALGORITHM


def triangulate(points):
  print("Triangulating")

  if len(points) == 1:
    print("Single microphone detection event...")
    return
    ##########################################################################
  #
  #
  #1.               Convert LLA to Cartesian Coorinates
  #
  #

  #Function to determine the distance between two points
  def distance(p1, p2):
      return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

  def lla_to_cartesian(origin, point):
      # Calculate distance and bearing
      distance = geodesic(origin, point).meters
      bearing = math.atan2(
          math.sin(math.radians(point[1] - origin[1])) * math.cos(math.radians(point[0])),
          math.cos(math.radians(origin[0])) * math.sin(math.radians(point[0])) -
          math.sin(math.radians(origin[0])) * math.cos(math.radians(point[0])) * math.cos(math.radians(point[1] - origin[1]))
      )

      # Convert distance and bearing to Cartesian coordinates
      x = distance * math.sin(bearing)
      y = distance * math.cos(bearing)

      return x, y

  # Define the LLA coordinates with timestamps
  """points = [
      {'_id': '655b17107bbd7bdf973f5281', 'sensorId': '63bf196e-262e-4af5-88ec-c492a57db849', 'microphoneLLA': [-38.79073162749655, 143.55529204420282, 10.0], 'timestamp': 24235243},
      {'_id': '655b17107bbd7bdf973f5282', 'sensorId': '2610e4f4-0f2f-487f-85e3-0f27d43449fd', 'microphoneLLA': [-38.79143446066542, 143.55458711596125, 10.0], 'timestamp': 24235136},
      {'_id': '655b17107bbd7bdf973f5283', 'sensorId': '6055589b-5d64-4cd6-9e6c-73d3b36e2ad1', 'microphoneLLA': [-38.79154621798846, 143.55554347038037, 10.0], 'timestamp': 24235203}
  ]"""

  print(points)
  #Array to store the approximated location of the source.
  #A 2 microphone detection event will store multiple points
  #For 3 Microphone detection events, it will store a single point
  #In the future once timing compensation is added into the triangulation algorithm, 3 microphone detection events will store multiple points
  source = []
  sourceLLA = []

  # Use the first point as the origin
  origin = points[0]['microphoneLLA'][:2]

  # Convert all points and include timestamp
  cartesian_points_with_timestamp = []

  circles = []

  for point in points:
      x, y = lla_to_cartesian(origin, point['microphoneLLA'][:2])
      timestamp = point['timestamp']
      sensorId = point['sensorId']

      dictionary = {
          'sensorId': sensorId,
          'hk': [x,y],
          'timestamp': timestamp,
          'radius': None
      }

      cartesian_points_with_timestamp.append((x, y, timestamp))
      circles.append(dictionary)

  ##########################################################################
  #
  #
  #2.        Calculate the time differences to determine the Radii
  #
  #

  # Find the dictionary with the smallest timestamp
  record_with_smallest_timestamp = min(circles, key=lambda x: x['timestamp'])

  # Extract just the timestamp
  smallest_timestamp = record_with_smallest_timestamp['timestamp']

  print(smallest_timestamp)

  #Calculate the radius for the circles that will be drawn
  for record in circles:
    record['radius'] = (record['timestamp'] - smallest_timestamp) * 0.343
    print(record['radius'])

  print(circles)

  ##########################################################################
  #
  #
  #3.                   Get Values for H1/H2 Etc
  #
  #

  #make new list for dictionaries of the circles drawn around the microphones
  circles2 = []
  init = True
  #create the dictionaries and append them to the newly created circles 2
  for record in circles:

    #If the clustered events contain 3 microphones
    if(len(circles)) >= 3:
      if(record['radius'] != 0):
        if init:
          h1 = record['hk'][0]
          k1 = record['hk'][1]
          r1 = record['radius']
          init = False
        else:
          h2 = record['hk'][0]
          k2 = record['hk'][1]
          r2 = record['radius']

    #if the clustered events contain 2 microphones
    if len(circles) == 2:
      if record['radius'] == 0:
        h1 = record['hk'][0]
        k1 = record['hk'][1]
        r1 = record['radius']
      else:
        h2 = record['hk'][0]
        k2 = record['hk'][1]
        r2 = record['radius']



  #For constructing the conic equation, the process is simplified by making variables 'a' 'b' 'c' 'R'
  #As these variables will be used to find the co-efficient for x^2, y^2, xy, etc. part of the equation
  a = 2*(h1) - 2*(h2)
  b = 2*(k1) - 2*(k2)
  c = h2**2 + k2**2 - h1**2 - k1**2 - (r1-r2)**2
  R = (2*(r1-r2))**2

  #-----Co-efficients-----
  x_squared_co = a**2 - R
  y_squared_co = b**2 - R
  xy = 2*a*b
  x_co = (2*a*c) + (2*R*h1)
  y_co = (2*b*c) + (2*R*k1)
  N = c**2 - R*(h1)**2 - R*(k1)**2

  print("x squared: ", x_squared_co )
  print("y squared: ", y_squared_co)
  print("xy: ", xy)
  print("x: ", x_co)
  print("y: ", y_co)
  print("N: ", N)

  #Define the variables x and y
  x, y = symbols('x y')

  #Define the quadratic equation
  equation = x_squared_co * x**2 + y_squared_co * y**2 + xy * x * y + x_co * x + y_co * y + N

  if len(points) >= 3:

    guesswidth = 100
    first = record_with_smallest_timestamp['hk']
    print(first)
    start = first[0] - guesswidth
    end = first [0] + guesswidth
    num_elements = 5

    numbers = np.linspace(start, end, num_elements)
    print(numbers)

    #The 3 items of this array will contain the distance measure, x co-ordinate, y co-ordinate
    smallest = [0,0,0]

    #For 3 Microphone detections, it places 5 lines on the graph and hones in on the unknown sauce location
    def find(numbers, axis):
      init = True
      guess = None
      for i in numbers:


        #If the hyperbola is north-south, use x as the guessing axis
        if axis == 1:

          equation_at_x_5 = equation.subs(x, i)

          solution_y = solve(equation_at_x_5, y)

        # Print the solutions
        #print(f"Solutions for y when x = {i} : {solution_y}")

          for ycord in solution_y:

            guess = (i,ycord)
            distanceMeasure1 = abs((distance((h1,k1),guess) - r1) - (distance(first,guess)))
            distanceMeasure2 = abs((distance((h2,k2),guess) - r2) - (distance(first,guess)))

            #We will recieve 2 solutions for ycord, we are only interested in one as we're only interested in the coordinates produced by one locus line.
            if abs(distanceMeasure1 - distanceMeasure2) > 1:
              print("Breaking...")
              continue

            if init:
              print("Init Found...")
              smallest[0] = distanceMeasure2
              smallest[1] = i
              smallest[2] = ycord
              init = False

            if distanceMeasure2 < smallest[0]:
              print("New smallest distance found...")
              smallest[0] = distanceMeasure2
              smallest[1] = i
              smallest[2] = ycord
              print("i: ",i)

        #If the hyperbola is east-west, use y as the guessing axis
        elif axis == 2:
          equation_at_y = equation.subs(y, i)

          solution = solve(equation_at_y, x)

          # Print the solutions
          for coord in solution:
            guess = (coord,i)
            distanceMeasure1 = abs((distance((h1,k1),guess) - r1) - (distance(first,guess)))
            distanceMeasure2 = abs((distance((h2,k2),guess) - r2) - (distance(first,guess)))

            #We will recieve 2 solutions for ycord, we are only interested in one as we're only interested in the coordinates produced by one locus line.
            if abs(distanceMeasure1 - distanceMeasure2) > 1:
              print("Breaking...")
              continue

            if init:
              print("Init found...")
              smallest[0] = distanceMeasure2
              smallest[1] = coord
              smallest[2] = i
              init = False

            if distanceMeasure2 < smallest[0]:
              print("New Smallest Found...")
              smallest[0] = distanceMeasure2
              smallest[1] = coord
              smallest[2] = i
              print("i: ",i)

      print("The smallest distance found was: ",smallest)
      print(guess)
      print((distance(record_with_smallest_timestamp['hk'],guess)))
      return smallest

    # Function to determine the angle between 3 points
    # An artificial point will be added on the same y axis as one of the microphones
    # This is so that we can determine the angle between 2 microphones with this point
    # Doing so will determine whether the hyperbola is East-West or North-South
    # It's be important to know the nature of the hyperbola as this will decide if the guessing algorithm uses x or y as it's guessing axis...
    # ...otherwise there is a risk the guessing algorithm can place a line where it doesn't intercept with the hyperbola, which can ruin the triangulation result
    def angle_between_points(p1, p2, p3):

        a = distance(p1, p2)
        b = distance(p2, p3)
        c = distance(p1, p3)

        # Apply the law of cosines
        angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))

        # Convert the angle to degrees
        return math.degrees(angle)

    #Add the artificial point as point 1
    #Get the angle that point 1 and point 3 make with point 2
    angle = angle_between_points((h1+10,k1),(h1,k1), (h2,k2))

    #If the angle is between 45 and 135 use the set the axis to X (guessing lines will be vertical), otherwise set them to horizontal Y
    if (angle >= 45 and angle <= 135):
      print("The angle is {:.2f} degrees. Use X =...".format(angle))
      axis = 1
    else:
      print("The angle is {:.2f} degrees. Use Y =...".format(angle))
      axis = 2

    find(numbers, axis)

    #Continue the guessing for 5 iterations, with each iteration the range of the guesses will become smaller and the source will be 'honed'
    for i in range(10):
      start_value = float(smallest[axis] - (guesswidth / (4 * (i+1))))
      end_value = float(smallest[axis] + (guesswidth / (4 * (i+1))))
      numbers = np.linspace(start_value, end_value, num_elements)
      #print('Numbers: ', numbers)
      find(numbers, axis)

    source.append([smallest[1],smallest[2]])
    print("Location of source: {}, {}. Within {} meters.".format(smallest[1], smallest[2], smallest[0]))

  elif len(points) == 2:
      #For 2 Mic detection clusters

    # Define the symbols
    x, y = symbols('x y')
    detectionRange = 100

    # Find the intersecting points of the conic equation and the detection range of the second microphone to detect the sound
    def find_intersections(conic_params, circle_params):
        x_squared_co, y_squared_co, xy, x_co, y_co, N, h2, k2, r2 = conic_params + circle_params

        # Define the function for fsolve
        def equations(vars):
            x_val, y_val = vars
            eq1 = x_squared_co * x_val**2 + y_squared_co * y_val**2 + xy * x_val * y_val + x_co * x_val + y_co * y_val + N
            eq2 = (x_val - h2)**2 + (y_val - k2)**2 - r2**2
            return [eq1, eq2]

        # Initial guesses
        initial_guesses = [[50, 50], [-50, 50], [50, -50], [-50, -50],
                          [100, 0], [-100, 0], [0, 100], [0, -100]]

        # Store unique solutions
        solutions = set()
        tolerance = 1e-4

        #Solve
        for guess in initial_guesses:
            sol = tuple(fsolve(equations, guess))
            if not any(np.linalg.norm(np.array(sol) - np.array(existing_sol)) < tolerance for existing_sol in solutions):
                solutions.add(sol)

        return list(solutions)

    #Example usage
    conic_params = [x_squared_co,y_squared_co, xy, x_co,  y_co, N]
    circle_params = [h2, k2, detectionRange]  #h2 = 0, k2 = 0, r2 = 100

    intersections = find_intersections(conic_params, circle_params)
    print(intersections)

    # Append the points to a points list, these points will be converted back into LLA co-ordinates in a later step
    for point in intersections:

      if abs((distance(point, (h1,k1))) - (distance(point, (h2,k2)) - r2)) < 0.0001:
        print(point)
        source.append(point)

    #A line is drawn between mic1 and mic2
    #This function finds the point where the line intercepts with the detection circle of microphone 2
    def point_along_line(point1, point2, distance):
        # Convert points to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)

        # Calculate direction vector and its magnitude
        direction = p2 - p1
        magnitude = np.linalg.norm(direction)

        # Normalize the direction vector
        unit_vector = direction / magnitude

        # Calculate the new point
        new_point = p1 + unit_vector * distance

        return new_point

    #This function calculates the midpoint between two points
    #In this case it'll be primarily used to find the mid point of the hyperbola that satisfies the conditions of a detection location
    def calculate_midpoint(p1, p2):
        print((np.array(p1) + np.array(p2)) / 2)
        midpoint = (np.array(p1) + np.array(p2)) / 2
        midpoint1 = (midpoint[0], midpoint[1])
        source.append(midpoint1)
        return (np.array(p1) + np.array(p2)) / 2

    # Example usage
    point1 = (h1, k1)
    point2 = (h2, k2)
    #distance = r2  # Distance to travel along the line
    t = distance((h1,k1),(h2,k2)) - r2

    #Get the point
    new_point = point_along_line(point1, point2, t)

    #Calculate the midpoint between the newly obtained point, and microphone 1's location
    calculate_midpoint((h1,k1),new_point)

  # Converting back to LLA

  def cartesian_to_lla(origin, cartesian_point):
      # Calculate the distance
      distance = math.sqrt(cartesian_point[0]**2 + cartesian_point[1]**2)

      # Calculate the bearing
      bearing = math.degrees(math.atan2(cartesian_point[0], cartesian_point[1]))

      # Find destination point
      destination = geodesic(meters=distance).destination(origin, bearing)

      altitude = 0
      for point in points:
        altitude += point['microphoneLLA'][2]
      altitude = altitude / len(points)

      return destination.latitude, destination.longitude, altitude

  # Example usage
  origin_lla = (-38.79073162749655, 143.55529204420282)  # Your origin in LLA


  for point in source:
    print(point)
    print(cartesian_to_lla(origin,point))
    sourceLLA.append(cartesian_to_lla(origin,point))

  for point in sourceLLA:
    print("Final Point(s):  ", point)

  current_utc_time = datetime.now(timezone.utc)
  formatted_timestamp = current_utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + current_utc_time.strftime('%z')
  formatted_timestamp = formatted_timestamp[:-2] + ":" + formatted_timestamp[-2:]

  vocalization = {
      "timestamp": current_utc_time,
      "sensorId": "SensorId",
      "species": "Felis catus",
      "microphoneLLA": [20,10,10] ,
      "animalEstLLA": [sourceLLA[0]],
      "animalTrueLLA": sourceLLA,
      "animalLLAUncertainty":50,
      "audioClip":"AAAAOAAAQLgAAAA4AACAtwAAAAAAAAA4AACAtwAAAAAAAAAAAACAtwAAADgAAAC4AAAAOA…",
      "confidence": 100,
      "sampleRate": 16000
    }

  inserted_id = triangulatedEvents.insert_one(vocalization).inserted_id
  print(f"Record inserted with ID: {inserted_id}")

  # Main execution
def main():
  # MongoDB URI and Database Name
  mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
  db_name = "EchoNet"

  # Initialize MongoDB Client
  client = MongoClient(mongo_uri)
  db = client[db_name]
  collection = db['dummyEvents5']

  # Pipeline for change stream
  pipeline = [{'$match': {'operationType': 'insert'}}]

  # Listening to the change stream
  try:
      with collection.watch(pipeline) as stream:
          for change in stream:
              cluster()
  except Exception as e:
      print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()