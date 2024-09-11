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
import paho.mqtt.client as paho
import os
import json
import requests
import sympy as sp


# Replace these with your connection details
# Replace these with your connection details
mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
db_name = "EchoNet"

# Connect to the MongoDB client
client = MongoClient(mongo_uri)

# Select the database
db = client[db_name]

triangulatedEvents = db["triangulatedEvents2"]
# Retrieve all documents from the collection

collection = db["microphones-deakin"]
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
def setupMics(docs):
   
   for doc in docs:
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



class EchoTriangulation ():

    def __init__(self) -> None:  
            
            # Load the engine config JSON file into a dictionary
            try:
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo_triangulation.json')
                with open(file_path, 'r') as f:
                    self.config = json.load(f)
                
                print(f"Echo Triangulation configuration successfully loaded.", flush=True)
            except:
                print(f"Could not load triangulation config : {file_path}") 
            
            # Load the project echo credentials into a dictionary
            
            # Setup database client and connect
            try:
                # database connection string
                # self.connection_string=f"mongodb://{self.credentials['DB_USERNAME']}:{self.credentials['DB_PASSWORD']}@{self.config['DB_HOSTNAME']}/EchoNet"

                # myclient = pymongo.MongoClient(self.connection_string)
                # self.echo_store = myclient["EchoNet"]

                mongo_uri = "mongodb+srv://bndct:2zZwTx4E1Rd8dKsJ@cluster0.ntu9thj.mongodb.net"
                db_name = "EchoNet"

                # Initialize MongoDB Client
                client = MongoClient(mongo_uri)
                db = client[db_name]
                test_events = db['test-events']

                print(f"Connected to mongodb", flush=True)
            except:
                print(f"Failed to establish database connection", flush=True)

    def execute(self):

        print("Triangulation.")
        client = paho.Client()
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message
        
        # retry connection until this succeeds
        connected = False
        while not connected:
            try:
                client.connect(self.config['MQTT_CLIENT_URL'], self.config['MQTT_CLIENT_PORT'])
                connected=True
            except:
                time.sleep(1)    
        
        print(f'Subscribing to MQTT: {self.config["MQTT_CLIENT_URL"]} {self.config["MQTT_PUBLISH_URL"]}')
        client.subscribe(self.config['MQTT_PUBLISH_URL'])
        client.subscribe("Triangulate")

        print("Waiting for event details to arrive...")
        client.loop_forever()
    
    def line_through_points(self,x1, y1, x2, y2):
        # Calculate the slope of the line
        if x1 == x2:
            raise ValueError("The points are vertical, slope is undefined.")
        m = (y2 - y1) / (x2 - x1)
        
        # Equation of the line in point-slope form: y - y1 = m * (x - x1)
        # Convert to slope-intercept form: y = mx + b
        b = y1 - m * x1

        return m, b

    def perpendicular_slope(self,m):
        # Calculate the slope of the perpendicular line
        if m == 0:
            raise ValueError("The line is horizontal, perpendicular slope is undefined.")
        return -1 / m
        
    def evenly_spaced_points(self,x1, y1, x2, y2, num_points):
        # Generate num_points evenly spaced points between (x1, y1) and (x2, y2)
        x_points = np.linspace(x1, x2, num_points)
        y_points = np.linspace(y1, y2, num_points)
        
        return list(zip(x_points, y_points))

    def perpendicular_lines_through_points(self,points, m_perp):
        # Create equations for lines perpendicular to the original line passing through each point
        equations = []
        for (x1, y1) in points:
            # Equation in point-slope form: y - y1 = m_perp * (x - x1)
            # Convert to slope-intercept form: y = m_perp * x + b
            b = y1 - m_perp * x1
            equations.append((m_perp, b))
        return equations

    def find_intersections_with_hyperbola(self,perpendicular_equations, hyperbola_params, x, y):
        x_squared_co, y_squared_co, xy, x_co, y_co, N = hyperbola_params

        # Define the symbols
        #x, y = sp.symbols('x y')

        # Define the hyperbola equation
        hyperbola_eq = x_squared_co * x**2 + y_squared_co * y**2 + xy * x * y + x_co * x + y_co * y + N

        intersection_points = []
        
        for m_perp, b_perp in perpendicular_equations:
            # Define the perpendicular line equation
            line_eq = m_perp * x + b_perp - y
            
            # Solve the system of equations
            solutions = sp.solve([hyperbola_eq, line_eq], (x, y))
            
            # Filter out complex solutions
            real_solutions = []
            for sol in solutions:
                if sol[0].is_real and sol[1].is_real:
                    real_solutions.append((float(sol[0]), float(sol[1])))
            
            intersection_points.extend(real_solutions)

        return intersection_points

    def hyperbola_eq(self, coeffs, x, y):
        a, b, c, d, e, f = coeffs
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    def tangent_slope_at_point(self,coeffs, x0, y0, x, y):
        # Define the hyperbola equation
        eq = self.hyperbola_eq(coeffs,x,y)
        
        # Implicit differentiation
        dydx = sp.simplify(-eq.diff(x) / eq.diff(y))
        
        # Evaluate the derivative at the given point
        slope = dydx.subs({x: x0, y: y0})
        return slope

    # Function to find the tangent line equation at a given point (x0, y0)
    def tangent_line_at_point(self,coeffs, coords, x, y):
        # Find the slope of the tangent line
        mb_values = []
        for x0, y0 in coords:
            m = float(self.tangent_slope_at_point(coeffs, x0, y0, x, y))
            
            # Use the point-slope form to find the tangent line equation
            b = y0 - m * x0
            mb_values.append((m,b))
        return mb_values

    def get_correct_locus_points(self,intersection_points, h1,k1,h2,k2,r2):
        correctPoints = []
        for coord in intersection_points:
            print(abs((self.distance(coord, (h1,k1))) - (self.distance(coord, (h2,k2)) - r2)))
            if abs((self.distance(coord, (h1,k1))) - (self.distance(coord, (h2,k2)) - r2)) < 0.0001:
                print("Potential point of source: ",coord)
                correctPoints.append(coord)
        return correctPoints

    def distance(self,p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def order_Polygon(self, polygonPoints):
        orderedPolygon = []

        for polygonPoint in polygonPoints:
            print("Appending:    ", polygonPoint[0])
            orderedPolygon.append(polygonPoint[0])
        
        for polygonPoint in reversed(polygonPoints):
            print("Appending:    ", polygonPoint[1])
            orderedPolygon.append(polygonPoint[1])
        
        orderedPolygon.append(polygonPoints[0][0])
        print("Appending:   ",polygonPoints[0][0])

        return orderedPolygon

    #This function is used to get the polygon points such that it obtains +- 1 unit distance along the function perpendicular to the tangent on the hyperbola at a given point
    def points_at_unit_distance(self, coordinates, mValues):
        # Calculate the changes in x and y based on the gradient
        polygonCoordinates = []
        count = 0
        for x0, y0 in coordinates:
            
            dx = 1 / np.sqrt(1 + mValues[count]**2)
            dy = mValues[count] * dx
            
            # Calculate the new points
            point1 = (x0 + dx, y0 + dy)
            point2 = (x0 - dx, y0 - dy)
            polygonCoordinates.append((point1,point2))
            count += 1

        return polygonCoordinates

    def cartesian_to_lla(self, origin, cartesian_point, points):
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

    def angle_between_points(self, p1, p2, p3):

            a = self.distance(p1, p2)
            b = self.distance(p2, p3)
            c = self.distance(p1, p3)

            # Apply the law of cosines
            angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))

            # Convert the angle to degrees
            return math.degrees(angle)
    # Define the LLA coordinates with timestamps

    def find(self, numbers, axis,h1,k1,r1,h2,k2,r2,x,y,equation,first,smallest):
        
        print(f"h1:{h1}   k1:{k1}   r1:{r1}")
        print(f"h2:{h2}   k2:{k2}   r2:{r2}")
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
                    distanceMeasure1 = abs((self.distance((h1,k1),guess) - r1) - (self.distance(first,guess)))
                    distanceMeasure2 = abs((self.distance((h2,k2),guess) - r2) - (self.distance(first,guess)))
                    print("Distance Measure 2:  ", distanceMeasure2)
                    #We will recieve 2 solutions for ycord, we are only interested in one as we're only interested in the coordinates produced by one locus line.
                    if abs(distanceMeasure1 - distanceMeasure2) > 1:
                        break

                    if self.init:
                        
                        smallest[0] = distanceMeasure2
                        smallest[1] = i
                        smallest[2] = ycord
                        self.init = False

                    if distanceMeasure2 < smallest[0]:
                        
                        smallest[0] = distanceMeasure2
                        smallest[1] = i
                        smallest[2] = ycord
                        

            #If the hyperbola is east-west, use y as the guessing axis
            elif axis == 2:
                equation_at_y = equation.subs(y, i)

                solution = solve(equation_at_y, x)

                # Print the solutions
                for coord in solution:
                    guess = (coord,i)

                    distanceMeasure1 = abs((self.distance((h1,k1),guess) - r1) - (self.distance(first,guess)))
                    distanceMeasure2 = abs((self.distance((h2,k2),guess) - r2) - (self.distance(first,guess)))
                    print("Here")
                    print("Distance Measure 2:  ", distanceMeasure2)

                    #We will recieve 2 solutions for ycord, we are only interested in one as we're only interested in the coordinates produced by one locus line.
                    if abs(distanceMeasure1 - distanceMeasure2) > 1:
                        break

                    if self.init:
                        smallest[0] = distanceMeasure2
                        smallest[1] = coord
                        smallest[2] = i
                        self.init = False

                    if distanceMeasure2 < smallest[0]:
                        smallest[0] = distanceMeasure2
                        smallest[1] = coord
                        smallest[2] = i
                        print("i: ",i)

        print("The smallest distance found was: ",smallest)
        print(guess)
        print((self.distance(self.record_with_smallest_timestamp['hk'],guess)))
        return smallest

    def lla_to_cartesian(self, origin, point):
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

    def triangulate(self,clusterID, times):
        
        print("Triangulating")

        points = []
        for time in times:
            parts = time.split('_')
            mic = parts[0]
            timing = int(parts[1])
            point = {
                'sensorId':mic,
                'microphoneLLA': Microphones[mic].lla ,
                'timestamp': timing 
            }
            points.append(point)
        # points = [
        #     {'_id': '655b17107bbd7bdf973f5281', 'sensorId': '63bf196e-262e-4af5-88ec-c492a57db849', 'microphoneLLA':  [-38.784207957502446, 143.57394194183942, 10], 'timestamp': 1711069670809.7725},
        #     {'_id': '655b17107bbd7bdf973f5282', 'sensorId': '2610e4f4-0f2f-487f-85e3-0f27d43449fd', 'microphoneLLA': [-38.784207957502446, 143.57509530505845, 10], 'timestamp': 1711069670855.416}
        #     #{'_id': '655b17107bbd7bdf973f5282', 'sensorId': '2610e4f4-0f2f-487f-85e3-0f27d43449fd', 'microphoneLLA': [-38.78342942350244, 143.57451862344894, 10], 'timestamp': 1711069670955.1536}
        # ]
        source = []
        sourceLLA = []

        if len(points) == 1:
            print("Single microphone detection event...")
            sourceLLA.append(Microphones[mic].lla)
            self.echo_api_send_locations_from_triangulation(clusterID,sourceLLA)
            #updateRecord(clusterID, points)
            return
        ##########################################################################
        #
        #
        #1.               Convert LLA to Cartesian Coorinates
        #
        #
        #Function to determine the distance between two points

        #Array to store the approximated location of the source.
        #A 2 microphone detection event will store multiple points
        #For 3 Microphone detection events, it will store a single point
        #In the future once timing compensation is added into the triangulation algorithm, 3 microphone detection events will store multiple points
        

        # Use the first point as the origin
        origin = points[0]['microphoneLLA'][:2]

        # Convert all points and include timestamp
        cartesian_points_with_timestamp = []

        circles = []

        for point in points:
            x, y = self.lla_to_cartesian(origin, point['microphoneLLA'][:2])
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
        self.record_with_smallest_timestamp = min(circles, key=lambda x: x['timestamp'])

        # Extract just the timestamp
        smallest_timestamp = self.record_with_smallest_timestamp['timestamp']

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
            if(len(circles)) == 3:
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

            guesswidth = 50
            first = self.record_with_smallest_timestamp['hk']
            print(first)
            start = first[0] - guesswidth
            end = first [0] + guesswidth
            num_elements = 5

            numbers = np.linspace(start, end, num_elements)
            print(numbers)

            #The 3 items of this array will contain the distance measure, x co-ordinate, y co-ordinate
            smallest = [float('inf'), None, None]

            #For 3 Microphone detections, it places 5 lines on the graph and hones in on the unknown sauce location
            

            # Function to determine the angle between 3 points
            # An artificial point will be added on the same y axis as one of the microphones
            # This is so that we can determine the angle between 2 microphones with this point
            # Doing so will determine whether the hyperbola is East-West or North-South
            # It's be important to know the nature of the hyperbola as this will decide if the guessing algorithm uses x or y as it's guessing axis...
            # ...otherwise there is a risk the guessing algorithm can place a line where it doesn't intercept with the hyperbola, which can ruin the triangulation result
            

            #Add the artificial point as point 1
            #Get the angle that point 1 and point 3 make with point 2
            angle = self.angle_between_points((h1+10,k1),(h1,k1), (h2,k2))

            #If the angle is between 45 and 135 use the set the axis to X (guessing lines will be vertical), otherwise set them to horizontal Y
            if (angle >= 45 and angle <= 135):
                print("The angle is {:.2f} degrees. Use X =...".format(angle))
                axis = 1
            else:
                print("The angle is {:.2f} degrees. Use Y =...".format(angle))
                axis = 2
            self.init = True
            smallest = self.find(numbers, axis,h1,k1,r1,h2,k2,r2,x,y,equation,first,smallest)
            print("Smallest:    ",smallest)
            #Continue the guessing for 5 iterations, with each iteration the range of the guesses will become smaller and the source will be 'honed'
            for i in range(5):
                start_value = float(smallest[axis] - (guesswidth / (4 * (i+1))))
                end_value = float(smallest[axis] + (guesswidth / (4 * (i+1))))
                numbers = np.linspace(start_value, end_value, num_elements)
                #print('Numbers: ', numbers)
                smallest = self.find(numbers, axis,h1,k1,r1,h2,k2,r2,x,y,equation,first,smallest)

            source.append([smallest[1],smallest[2]])
            print("Location of source: {}, {}. Within {} meters.".format(smallest[1], smallest[2], smallest[0]))

        ####################################################################################################################################################################################

        elif len(points) == 2:
            #For 2 Mic detection clusters

        # Define the symbols
            x, y = symbols('x y')
            detectionRange = 200

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
                initial_guesses = [[0, 100], [-50, 50], [50, -50], [-50, -50],
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
            print("The intersections of the hyperbola and the range of mic 2: ",intersections)

            # Append the points to a points list, these points will be converted back into LLA co-ordinates in a later step
            sourceRange = []
            rangeIntercept = self.get_correct_locus_points(intersections,h1,k1,h2,k2,r2)

            print("Source range:  ",sourceRange)
            print(rangeIntercept)
            x1 = rangeIntercept[0][0]
            y1 = rangeIntercept[0][1]
            x2 = rangeIntercept[1][0]
            y2 = rangeIntercept[1][1]
            m, b = self.line_through_points(x1,y1,x2,y2)
            m1 = self.perpendicular_slope(m)
            print(f"m: {m},   b: {b}  m1: {m1}")

            ten_points = self.evenly_spaced_points(x1, y1, x2, y2, num_points= 10)

            perpendicular_equations = self.perpendicular_lines_through_points(ten_points, m1)
            intersection_points = self.find_intersections_with_hyperbola(perpendicular_equations, conic_params, x, y)
            polygonSpine = self.get_correct_locus_points(intersection_points,h1,k1,h2,k2,r2)
            mb_values = self.tangent_line_at_point(conic_params, polygonSpine, x, y)
            m1_tangents = []

            for mb in mb_values:
                print(f"MB:   {mb}")
                m = mb[0]
                m1_tangents.append(self.perpendicular_slope(m))

            count = 0
            perpendicular_equations_for_points_on_hyperbola = []
            for vertibre in polygonSpine:
                vert = []
                vert.append(vertibre)
                perpendicular_equations_for_points_on_hyperbola.append(self.perpendicular_lines_through_points(vert,m1_tangents[count]))
                count+=1
            
            polygonPoints = self.points_at_unit_distance(polygonSpine, m1_tangents)
            for p in polygonPoints:
                print(p[0],",",p[1],",")

            source = self.order_Polygon(polygonPoints)

        elif len(points) == 1:

            print("Do 1 Microphone detection: ")

        # Converting back to LLA

        # Example usage
        #origin_lla = (-38.79073162749655, 143.55529204420282)  # Your origin in LLA

        origin_lla = origin

        print("Source:   ", source)

        for point in source:
            print("Printing point in spine:   ", point)
            #print(cartesian_to_lla(origin_lla,point))
            sourceLLA.append(self.cartesian_to_lla(origin_lla,point,points))

        for point in sourceLLA:
            print("Final Point(s):  ", point)

        # current_utc_time = datetime.now(timezone.utc)
        # formatted_timestamp = current_utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + current_utc_time.strftime('%z')
        # formatted_timestamp = formatted_timestamp[:-2] + ":" + formatted_timestamp[-2:]

        # vocalization = {
        #     "timestamp": current_utc_time,
        #     "sensorId": "SensorId",
        #     "species": "Felis catus",
        #     "microphoneLLA": [20,10,10] ,
        #     "animalEstLLA": [sourceLLA[0]],
        #     "animalTrueLLA": sourceLLA,
        #     "animalLLAUncertainty":50,
        #     "audioClip":"AAAAOAAAQLgAAAA4AACAtwAAAAAAAAA4AACAtwAAAAAAAAAAAACAtwAAADgAAAC4AAAAOA…",
        #     "confidence": 100,
        #     "sampleRate": 16000
        #   }

        #self.echo_api_send_locations_from_triangulation(clusterID,sourceLLA)

        #inserted_id = triangulatedEvents.insert_one(vocalization).inserted_id
        #print(f"Record inserted with ID: {inserted_id}")

        # vocalization = {
        #     "timestamp": current_utc_time,
        #     "sensorId": "SensorId",
        #     "species": "Felis catus",
        #     "microphoneLLA": [20,10,10] ,
        #     "animalEstLLA": [sourceLLA[0]],
        #     "animalTrueLLA": sourceLLA,
        #     "animalLLAUncertainty":50,
        #     "audioClip":"AAAAOAAAQLgAAAA4AACAtwAAAAAAAAA4AACAtwAAAAAAAAAAAACAtwAAADgAAAC4AAAAOA…",
        #     "confidence": 100,
        #     "sampleRate": 16000
        #   }

        #self.echo_api_send_locations_from_triangulation(clusterID,sourceLLA)

        #inserted_id = triangulatedEvents.insert_one(vocalization).inserted_id
        #print(f"Record inserted with ID: {inserted_id}")

        self.echo_api_send_locations_from_triangulation(clusterID, sourceLLA)

    def on_message(self, client, userdata, msg):
            
            
            try:
                print("Recieved event timings, processing via triangulation algorithm...")
                vocalization_events = json.loads(msg.payload)
                print(vocalization_events)
                clusterID = vocalization_events['clusterID']
                times = vocalization_events['times']
                print(f"Cluster ID: {clusterID} Times: {times}")
                self.triangulate(clusterID, times)

            except Exception as e:
                print(f"An error occurred: {e}", flush=True)
    
            # try:   
            #     points = json.loads(msg.payload)
            #     self.triangulate(points)
            #     #On message that comes from the clustering algorithm
            #     if 'type' in points:
            #        print()

            #     else:
            #        print()
                
            # except Exception as e:
            #     # Catch the exception and print it to the console
            #     print(f"An error occurred: {e}", flush=True)

    def on_subscribe(self, client, userdata, mid, granted_qos):
            

            print(f"Subscribed: message id {mid} with qos {granted_qos}")

    def echo_api_send_locations_from_triangulation(self, clusterID, sourceLLA):
            
            detection_event = {
                "clusterID": clusterID,
                "sourceLLA": sourceLLA       
            }

            url = 'http://ts-api-cont:9000/triangulation/update_record'
            x = requests.post(url, json = detection_event)
            print(x.text)

  # Main execution




if __name__ == "__main__":
    
    setupMics(document_list)
    triangulation = EchoTriangulation()
    #triangulation.triangulate()
    triangulation.execute()