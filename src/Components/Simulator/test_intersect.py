
import numpy as np

from entities.animal import Animal 
from entities.microphone import MicrophoneStation

# import entities.microphone

# the following function generated using GPT-4
def trilaterate(p1, r1, p2, r2, p3, r3):
    # Calculate relative positions of point 2 and point 3
    ex = (p2 - p1) / np.linalg.norm(p2 - p1)
    i = np.dot(ex, p3 - p1)
    ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
    ez = np.cross(ex, ey)

    # Calculate the distances
    d = np.linalg.norm(p2 - p1)
    j = np.dot(ey, p3 - p1)

    # Calculate the position of the intersection point
    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2) / (2 * j) - (i / j) * x

    # Calculate the z-coordinate, if possible
    z_squared = r1 ** 2 - x ** 2 - y ** 2
    if z_squared < 0:
        return None  # No intersection
    z = np.sqrt(z_squared)

    # Calculate the coordinates
    p = p1 + x * ex + y * ey + z * ez
    return p

# Set an animal true location
truth_animal = Animal(lla=(-48.0,134.0,10.0))
print(f'True animal LLA', truth_animal.getLLA())

# set the location of three sensors around this animal
sensor_1 = MicrophoneStation(lla=(-48.0,133.09,10.0))
sensor_2 = MicrophoneStation(lla=(-48.0,134.01,10.0))
sensor_3 = MicrophoneStation(lla=(-48.01,134.0,10.0))

print("distance 1", sensor_1.distance(truth_animal))
print("distance 2", sensor_2.distance(truth_animal))
print("distance 3", sensor_3.distance(truth_animal))

# Sphere centers and radii
# p1 = np.array([0, 0, 0])
# r1 = 5
# p2 = np.array([10, 0, 0])
# r2 = 7
# p3 = np.array([5, 10, 0])
# r3 = 9

# Sphere centers and radii
p1 = np.array(sensor_1.getECEF())
r1 = sensor_1.distance(truth_animal)
p2 = np.array(sensor_2.getECEF())
r2 = sensor_2.distance(truth_animal)
p3 = np.array(sensor_3.getECEF())
r3 = sensor_3.distance(truth_animal)

# Trilateration
intersection = trilaterate(p1, r1, p2, r2, p3, r3)
if intersection is not None:
    print(f"Intersection point: {intersection}")
    predicted_animal = Animal()
    predicted_animal.setECEF(intersection)
    print(f'Predicted animal LLA', predicted_animal.getLLA())
else:
    print("No intersection found.")