
import numpy as np

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

# Sphere centers and radii
p1 = np.array([0, 0, 0])
r1 = 5
p2 = np.array([10, 0, 0])
r2 = 7
p3 = np.array([5, 10, 0])
r3 = 9

# Trilateration
intersection = trilaterate(p1, r1, p2, r2, p3, r3)
if intersection is not None:
    print(f"Intersection point: {intersection}")
else:
    print("No intersection found.")