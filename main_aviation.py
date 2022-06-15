from FlightOperations import *

# Reading data and initializing our class
F = FlightOperations("airports_CO.dat", "borders_CO.dat")

# Visualizing the Voronoi diagram
F.plot()

# Finding the airports with the biggest and smallest areas of service
airport_max_area, airport_min_area = F.biggest_smallest_covg_area(True)
print("The airport with the biggest area of service is", airport_max_area,
      ", and the one with the smallest area is", airport_min_area, "\n")

# Finding the optimal location for a new airport
center, radius = F.build_new_airport(True)
print(
    "The location of the new airport should be at", round(
        center[0], 5), ",", round(center[1], 5),
    ". Also, the radius of the circle is", round(
        radius, 5), "\n")

# Finding the airports with the largest and smallest number of neighboring
# airports
airport_most_neigh, airport_less_neigh = F.most_less_crowded(True)
print(
    "The airport with the largest number of neighbors is",
    airport_most_neigh,
    ", and the one with the smallest number of neighbors is",
    airport_less_neigh, "\n")

# Deciding which pair of airports should be merged together
[airport_1, airport_2], min_distance, new_airport = F.merge_airports(True)
print("The airports", airport_1, "and", airport_2,
      "should be merged together at the new location",
      round(new_airport[0], 5), ",", round(new_airport[1], 5),
      ", since since the distance between them is only",
      round(min_distance, 5), "\n")
