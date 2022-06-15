# Flight-Planning

Computational application using Voronoi Diagrams that addresses features of a flight plan. The functionality required evaluates aspects of the flight plan, given a set of airport sites, and departure and destination points of the flight. These aspects include global and local properties such as preprocessing of airport sites, nearest and farthest airports, types of flight paths or routes, potential threats, and desolated areas.

The requirements are:

- Read airport and border locations from input data files. The airport’s input file contains extra information on altitude, cities and name.
- Airport coverage area. It is always important to know how large and what population is serviced by a given airport; this helps minimizing the ration of fuel to passenger number. Find the airport names that cover the biggest and smallest areas of service. 
- Build new airports. Suppose you want to explore desolated areas to gather information on where it will be beneficial to build new airports, regardless of population and other factors. You can find such information by finding the largest circle centered within the convex hull and enclosing none of the already existing airports. Report this circle as the location of its center and corresponding radius
- Most and least crowded airports. In case of an emergency landing it is important to know how populated are the airports in order to cause the less damage. Report the names, not locations, of the airports with the smallest and the largest number of neighboring airports. Make sure that this search problem is solved in linear time in the structural complexity of the Voronoi diagram.
- Merging of airports. Now you wish to find out if it is possible to get rid of airports that are too close to each other. If that is the case, merging two airports could lead to less air traffic and more efficiently used airways. You can find such information by finding the closest pair of airports and reporting their position as a pair of locations. We can then insert a new airport, located at the mean location of the closest pair, and delete the pair in the airport network.
