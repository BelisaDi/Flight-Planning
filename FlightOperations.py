from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import matplotlib.path as mplPath


class FlightOperations:

    """
    Class that stores the map of interest and airports with latitude,
    longitude, altitude, city, department and name. In particular, it
    stores the Voronoi diagram with airports as sites.

    Atributes:
    ------------
    vor: Voronoi diagram of airports based on their longitude and latitude
    altitudes: numpy array that stores airport altitudes
    additional_info: numpy array of size (N,3) (where N is the number of
        airports), that stores city, department and airport name in order
    border: numpy array of size (M,2) (where M is the number of line
        segments), that stores the map to consider

    Public methods:
    -------------
    plot(): plots the Voronoi diagram and the considered map
    biggest_smallest_covg_area(plot = False): finds the airport names that
        cover the biggest and smallest areas of service, and plots if
        indicated
    most_less_crowded(plot = False): finds the airport names with the smallest
        and the largest number of neighboring airports, and plots if indicated
    build_new_airport(plot = False): finds the largest circle centered within
        the convex hull and within the map considered, enclosing none of the
        already existing airports, and plots if indicated
    merge_airports(plot = False): finds the closest pair of airports,
        and plots if indicated

    """

    def __init__(self, airports_filename, map_filename):
        """
        Parameters:
        -------------
        airports_filename: filename with airport data, including its extension
        map_filename: filename with border data, including its extension
        """
        with open(airports_filename) as file:
            initial = True
            dat = reader(file)
            for k in dat:
                row = [x.strip()
                       for x in k[0].replace('"', '').split('  ') if x != '']
                if initial:
                    points = np.array([[float(row[1]), float(row[0])]])
                    self.altitudes = np.array(float(row[2]))
                    self.additional_info = np.array([[row[3], row[4], row[5]]])
                    initial = False
                else:
                    points = np.append(
                        points, [[float(row[1]), float(row[0])]], axis=0)
                    self.altitudes = np.append(
                        self.altitudes, [float(row[2])])
                    self.additional_info = np.append(
                        self.additional_info, [[row[3], row[4], row[5]]],
                        axis=0)

        with open(map_filename) as file:
            initial = True
            dat = reader(file)
            for k in dat:
                row = [x.strip() for x in k[0].replace('"', '').split('  ')
                       if x != '']
                if initial:
                    self.border = np.array([[float(row[1]), float(row[0])]])
                    initial = False
                else:
                    self.border = np.append(
                        self.border, [[float(row[1]), float(row[0])]], axis=0)

        self.vor = Voronoi(points)

    def plot(self):
        """
        Plots the Voronoi diagram and the edges of the map considered
        """
        fig = voronoi_plot_2d(self.vor, show_vertices=False,
                              line_colors="dodgerblue")
        plt.plot(self.border[:, 0], self.border[:, 1], 'k')
        plt.ylabel('Latitude (degrees)')
        plt.xlabel('Longitude (degrees)')
        plt.title("Voronoi Map")
        plt.show()

    def _cell_area(self, cell):
        """
        Calculate the area of a cell or region of the Voronoi diagram using the
        shoelace rule

        Parameters
        -------------
        cell: Voronoi diagram region given in the format of the .regions
            attribute

        Output
        -------------
        Area of the region

        """
        v_list = [self.vor.vertices[x] for x in cell]
        n = len(v_list)
        area = 0

        for i in range(n):
            j = (i + 1) % n
            area += v_list[i][0] * v_list[j][1]
            area -= v_list[j][0] * v_list[i][1]

        return abs(area) / 2

    def biggest_smallest_covg_area(self, plot=False):
        """
        Finds the airport names that cover the biggest and smallest areas of
        service

        Parameters
        -------------
        plot: boolean, if True, plots the Voronoi diagram indicating the
            biggest area in light blue, and the smallest in yellow.

        Output
        -------------
        airport_max_area: name of the airport with the biggest area of
            service
        airport_min_area: name of the airport with the smallest area of
            service
        """
        initial = True

        for i in range(len(self.vor.point_region)):
            reg_index = self.vor.point_region[i]
            cell = self.vor.regions[reg_index]
            if -1 not in cell:
                area = self._cell_area(cell)
                if initial:
                    min_area = area
                    max_area = area
                    min_area_idx = i
                    max_area_idx = i
                    initial = False
                if area <= min_area:
                    min_area = area
                    min_area_idx = i
                if area >= max_area:
                    max_area = area
                    max_area_idx = i

        airport_max_area = self.additional_info[max_area_idx, 2]
        airport_min_area = self.additional_info[min_area_idx, 2]

        if plot:
            fig = voronoi_plot_2d(
                self.vor,
                show_vertices=False,
                line_colors="grey",
                line_alpha=0.8)
            plt.plot(self.border[:, 0], self.border[:, 1], 'k')

            reg_max = self.vor.regions[self.vor.point_region[max_area_idx]]
            reg_min = self.vor.regions[self.vor.point_region[min_area_idx]]

            polygon_max = [self.vor.vertices[i] for i in reg_max]
            plt.fill(*zip(*polygon_max), color="lightskyblue",
                     label=airport_max_area + ": " + str(round(max_area, 5)))

            polygon_min = [self.vor.vertices[i] for i in reg_min]
            plt.fill(*zip(*polygon_min), color="gold",
                     label=airport_min_area + ": " + str(round(min_area, 5)))

            plt.legend(loc="best", fancybox=True,
                       title="Area")
            plt.ylabel('Latitude (degrees)')
            plt.xlabel('Longitude (degrees)')
            plt.title("Airports with biggest and smallest coverage area")
            plt.show()

        return airport_max_area, airport_min_area

    def most_less_crowded(self, plot=False):
        """
        Finds the airport names with the smallest and the largest number of
        neighboring airports

        Parameters
        -------------
        plot: boolean, if True, plots the Voronoi diagram indicating the most
            crowded in light blue, and the less crowded in yellow.

        Output
        -------------
        airport_most_crowded: name of the airport with the largest number
            of neighbors
        airport_less_crowded: name of the airport with the smallest number
            of neighbors
        """
        initial = True

        neighs = np.zeros((len(self.vor.points)))

        for couple in self.vor.ridge_points:
            neighs[couple[0]] += 1
            neighs[couple[1]] += 1

        most_crowded_idx = np.argmax(neighs)
        less_crowded_idx = np.argmin(neighs)
        neigh_most = neighs[most_crowded_idx]
        neigh_less = neighs[less_crowded_idx]

        airport_most_crowded = self.additional_info[most_crowded_idx, 2]
        airport_less_crowded = self.additional_info[less_crowded_idx, 2]

        if plot:
            fig = voronoi_plot_2d(
                self.vor,
                show_vertices=False,
                line_colors="grey",
                line_alpha=0.8)
            plt.plot(self.border[:, 0], self.border[:, 1], 'k')

            reg_max = self.vor.regions[self.vor.point_region[most_crowded_idx]]
            reg_min = self.vor.regions[self.vor.point_region[less_crowded_idx]]

            polygon_max = [self.vor.vertices[i] for i in reg_max]
            plt.fill(*zip(*polygon_max), color="lightskyblue",
                     label=airport_most_crowded + ": " + str(neigh_most))

            polygon_min = [self.vor.vertices[i] for i in reg_min]
            plt.fill(*zip(*polygon_min), color="gold",
                     label=airport_less_crowded + ": " + str(neigh_less))

            plt.legend(loc="best", fancybox=True,
                       title="Number of Neighbors")
            plt.ylabel('Latitude (degrees)')
            plt.xlabel('Longitude (degrees)')
            plt.title("Airports with largest and smallest number of neighbors")
            plt.show()

        return airport_most_crowded, airport_less_crowded

    def _direction(self, p_i, p_j, p_k):
        """
        Determines whether the vector p_i,p_j is counterclockwise
        or clockwise with respect to the vector p_i,p_k

        Parameters
        --------------
        p_i: coordinates of origin p_i
        p_j: coordinates of endpoint p_j
        p_k: coordinates of endpoint p_k

        Output
        ---------------
        Cross product of the vectors
        """
        return np.cross(p_k - p_i, p_j - p_i)

    def _on_segment(self, p_i, p_j, p_k):
        """
        Determines whether a point p_k known to be colinear with a
        segment p_i,p_j lies on that segment

        Parameters
        --------------
        p_i: coordinates of one endpoint of the segment
        p_j: coordinates of the other endpoint of the segment
        p_k: coordinates of the known colinear point

        Output:
        --------------
        True or False
        """
        if (min(p_i[0], p_j[0]) <= p_k[0] <= max(p_i[0], p_j[0])) and (
                min(p_i[1], p_j[1]) <= p_k[1] <= max(p_i[1], p_j[1])):
            return True
        else:
            return False

    def _segments_intersect(self, p1, p2, p3, p4):
        """
        Determines whether or not segments p1,p2 and p3,p4 intersect

        Parameters
        ---------------
        p1: coordinates of one endpoint of first segment
        p2: coordinates of the other endpoint of first segment
        p3: coordinates of one endpoint of second segment
        p4: coordinates of the other endpoint of second segment

        Output
        ---------------
        True or False
        """
        d1, d2, d3, d4 = self._direction(
            p3, p4, p1), self._direction(
            p3, p4, p2), self._direction(
            p1, p2, p3), self._direction(
            p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
                (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        elif d1 == 0 and self._on_segment(p3, p4, p1):
            return True
        elif d2 == 0 and self._on_segment(p3, p4, p2):
            return True
        elif d3 == 0 and self._on_segment(p1, p2, p3):
            return True
        elif d4 == 0 and self._on_segment(p1, p2, p4):
            return True
        else:
            return False

    def _right_ray_intersect(self, p0, p1, p2):
        """
        Determines whether the right horizontal ray from p0
        intersects the line segment p1,p2

        Parameters
        ---------------
        p0: coordinates of origin of horizontal ray
        p1: coordinates of one endpoint of the line segment
        p2: coordinates of the other endpoint of the line segment

        Output
        ---------------
        True or False
        """
        p3 = np.array([max(p1[0], p2[0]) + 1, p0[1]])
        return self._segments_intersect(p0, p3, p1, p2)

    def _point_inside(self, p0, P):
        """
        Determines whether a point p0 is in the interior of a simple
        polygon P

        Parameters
        --------------
        p0: coordinates of point to evaluate
        P: list of polygon edges (not necessarily in order)

        Output
        --------------
        True if p0 is inside, False otherwise
        """
        counter = 0
        for edge in P:
            if self._right_ray_intersect(p0, edge[0], edge[1]):
                if((edge[0][1] < p0[1] and edge[1][1] >= p0[1]) or
                        (edge[1][1] < p0[1] and edge[0][1] >= p0[1])):
                    counter += 1
        if counter % 2 != 0:
            return True
        else:
            return False

    def build_new_airport(self, plot=False):
        """
        Finds the largest circle centered within the convex hull and within
        the map considered, enclosing none of the already existing airports

        Parameters
        -------------
        plot: boolean, if True, plots the Voronoi diagram, the convex hull
            and the largest empty circle

        Output
        -------------
        center: circle center coordinates
        radius: circle radius
        """
        hull_edges = []
        for j, vor_edge in enumerate(self.vor.ridge_vertices):
            if -1 in vor_edge:
                idx = self.vor.ridge_points[j]
                hull_edges.append(
                    [self.vor.points[idx[0]], self.vor.points[idx[1]]])

        m = len(self.border[:, 0])
        map_poly = []

        for i in range(m + 1):
            map_poly.append((self.border[i % m, 0], self.border[i % m, 1]))
        map_path = mplPath.Path(map_poly)

        possible_centers = []
        possible_radius = []

        for i, v in enumerate(self.vor.vertices):
            if self._point_inside(
                    v, hull_edges) and map_path.contains_point(
                    tuple(v)):
                possible_centers.append(v)
                for j, reg in enumerate(self.vor.regions):
                    if i in reg:
                        idx_site = self.vor.point_region.tolist().index(j)
                        site = self.vor.points[idx_site]
                        possible_radius.append(np.linalg.norm(site - v))
                        break

        max_idx = np.argmax(possible_radius)
        center = possible_centers[max_idx]
        radius = possible_radius[max_idx]

        if plot:
            fig, ax = plt.subplots()
            voronoi_plot_2d(self.vor, ax, show_vertices=False,
                            line_colors="grey", line_alpha=0.8)
            plt.plot(self.border[:, 0], self.border[:, 1], 'k')

            for edge in hull_edges:
                plt.plot([edge[0][0], edge[1][0]], [
                         edge[0][1], edge[1][1]], color="dodgerblue")

            circle = plt.Circle(tuple(center), radius,
                                fill=False, color="red")
            circle.set_label(radius)
            ax.add_artist(circle)
            plt.scatter(center[0], center[1], marker="*", color="red")

            plt.legend([circle], [round(radius, 5)], loc="best", fancybox=True,
                       title="Radius of Largest Empty Circle")

            plt.ylabel('Latitude (degrees)')
            plt.xlabel('Longitude (degrees)')
            plt.title("Recommendend location for new airport")
            plt.show()

        return center, radius

    def merge_airports(self, plot=False):
        """
        Finds the closest pair of airports and its mean as the new recommended
        point

        Parameters
        -------------
        plot: boolean, if True, plots the Voronoi diagram, with the closest
            pair of airports in red and the new location as a blue star

        Output
        -------------
        [airport_1, airport_2]: list with the coordinates of the two closest
            airports
        min_distance: distance between the resulting airports
        new_airport: coordinates of the mean of the two closest airports
        """

        initial = True

        for couple in self.vor.ridge_points:
            d = np.linalg.norm(
                self.vor.points[couple[0]] - self.vor.points[couple[1]])
            if initial:
                min_couple = couple
                min_distance = d
                initial = False
            if d < min_distance:
                min_couple = couple
                min_distance = d

        airport_1 = self.additional_info[min_couple[0], 2]
        airport_2 = self.additional_info[min_couple[1], 2]

        loc_1 = self.vor.points[min_couple[0]]
        loc_2 = self.vor.points[min_couple[1]]

        new_airport = np.mean(np.array([loc_1, loc_2]), axis=0)

        if plot:
            fig = voronoi_plot_2d(
                self.vor,
                show_vertices=False,
                show_points=False,
                line_colors="grey",
                line_alpha=0.3,
                zorder=1)
            plt.plot(self.border[:, 0], self.border[:, 1], 'k', zorder=1)

            plt.plot(self.vor.points[:, 0], self.vor.points[:, 1],
                     'ko', zorder=1, markersize=2, alpha=0.3)

            plt.scatter(loc_1[0], loc_1[1], color="red",
                        zorder=2, s=10, label=airport_1)
            plt.scatter(loc_2[0], loc_2[1], color="green",
                        zorder=2, s=10, label=airport_2)
            plt.scatter(new_airport[0], new_airport[1],
                        marker="*", color="blue", zorder=2, label=new_airport)

            plt.legend(loc="best", fancybox=True,
                       title="Old airports and location of new")

            plt.ylabel('Latitude (degrees)')
            plt.xlabel('Longitude (degrees)')
            plt.title("Recommended merge of airports")
            plt.show()

        return [airport_1, airport_2], min_distance, new_airport
