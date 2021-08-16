import csv
import math
import pickle

BASE_DIR = "/home/lincoln/lab/catkin_ws/src/drone_processor/src/"
# For daytime
LEPTON_G2O = BASE_DIR + "lepton_day_complete/output.g2o"
BOSON_G2O = BASE_DIR + "boson_day_complete/optimized.g2o"
BLACKFLY_G2O = BASE_DIR + "blackfly_day_complete/optimized.g2o"
PICKLE = "gnss_data_flight_1.p"

# For nightitme
LEPTON_G2O = BASE_DIR + "lepton_night_complete/output.g2o"
BOSON_G2O = BASE_DIR + "boson_night_complete/optimized.g2o"
BLACKFLY_G2O = BASE_DIR + "blackfly_night_complete/output.g2o"
PICKLE = "gnss_data_flight_4.p"

# seconds
BAG_TIME = 277

class Vertex:
    id = 0
    x = 0
    y = 0
    theta = 0

    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta

def Parse_g2o(filename):
    vertices = []
    with open(filename, 'r') as g2o_file:
        lines = g2o_file.readlines()
        for line in lines:
            words = line.split()
            if (words[0] == "VERTEX_SE2"):
                vertex = Vertex(int(words[1]),
                                float(words[2]),
                                float(words[3]),
                                float(words[4]))
                vertices.append(vertex)
    return vertices

def main():
    lepton_vertices = Parse_g2o(LEPTON_G2O)
    boson_vertices = Parse_g2o(BOSON_G2O)
    blackfly_vertices = Parse_g2o(BLACKFLY_G2O)
    gps_data = pickle.load(open(PICKLE, "rb"))

    lepton_timestep = float(BAG_TIME) / len(lepton_vertices)
    boson_timestep = float(BAG_TIME) / len(boson_vertices)
    blackfly_timestep = float(BAG_TIME) / len(blackfly_vertices)
    gps_timestep = float(BAG_TIME) / len(gps_data)

    outfile = BASE_DIR + "output.csv"

    gps_x_mult = 306.7274
    gps_y_mult = 230.0456
    gps_rotation = 1.291964

    lepton_x_mult = 9.287861 / gps_x_mult
    lepton_y_mult = 8.680618 / gps_y_mult
    lepton_t_offset = 0.0

    boson_x_mult = 4.47211 / gps_x_mult
    boson_y_mult = 2.7647436 / gps_y_mult
    boson_t_offset = 0

    blackfly_x_mult = 1.0 / gps_x_mult
    blackfly_y_mult = 1.0 / gps_y_mult
    blackfly_t_offset = 0.0

    gps_x_offset = gps_data[0][1]
    gps_y_offset = gps_data[0][0]

    gps_x_mult = 1.0
    gps_y_mult = 1.0


    with open(outfile, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Time", "Lepton X", "Lepton Y", "Lepton T", "Boson X", "Boson Y", "Boson T", "Blackfly X", "Blackfly Y", "Blackfly T", "GPS X", "GPS Y"])

        for timestep in range(BAG_TIME * 10):
            time = float(timestep) / 10.0

            lepton_vertex = int(math.floor(time / lepton_timestep))
            boson_vertex = int(math.floor(time / boson_timestep))
            blackfly_vertex = int(math.floor(time / blackfly_timestep))
            gps_vertex = int(math.floor(time / gps_timestep))

            lepton_x = str(lepton_vertices[lepton_vertex].x * lepton_x_mult)
            lepton_y = str(lepton_vertices[lepton_vertex].y * lepton_y_mult)
            lepton_t = str(lepton_vertices[lepton_vertex].theta + lepton_t_offset)

            boson_x = str(boson_vertices[boson_vertex].x * boson_x_mult)
            boson_y = str(boson_vertices[boson_vertex].y * boson_y_mult)
            boson_t = str(boson_vertices[boson_vertex].theta + boson_t_offset)

            blackfly_x = str(blackfly_vertices[blackfly_vertex].x * blackfly_x_mult)
            blackfly_y = str(blackfly_vertices[blackfly_vertex].y * blackfly_y_mult)
            blackfly_t = str(blackfly_vertices[blackfly_vertex].theta + blackfly_t_offset)

            gps_x = (gps_data[gps_vertex][1] - gps_x_offset)
            gps_y = (gps_data[gps_vertex][0] - gps_y_offset)

            gps_rot_x = gps_x_mult * (gps_x * math.cos(gps_rotation) - gps_y * math.sin((gps_rotation)))
            gps_rot_y = gps_y_mult * (gps_x * math.sin(gps_rotation) + gps_y * math.cos((gps_rotation)))

            csvwriter.writerow([str(time), lepton_x, lepton_y, lepton_t, boson_x, boson_y, boson_t, blackfly_x, blackfly_y, blackfly_t, gps_rot_x, gps_rot_y])



    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
