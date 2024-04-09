import re
import csv
import numpy as np


# Function to safely evaluate mathematical expressions within the strings
def safe_eval(expr):
    # Replace sqrt with **0.5 for square root calculation
    expr = re.sub(r"sqrt\((.*?)\)", r"(\1)**0.5", expr)
    return np.array(eval(expr), dtype=float)


def parse_position_velocity(row, start_index):
    position_str = "(" + ", ".join(row[start_index : start_index + 3]) + ")"
    velocity_str = "(" + ", ".join(row[start_index + 3 : start_index + 6]) + ")"
    return safe_eval(position_str), safe_eval(velocity_str)


# Initialize a new list to hold the corrected data with the updated parsing function
def special_positions_velocities(solution = None):
    corrected_data_safe = []

    with open("special.csv", mode="r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            solution = row[0]
            bodies_data = []
            for i in range(3):  # Assuming 3 bodies
                position, velocity = parse_position_velocity(row, 1 + i * 6)
                bodies_data.append({"position": position, "velocity": velocity})
            corrected_data_safe.append({"solution": solution, "bodies": bodies_data})
    return corrected_data_safe
