import numpy as np
import json


def load_config(file_path: str):
    """Load the robot and environment config file"""
    output = {}
    
    with open(file_path, "r") as f:
        config = json.load(f)
    robots = config["robots"]
    obstacles = config["obstacles"]
    output["qr"] = obstacles[0]["width"]
    output["ql"] = [obstacle["length"] for obstacle in obstacles]
    output["ws"] = [obstacle["position"] for obstacle in obstacles]
    output["angles"] = [obstacle["angle"] for obstacle in obstacles]
    
    output["dr"] = robots[0]["radius"]
    output["q_initial"] = [position for robot in robots for position in robot["start_position"]]
    output["q_terminal"] = [position for robot in robots for position in robot["goal_position"]]
    return output

