import json
import numpy as np

# THIS SCRIPT IS JUST RAN ONCE TO GENERATE THE CONFIG FILES

problem_name = "door2"
if problem_name == "maze":
    qr = 0.1
    dr = 0.2
    ql = 2 / 3**0.5
    ws = [
        [-4, -4],
        [-4, 0],
        [-4, 2],
        [-4 + ql / 2, 3],
        [-4 + 3 * ql / 2, -3],
        [-4 + 3 * ql / 2, -1],
        [-4 + 3 * ql / 2, 1],
        [-4 + 3 * ql / 2, 3],
        [-4 + 3 * ql / 2, -3],
        [-4 + 3 * ql / 2, 3],
        [-4 + 2 * ql, -2],
        [-4 + 2 * ql, 2],
        [-4 + 3 * ql, -2],
        [-4 + 3 * ql, 0],
        [-4 + 3 * ql, 2],
        [-4 + 7 * ql / 2, -3],
        [-4 + 7 * ql / 2, 1],
        [-4 + 9 * ql / 2, -3],
        [-4 + 9 * ql / 2, -1],
        [-4 + 9 * ql / 2, -1],
        [-4 + 9 * ql / 2, 1],
        [-4 + 9 * ql / 2, 3],
        [-4 + 5 * ql, 0],
        [-4 + 6 * ql, -2],
        [-4 + 6 * ql, 0],
        [-4 + 6 * ql, 2],
    ]
    angles = [
        np.pi / 3,
        -np.pi / 3,
        -np.pi / 3,
        0,
        -np.pi / 3,
        np.pi / 3,
        -np.pi / 3,
        np.pi / 3,
        np.pi / 3,
        -np.pi / 3,
        0,
        0,
        np.pi / 3,
        np.pi / 3,
        np.pi / 3,
        0,
        0,
        np.pi / 3,
        -np.pi / 3,
        np.pi / 3,
        np.pi / 3,
        -np.pi / 3,
        0,
        np.pi / 3,
        -np.pi / 3,
        np.pi / 3,
    ]
    # q_initial = [0, -5, 0, 5, -5, 0, 5, 0, -5, -5, 5, 5, 5, -5, -5, 5]
    # q_terminal = [0, 5, 0, -5, 5, 0, -5, 0, 5, 5, -5, -5, -5, 5, 5, -5]
    q_initial = [0, -5, 0, 5]
    q_terminal = [0, 5, 0, -5]
    ql = [ql] * len(ws)
elif problem_name == "room":
    qr = 0.1
    dr = 0.2
    ql = [3, 4, 1, 5, 0.6, 2.6, 2.6, 3, 0.6]
    ws = [[-1, 5], [-1, 2], [3, 2], [-5, 0], [0, 0], [0, -5], [2, 0], [2, 0], [2, -5]]
    angles = [
        -np.pi / 2,
        0,
        np.pi / 2,
        0,
        -np.pi / 2,
        np.pi / 2,
        -np.pi / 2,
        0,
        np.pi / 2,
    ]
    q_initial = [-3.2, -1.9, -3.2, 3.8, -2.2, 3.2, 0.4, 3.7]
    q_terminal = [0.4, 3.4, -2.4, -1.4, 3.2, -2.8, -3.6, 1.6]
elif problem_name == "door":
    qr = 0.5
    dr = 0.5
    ql = [1.1, 1.1, 1.1, 1.1]
    ws = [[-2.5, 0], [-2.5, -2], [1.4, 0], [1.4, -2]]
    angles = [0, 0, 0, 0]
    q_initial = [-2, -4, 2, -4]
    q_terminal = [2, 2, -2, 2]
elif problem_name == "door2":
    qr = 0.5  # width of the obstacle
    ql = [1.1, 1.1]  # length of the obstacle
    ws = [[-2.5, 0], [1.4, 0]]  # starting points of the obstacle
    angles = [0, 0]  # angles of the obstacle

    dr = 0.5  # radius of drone
    q_initial = [-2, -2, 2, -2, 2, 2, -2, 2]
    q_terminal = [2, 2, -2, 2, -2, -2, 2, -2]
else:
    raise NotImplementedError

data = {
    "obstacles": [],
    "robots": [],
}

for i in range(len(ql)):
    data["obstacles"].append(
        {
            "id": i,
            "width": qr,
            "length": ql[i],
            "position": ws[i],
            "angle": angles[i],
        }
    )
for j in range(len(q_initial) // 2):
    data["robots"].append(
        {
            "id": j,
            "radius": dr,
            "start_position": [q_initial[2 * j], q_initial[2 * j + 1]],
            "goal_position": [q_terminal[2 * j], q_terminal[2 * j + 1]],
        }
    )

import os


with open(f"config/{problem_name}.json", "w") as f:
    json.dump(data, f, indent=4)

