import numpy as np
import matplotlib.pyplot as plt
import json
import itertools


def collision_check(q_pred, config_json, instance_index=None):
    traj_count = q_pred.shape[0]  # number of instances, 100 in default exp.
    time_count = q_pred.shape[1]  # number of time steps, 200 in default exp.
    robot_count = q_pred.shape[2] // 2  # number of robots
    # assume all robots have the same radius
    robot_radius = config_json["robots"][0]["radius"]

    # check robot-robot collision
    for instance in range(traj_count):
        if instance_index is not None and instance != instance_index:
            continue
        for time in range(time_count):
            for i, j in itertools.combinations(range(robot_count), 2):
                q_i = q_pred[instance, time, 2 * i : 2 * i + 2]
                q_j = q_pred[instance, time, 2 * j : 2 * j + 2]
                distance = np.linalg.norm(q_i - q_j)
                if distance < 2 * robot_radius:
                    print(
                        "Robot-robot collision: instance {}, time {}, robot {} and robot {}".format(
                            instance, time, i, j
                        )
                    )
                    if distance < 0.995 * 2 * robot_radius:
                        print("Distance: {}".format(distance))

    # check robot-obstacle collision
    obstacles = config_json["obstacles"]

    for instance in range(traj_count):
        if instance_index is not None and instance != instance_index:
            continue
        for time in range(time_count):
            for i in range(robot_count):
                q_i = q_pred[instance, time, 2 * i : 2 * i + 2]
                for obstacle in obstacles:
                    position = np.array(obstacle["position"])
                    distance = np.linalg.norm(q_i - position)
                    # Assume all obstacles are circles, width and length are the same
                    assert obstacle["width"] == obstacle["length"]
                    if distance < robot_radius + obstacle["width"]:
                        print(
                            "Robot-obstacle collision: instance {}, time {}, robot {} and obstacle {}".format(
                                instance, time, i, obstacle["id"]
                            )
                        )
                        if distance < 0.995 * (robot_radius + obstacle["width"]):
                            print("Distance: {}".format(distance))


def derivative_check(q_pred, config_json, instance_index=None, derivative_order=2):
    traj_count = q_pred.shape[0]  # number of instances, 100 in default exp.
    time_count = q_pred.shape[1]  # number of time steps, 200 in default exp.
    robot_count = q_pred.shape[2] // 2  # number of robots

    dt = 1 / time_count  # time interval

    for instance in range(traj_count):
        if instance_index is not None and instance != instance_index:
            continue
        # for each robot, compute the first-order derivative by taking differential of q_pred
        for i in range(robot_count):
            q_i = q_pred[instance, :, 2 * i : 2 * i + 2]
            for order in range(derivative_order + 1):
                plt.plot(q_i[:, 0], label="{}q_i_x".format("d" * order))
                plt.plot(q_i[:, 1], label="{}q_i_y".format("d" * order))
                q_i = np.diff(q_i, axis=0) / dt
            plt.legend()
            plt.title(
                "Derivatives for instance {}, robot {}".format(
                    derivative_order, instance, i
                )
            )
            plt.show()


if __name__ == "__main__":
    q_pred_filepath = "results/2024-06-11_13-46-44/q_pred.npy"
    config_json_path = "src/config/demo_32.json"

    q_pred = np.load(q_pred_filepath)
    config_json = json.load(open(config_json_path, "r"))

    i = 0

    #collision_check(q_pred=q_pred, config_json=config_json, instance_index=i)
    derivative_check(q_pred=q_pred, config_json=config_json, instance_index=i)
