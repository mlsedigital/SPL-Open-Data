import numpy as np
import json
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Default connections between joints. Change these as you please
connections = [
    ("R_EYE", "L_EYE"),
    ("R_EYE", "NOSE"),
    ("L_EYE", "NOSE"),
    ("R_EYE", "R_EAR"),
    ("L_EYE", "L_EAR"),
    ("R_SHOULDER", "L_SHOULDER"),
    ("R_SHOULDER", "R_ELBOW"),
    ("L_SHOULDER", "L_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_HIP"),
    ("L_SHOULDER", "L_HIP"),
    ("R_HIP", "L_HIP"),
    ("R_HIP", "R_KNEE"),
    ("L_HIP", "L_KNEE"),
    ("R_KNEE", "R_ANKLE"),
    ("L_KNEE", "L_ANKLE"),
    ("R_WRIST", "R_1STFINGER"),
    ("R_WRIST", "R_5THFINGER"),
    ("L_WRIST", "L_1STFINGER"),
    ("L_WRIST", "L_5THFINGER"),
    ("R_ANKLE", "R_1STTOE"),
    ("R_ANKLE", "R_5THTOE"),
    ("L_ANKLE", "L_1STTOE"),
    ("L_ANKLE", "L_5THTOE"),
    ("R_ANKLE", "R_CALC"),
    ("L_ANKLE", "L_CALC"),
    ("R_1STTOE", "R_5THTOE"),
    ("L_1STTOE", "L_5THTOE"),
    ("R_1STTOE", "R_CALC"),
    ("L_1STTOE", "L_CALC"),
    ("R_5THTOE", "R_CALC"),
    ("L_5THTOE", "L_CALC"),
    ("R_1STFINGER", "R_5THFINGER"),
    ("L_1STFINGER", "L_5THFINGER"),
    ("R_1STFINGER", "R_5THFINGER"),
    ("L_1STFINGER", "L_5THFINGER"),
]

JOINT_ALIASES = {
    "R_EYE": ["RIGHT_EYE"],
    "L_EYE": ["LEFT_EYE"],
    "R_EAR": ["RIGHT_EAR"],
    "L_EAR": ["LEFT_EAR"],
    "R_SHOULDER": ["RIGHT_SHOULDER"],
    "L_SHOULDER": ["LEFT_SHOULDER"],
    "R_ELBOW": ["RIGHT_ELBOW"],
    "L_ELBOW": ["LEFT_ELBOW"],
    "R_WRIST": ["RIGHT_WRIST"],
    "L_WRIST": ["LEFT_WRIST"],
    "R_HIP": ["RIGHT_HIP"],
    "L_HIP": ["LEFT_HIP"],
    "R_KNEE": ["RIGHT_KNEE"],
    "L_KNEE": ["LEFT_KNEE"],
    "R_ANKLE": ["RIGHT_ANKLE"],
    "L_ANKLE": ["LEFT_ANKLE"],
    "R_1STTOE": ["RIGHT_BIG_TOE"],
    "L_1STTOE": ["LEFT_BIG_TOE"],
    "R_5THTOE": ["RIGHT_SMALL_TOE"],
    "L_5THTOE": ["LEFT_SMALL_TOE"],
    "R_CALC": ["RIGHT_HEEL"],
    "L_CALC": ["LEFT_HEEL"],
    "R_1STFINGER": ["RIGHT_FIRST_FINGER_MCP"],
    "L_1STFINGER": ["LEFT_FIRST_FINGER_MCP"],
    "R_5THFINGER": ["RIGHT_FIFTH_FINGER_MCP"],
    "L_5THFINGER": ["LEFT_FIFTH_FINGER_MCP"],
}


def _resolve_joint_name(joint_name, available_joints):
    if joint_name in available_joints:
        return joint_name
    for alias in JOINT_ALIASES.get(joint_name, []):
        if alias in available_joints:
            return alias
    return None


def animate_trial(
    path_to_json,
    connections=connections,
    xbuffer=4.0,
    ybuffer=4.0,
    zlim=8.0,
    elev=15.0,
    azim=40.0,
    player_color="purple",
    player_lw=2,
    ball_color="#ee6730",
    ball_size=20.0,
    show_court=True,
    notebook_mode=True,
):
    """
    Function to animate a single trial of 3D pose data.

    Parameters:
    -----------
    - path_to_json: str
        The path to the JSON file containing the 3D pose data.
    - connections: list of tuples
        A list of tuples, where each tuple contains two strings representing the joints to connect.
    - xbuffer: float
        The buffer to add to the x-axis limits.
    - ybuffer: float
        The buffer to add to the y-axis limits.
    - zlim: float
        The limit for the z-axis height.
    - elev: float
        The elevation angle for the 3D plot.
    - azim: float
        The azimuth angle for the 3D plot.
    - player_color: str
        The color to use for the player lines.
    - player_lw: float
        The line width to use for the player lines.
    - ball_color: str
        The color to use for the ball.
    - ball_size: float
        The size to use for the ball.
    - show_court: bool
        Whether to show the basketball court in the background.
    - notebook_mode: bool
        Whether function is used within a Jupyter notebook.

    Returns:
    --------
    - anim: matplotlib.animation.FuncAnimation
        The animation object created by the function.
    """

    if notebook_mode:
        plt.rcParams["animation.html"] = "jshtml"

    if show_court:
        try:
            from mplbasketball.court3d import draw_court_3d
        except ModuleNotFoundError:
            print("mplbasketball not installed. Cannot show court.")
            show_court = False

    with open(path_to_json, "r") as f:
        data = json.load(f)

    player_joint_dict = {}
    ball_data_array = []

    N_frames = len(data["tracking"])

    # The block of code below returns a dictionary, where each key is a 3D time series for coordinates of a joint.
    # Note that it is a list of N_frames elements, each of which is a 3-element list.
    # If you want to use numpy, you will have to cast it into a numpy array.
    for frame_data in data["tracking"]:
        for joint in frame_data["data"]["player"]:
            if joint not in player_joint_dict:
                player_joint_dict[joint] = []
            player_joint_dict[joint].append(frame_data["data"]["player"][joint])
        ball_data_array.append(frame_data["data"]["ball"])

    # For convenience, we will cast everything to numpy arrays here, but you can keep them as lists if you prefer.
    for joint in player_joint_dict:
        player_joint_dict[joint] = np.array(player_joint_dict[joint], dtype=float)
    ball_data_array = np.array(ball_data_array, dtype=float)
    available_joints = set(player_joint_dict.keys())

    resolved_connections = []
    seen_connections = set()
    for part1, part2 in connections:
        resolved_part1 = _resolve_joint_name(part1, available_joints)
        resolved_part2 = _resolve_joint_name(part2, available_joints)
        if resolved_part1 is None or resolved_part2 is None:
            continue
        connection = (resolved_part1, resolved_part2)
        if connection in seen_connections:
            continue
        seen_connections.add(connection)
        resolved_connections.append(connection)

    if not resolved_connections:
        raise ValueError("No valid connections available for the current trial.")

    right_hip_joint = _resolve_joint_name("R_HIP", available_joints)
    left_hip_joint = _resolve_joint_name("L_HIP", available_joints)

    # Animate the data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Set up initial plot properties
    ax.set_zlim([0, zlim])
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    # Prepare the lines to be updated
    lines = [
        ax.plot([], [], [], c=player_color, lw=player_lw)[0]
        for _ in resolved_connections
    ]

    (ball,) = ax.plot([], [], [], "o", markersize=ball_size, c=ball_color)

    def update(frame):

        # Use the average of the right and left hip to center the view.
        if right_hip_joint is not None and left_hip_joint is not None:
            rh_xy = player_joint_dict[right_hip_joint][frame][:2]
            lh_xy = player_joint_dict[left_hip_joint][frame][:2]
            mh_xy = (rh_xy + lh_xy) / 2
        else:
            # Fallback to any available joint if hip markers are unavailable.
            first_joint = next(iter(player_joint_dict))
            mh_xy = player_joint_dict[first_joint][frame][:2]

        ax.set_xlim([mh_xy[0] - xbuffer, mh_xy[0] + xbuffer])
        ax.set_ylim([mh_xy[1] - ybuffer, mh_xy[1] + ybuffer])

        # Update the line data for each connection
        for line, connection in zip(lines, resolved_connections):
            part1, part2 = connection
            x = [
                player_joint_dict[part1][frame, 0],
                player_joint_dict[part2][frame, 0],
            ]
            y = [
                player_joint_dict[part1][frame, 1],
                player_joint_dict[part2][frame, 1],
            ]
            z = [
                player_joint_dict[part1][frame, 2],
                player_joint_dict[part2][frame, 2],
            ]
            line.set_data_3d(x, y, z)

        # Update ball data
        x = ball_data_array[frame, 0]
        y = ball_data_array[frame, 1]
        z = ball_data_array[frame, 2]
        ball.set_data_3d([x], [y], [z])

    if show_court is True:
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)
        draw_court_3d(ax, origin=np.array([0.0, 0.0]), line_width=2)

    # plt.tight_layout()
    plt.subplots(layout="constrained")
    plt.close()

    sampling_rate = data.get("sampling_rate", 30)
    if not isinstance(sampling_rate, (float, int)) or sampling_rate <= 0:
        sampling_rate = 30

    anim = FuncAnimation(fig, update, frames=N_frames, interval=1000 / sampling_rate)
    return anim
