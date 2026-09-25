import cv2
from ultralytics import YOLO
import numpy as np

# YOLO model
MODEL_PATH = "yolo11s.pt"

# Settings for YOLO model
CONFIDENCE = 0.20
IMAGE_SIZE = 1280

# Video file
VIDEO_PATH = "messi.mp4"

# Enter the real dimensions of the football pitch/area you know in metres (for pixel-real position conversion).
PITCH_LENGTH = 16.5
PITCH_WIDTH = 23.82

# Clicked players to track
TARGET_IDS = []

# Number of previous speed measurements to use for speed smoothing
SPEED_HISTORY_SIZE = 10

# Number of previous positions to show in trail
TRAIL_LENGTH = 200

# Colours for players
PLAYER_COLOURS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]


def calculate_distance(previous_position, current_position):

    previous_x, previous_y = previous_position
    current_x, current_y = current_position

    distance = (
        (current_x - previous_x) ** 2
        + (current_y - previous_y) ** 2
    ) ** 0.5

    return distance # (metres)


def calculate_speed(distance, elapsed_time):

    if elapsed_time <= 0:
        return 0

    speed = distance / elapsed_time

    return speed # (m/s)


def calculate_smooth_speed(speed_history):

    if len(speed_history) == 0:
        return 0

    average = sum(speed_history) / len(speed_history)

    return average # (m/s)


def draw_trail(frame, trail, colour):

    for i in range(1, len(trail)):

        cv2.line(
            frame,
            trail[i - 1],
            trail[i],
            colour,
            3
        )


def draw_player_information(
    frame,
    box,
    player_id,
    speed,
    colour
):

    x1, y1, x2, y2 = box

    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)

    player_position = (
        center_x,
        bottom_y
    )

    cv2.circle(
        frame,
        player_position,
        8,
        colour,
        -1
    )

    cv2.putText(
        frame,
        f"ID {player_id} Speed: {speed:.1f} m/s",
        (int(x1), int(y2) + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2
    )


def draw_stats_box(frame, players):

    selected_players = []

    for player_id in TARGET_IDS:

        if player_id in players:

            selected_players.append(
                (player_id, players[player_id])
            )

    if len(selected_players) == 0:
        return


    # Frame dimentions
    frame_height, frame_width = frame.shape[:2]

    # Each players stats box size
    player_width = 150
    box_height = 120

    # Total box width
    box_width = player_width * len(selected_players)

    # Bottom centre
    box_x = int((frame_width - box_width) / 2)
    box_y = frame_height - box_height - 20

    # Black background for box
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 0, 0),
        -1
    )

    # Show each players stats
    for i, (player_id, player) in enumerate(selected_players):

        player_x = box_x + (i * player_width)

        colour = player["colour"]

        # Line to seperate players
        if i > 0:

            cv2.line(
                frame,
                (player_x, box_y),
                (player_x, box_y + box_height),
                (100, 100, 100),
                1
            )

        # Player ID
        cv2.putText(
            frame,
            f"PLAYER {player_id}",
            (player_x + 10, box_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2
        )

        # Player distance
        cv2.putText(
            frame,
            f"Distance: {player['total_distance']:.0f} m",
            (player_x + 10, box_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Player max speed
        cv2.putText(
            frame,
            f"Max: {player['max_speed']:.1f} m/s",
            (player_x + 10, box_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Player avg speed
        cv2.putText(
            frame,
            f"Avg: {player['average_speed']:.1f} m/s",
            (player_x + 10, box_y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

# To create all variables needed to track one player
def create_player_data(colour):

    player = {

        "previous_position": None, #(m)

        "previous_frame_number": None, # (where player was detected successfully)

        "trail": [], # (pixel coord's to draw trail onto frame)

        "speed_history": [], #(m/s)

        "smooth_speed": 0, #(m/s)

        "total_distance": 0, #(m)

        "max_speed": 0, #(m/s)

        "tracked_time": 0,

        "average_speed": 0, #(m/s)

        "colour": colour

    }

    return player


# For clicks in pitch setup
def pitch_point_callback(event, x, y, flags, param):

    points = param

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append((x, y))

            print(
                f"Pitch point {len(points)} selected: "
                f"({x}, {y})"
            )


def select_pitch_corners(frame):

    points = []

    window_name = "Select Pitch Corners"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        pitch_point_callback,
        points
    )

    while len(points) < 4:

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            "Click the corners:",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "1: Length Start",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "2: Length End",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "3: Width End",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "4: Width Start",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "IN ORDER",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Draw points which have been clicked already
        for i, point in enumerate(points):

            cv2.circle(
                display_frame,
                point,
                8,
                (0, 0, 255),
                -1
            )

            # Number next to each pitch corner selected
            cv2.putText(
                display_frame,
                str(i + 1),
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Lines between selected corners
        for i in range(1, len(points)):

            cv2.line(
                display_frame,
                points[i - 1],
                points[i],
                (0, 255, 0),
                2
            )


        # Connect point 4 back to point 1
        if len(points) == 4:

            cv2.line(
                display_frame,
                points[3],
                points[0],
                (0, 255, 0),
                2
            )

        # Show modified frame
        cv2.imshow(
            window_name,
            display_frame
        )

        # Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            cv2.destroyWindow(window_name)

            return None

    cv2.destroyWindow(window_name)

    return points


def create_perspective_matrix(pitch_points):

    pixel_points = np.float32(pitch_points) # OpenCV perspective transformation format

    real_points = np.float32([
        [0, 0],
        [PITCH_LENGTH, 0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0, PITCH_WIDTH]
    ])

    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(
        pixel_points,
        real_points
    )

    return matrix


def pixel_to_metres(pixel_position, perspective_matrix):

    point = np.float32([
        [[
            pixel_position[0],
            pixel_position[1]
        ]]
    ])

    transformed_point = cv2.perspectiveTransform(
        point,
        perspective_matrix
    )

    x = transformed_point[0][0][0]
    y = transformed_point[0][0][1]

    return (
        float(x),
        float(y)
    )



# Runs when mouse is clicked during video
def mouse_callback(event, x, y, flags, param): #param = mouse_data dictionary

    global TARGET_IDS

    boxes = param["boxes"] # Current YOLO bounding boxes
    ids = param["ids"] # IDs belonging to those boxes
    players = param["players"] # Player info

    if (
        event != cv2.EVENT_LBUTTONDOWN
        and event != cv2.EVENT_RBUTTONDOWN
    ):
        return

    # For every detected player...
    for box, player_id in zip(boxes, ids):

        x1, y1, x2, y2 = box

        # Check if mouse click inside this players box
        if x1 <= x <= x2 and y1 <= y <= y2:

            player_id = int(player_id)

            # Left click = select
            if event == cv2.EVENT_LBUTTONDOWN:

                if player_id not in TARGET_IDS:

                    TARGET_IDS.append(player_id)

                    # If this player has never been selected before, create their stats
                    if player_id not in players:

                        colour_number = (
                            len(players)
                            % len(PLAYER_COLOURS)
                        )

                        colour = PLAYER_COLOURS[
                            colour_number
                        ]

                        players[player_id] = (
                            create_player_data(colour)
                        )

                    print(
                        f"Player {player_id} selected"
                    )

            # Right click = unselect
            elif event == cv2.EVENT_RBUTTONDOWN:

                if player_id in TARGET_IDS:

                    TARGET_IDS.remove(player_id)

                    print(
                        f"Player {player_id} unselected"
                    )

            break


# Main program

model = YOLO(MODEL_PATH)

video = cv2.VideoCapture(VIDEO_PATH)


if not video.isOpened():

    print("Error: Could not open video.")

    exit()


fps = video.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")


# Get first frame for pitch setup
success, first_frame = video.read()


if not success:

    print("Error: Could not read the first frame.")

    video.release()

    exit()


# User selects corners
pitch_points = select_pitch_corners(first_frame)


if pitch_points is None:

    video.release()

    cv2.destroyAllWindows()

    exit()


# Create perspective transformation matrix
perspective_matrix = create_perspective_matrix(
    pitch_points
)


print()
print("Pitch setup complete.")
print(f"Pitch length: {PITCH_LENGTH} metres")
print(f"Pitch width: {PITCH_WIDTH} metres")
print(f"Perspective matrix created.")
print()


# Back to first frame
video.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)


# Stats of players selected
players = {}


# Current YOLO detections, updated every frame
current_boxes = []
current_ids = []


window_name = "Football Player Tracking"

cv2.namedWindow(window_name)


# Give the mouse callback access to the current boxes, IDs and player data
mouse_data = {
    "boxes": current_boxes,
    "ids": current_ids,
    "players": players
}

cv2.setMouseCallback(
    window_name,
    mouse_callback,
    mouse_data
)


# Video loop
while True:

    # Read next frame
    success, frame = video.read()


    if not success:
        break


    current_frame_number = int(
        video.get(cv2.CAP_PROP_POS_FRAMES)
    )


    # YOLO tracking results
    results = model.track(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        persist=True
    )

    result = results[0] # Get results needed


    # YOLO tracking info
    if result.boxes.id is not None:

        # Get bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()

        # Get player IDs
        ids = result.boxes.id.cpu().numpy().astype(int)

    # No players detected
    else:
        boxes = []
        ids = []


    # Update mouse with latest YOLO boxes and IDs
    mouse_data["boxes"] = boxes
    mouse_data["ids"] = ids


    
    # To show bounding boxes, IDs and confidence if no players selected
    if len(TARGET_IDS) == 0:

        annotated_frame = result.plot()

    # if at least one player selected, hide YOLO bounding boxes, IDs and confidence, show custom tracking instead
    else:

        annotated_frame = frame.copy()


    # Track selected players
    for box, player_id in zip(boxes, ids):

        if player_id not in TARGET_IDS:
            continue # Check next player (next iteration of loop)


        # Get player stored data
        player = players[player_id]

        # Player position
        x1, y1, x2, y2 = box


        # Roughly players feet area in bounding box
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)

        current_pixel_position = (
            center_x,
            bottom_y
        )

        current_metre_position = pixel_to_metres(
            current_pixel_position,
            perspective_matrix
        )

        # Add pixel position to trail
        player["trail"].append(
            current_pixel_position
        )

        # Only keep the last 200 positions
        if len(player["trail"]) > TRAIL_LENGTH:
            player["trail"].pop(0)

        # Draw player trail
        draw_trail(
            annotated_frame,
            player["trail"],
            player["colour"]
        )


        # Calculate movement and speed
        if player["previous_position"] is not None:

            # Calculate how many frames have passed
            frame_difference = (
                current_frame_number
                - player["previous_frame_number"]
            )

            # Convert to seconds
            elapsed_time = frame_difference / fps


            
            # If player stops being tracked...
            if frame_difference > 1:

                # Don't know where the player moved, so don't calculate distance during the missing frames

                player["smooth_speed"] = 0

                # Clear old speed measurements
                player["speed_history"].clear()


            # Player was detected in previous frame...
            else:

                # Calculate distance moved
                distance = calculate_distance(
                    player["previous_position"],
                    current_metre_position
                ) #(m)


                player["total_distance"] += distance


                # Calculate speed
                speed = calculate_speed(
                    distance,
                    elapsed_time
                )


                player["speed_history"].append(
                    speed
                )


                # Only keep the last 10 speeds (for speed smoothing)
                if len(player["speed_history"]) > SPEED_HISTORY_SIZE:
                    player["speed_history"].pop(0)


                # Calculate smooth speed
                player["smooth_speed"] = (
                    calculate_smooth_speed(
                        player["speed_history"]
                    )
                )


                # Update max speed
                if (
                    player["smooth_speed"]
                    > player["max_speed"]
                ):

                    player["max_speed"] = (
                        player["smooth_speed"]
                    )


                # Time player is actually tracked
                player["tracked_time"] += elapsed_time


                # Calculate avg speed
                if player["tracked_time"] > 0:

                    player["average_speed"] = (
                        player["total_distance"]
                        / player["tracked_time"]
                    )


                print(
                    f"Player {player_id} | "
                    f"Speed: "
                    f"{player['smooth_speed']:.1f} m/s"
                )


        # Save current position and frame number
        # Save position in metres for next distance calc

        player["previous_position"] = (
            current_metre_position
        )

        player["previous_frame_number"] = (
            current_frame_number
        )


        # Draw player info
        draw_player_information(
            annotated_frame,
            box,
            player_id,
            player["smooth_speed"],
            player["colour"]
        )


    # Draw stats box
    if len(TARGET_IDS) > 0:

        draw_stats_box(
            annotated_frame,
            players
        )


    # Show current modified frame
    cv2.imshow(
        window_name,
        annotated_frame
    )


   # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()

cv2.destroyAllWindows()
