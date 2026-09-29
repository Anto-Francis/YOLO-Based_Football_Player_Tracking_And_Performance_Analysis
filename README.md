# Football Player Tracking

A simple football player tracking system using YOLO and OpenCV.

## Features

- Detects and tracks football players using YOLO.
- Shows player bounding boxes, IDs and confidence scores.
- Allows players to be selected for detailed tracking.
- Calculates:
  - Current speed 
  - Distance travelled
  - Maximum speed
  - Average speed
- Displays movement trails for selected players.
- Uses the given pitch measurements to convert pixel positions into metres.

## Limitations

- Only works with a still camera.
- Zooming is not supported.
- Players need to be detected continuously for accurate tracking and statistics.
- The four corners must be selected based on the real measurements provided.

## Demo

Download and watch the demo video: `Demo_720p.mp4`

Refer to the demo pictures for examples:

- `Demo_Picture_1.png` — Initial setup where the user selects the four corners based on the measurements provided.
- `Demo_Picture_2.png` — Players detected by the model with bounding boxes, IDs and confidence scores.
- `Demo_Picture_3.png` — Three selected players showing their statistics and movement trails.

Short clip used for demo from: https://www.youtube.com/watch?v=M4Ccsmk3gcE
