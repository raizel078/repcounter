from pathlib import Path
from ultralytics import YOLO
import math as ma

video_file = Path('/home/nowa/Desktop/Project/Repcounter/8837221-hd_1080_2048_25fps.mp4')
model_path = Path('/home/nowa/Desktop/Project/models/yolo26n-pose.pt')
model = YOLO(model_path)


def processing_video(video):
    results = model.predict(source=str(video), stream=True, show=True)

    state = 'up'      # before the loop, so it's remembered between frames
    count = 0
    shallow =0

    for result in results:
        if result.keypoints is None:
            continue

        hip = result.keypoints.xy[0][12]
        knee  = result.keypoints.xy[0][14]

        dx =hip[0] - knee[0]
        dy = hip[1] - knee[1]

        angle = ma.degrees(ma.atan2(abs(dy), abs(dx)))   # 0° = flat thigh, 90° = standing

        if angle < 20:
            state = 'down'
        elif angle < 60 and state == 'up':
            state = 'half'

        if angle > 70:
            if state == 'down':
                count += 1
            elif state == 'half':
                shallow += 1
            state = 'up'

    print('Reps:', count, 'Shallow:', shallow)


processing_video(video_file)