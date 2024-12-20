# Event Camera Explanation Video

This repository provides a simple script to generate a video to explain how the event camera works. You can use the generated video in your presentations to introduce the basic concepts of event camera. This video is inspired by the presentation by [Mueggler et al. (IROS 2014)](https://www.youtube.com/watch?v=LauQ6LWTkxM&t=38s).

![screenshot_ppt](docs/screenshot_ppt.jpeg)

## Requirements

- NumPy
- OpenCV
- tqdm
- PyVista

## Gallary

### Event only

```bash
python gen_video.py 
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event.mp4)
![docs/event.jpg](docs/event.jpg)

### Frame only

```bash
python gen_video.py --hide_event --show_frame
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/frame.mp4)
![docs/frame.jpg](docs/frame.jpg)

### Event only, fast rotation, don't stop rotation

```bash
python gen_video.py --speed 32 --rotation_duration 36 --rotation_speed 2 --stop_duration 0 --reduction_ratio 0.1 --frame_interval 480 --max_frames 2880  
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_fast.mp4)
![docs/event_fast.jpg](docs/event_fast.jpg)

### Frame only, fast rotation, don't stop rotation

```bash
python gen_video.py --average_frames 20 --speed 32 --rotation_duration 36 --rotation_speed 2 --stop_duration 0 --frame_interval 480 --max_frames 2880 --hide_event --show_frame
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/frame_fast.mp4)
![docs/frame_fast.jpg](docs/frame_fast.jpg)

### Event + Frame

```bash
python gen_video.py --show_frame
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_frame.mp4)
![docs/event_frame.jpg](docs/event_frame.jpg)

### Event only, camera1

```bash
python gen_video.py --camera 1
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_camera1.mp4)
![docs/event_camera1.jpg](docs/event_camera1.jpg)

### Event only, camera2

```bash
python gen_video.py --camera 2
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_camera2.mp4)
![docs/event_camera2.jpg](docs/event_camera2.jpg)

### Event only, dark theme

```bash
python gen_video.py --dark
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_dark.mp4)
![docs/event_dark.jpg](docs/event_dark.jpg)

### Frame only, dark theme

```bash
python gen_video.py --hide_event --show_frame --dark
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/frame_dark.mp4)
![docs/frame_dark.jpg](docs/frame_dark.jpg)

### Event only, fast rotation, dark theme

```bash
python gen_video.py --speed 32 --rotation_duration 36 --rotation_speed 2 --stop_duration 0 --reduction_ratio 0.1 --frame_interval 480 --max_frames 2880 --dark
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_fast_dark.mp4)
![docs/event_fast_dark.jpg](docs/event_fast_dark.jpg)

### Frame only, fast rotation, dark theme

```bash
python gen_video.py --average_frames 20 --speed 32 --rotation_duration 36 --rotation_speed 2 --stop_duration 0 --frame_interval 480 --max_frames 2880 --hide_event --show_frame --dark
```

[[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/frame_fast_dark.mp4)
![docs/frame_fast_dark.jpg](docs/frame_fast_dark.jpg)

### Scene Video

| Default | Fast |
|---|---|
| [[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_src.mp4) | [[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_fast_src.mp4) |
| ![docs/event_src.jpg](docs/event_src.jpg) | ![docs/event_fast_src.jpg](docs/event_fast_src.jpg) |

| Default (dark) | Fast (dark) |
|---|---|
| [[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_dark_src.mp4) | [[Download video]](https://github.com/elerac/event_camera_explanation_video/raw/refs/heads/main/videos/event_fast_dark_src.mp4) |
| ![docs/event_dark_src.jpg](docs/event_dark_src.jpg) | ![docs/event_fast_dark_src.jpg](docs/event_fast_dark_src.jpg) |
