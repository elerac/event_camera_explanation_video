import argparse
import datetime
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import pyvista as pv


def draw_rot_circle(theta: float, shape=(512, 512), r_large=0.9, r_small=0.025, r_target=0.1, r_target_orbit=0.65, thickness=3, background_color=(255, 255, 255), line_color=(0, 0, 0)):
    """Generate a rotating circle for event camera explanation.

    This image is mimiced from the following video:
    - Event-based, 6-DOF Pose Tracking for High-Speed Maneuvers using a Dynamic Vision Sensor
        - https://www.youtube.com/watch?v=LauQ6LWTkxM&t=35s
    """
    h, w = shape
    img = np.full((h, w, 3), background_color, dtype=np.uint8)
    center = np.array([w // 2, h // 2], dtype=np.int32)
    img = cv2.circle(img, center, int(min(h, w) * 0.5 * r_large), (210, 210, 210), -1, cv2.LINE_AA)
    img = cv2.circle(img, center, int(min(h, w) * 0.5 * r_large), line_color, thickness, cv2.LINE_AA)
    img = cv2.circle(img, center, int(min(h, w) * 0.5 * r_small), (0, 0, 0), -1, cv2.LINE_AA)
    center_target = (np.array([np.cos(theta), np.sin(theta)]) * min(h, w) * 0.5 * r_target_orbit + center).astype(np.int32)
    img = cv2.circle(img, center_target, int(min(h, w) * 0.5 * r_target), (0, 0, 0), -1, cv2.LINE_AA)

    # img = cv2.putText(img, "Event camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def rotate(p, rvec, p_center):
    p = np.array(p)
    rvec = np.array(rvec)
    p_center = np.array(p_center)
    if np.linalg.norm(rvec) < 1e-6:
        return p
    rmat, _ = cv2.Rodrigues(rvec)
    p = p - p_center
    p = np.dot(rmat, p)
    p = p + p_center
    return p


class EventCameraAnimation:
    def __init__(self, theme=None):
        # User-defined parameters
        self.camera = None  # pv.Camera()
        self.max_frames = 360  # Maximum number of frames
        self.t_box_ratio = 2.0  # Time window ratio to the maximum of the image size
        self.box_line_width = 2
        self.pl = pv.Plotter(off_screen=True, window_size=(1920, 1080), theme=theme)
        self.show_axes = False

        # Event data
        self.show_events = True
        self.reduction_ratio = 0.5  # Reduction ratio of the number of events (save memory and increase rendering speed)
        self.point_size = 0.5  # Point size of the events
        self.C = 0.25  # Contrast threshold
        self.color_pos = self.pl.theme.axes.x_color.int_rgb  # Color of the positive events (RGB)
        self.color_neg = self.pl.theme.axes.z_color.int_rgb  # Color of the negative events (RGB)

        # Intensity frame
        self.show_frames = False
        self.frame_interval = 120  # Interval of the frames to show
        self.frame_opacity = 0.75  # Opacity of the frames
        self.average_frames = 1  # Average number of frames (represent the blurred image)

        # Private variables
        self.i = 0
        self.x = []
        self.y = []
        self.t = []
        self.p = []
        self.frames = []
        self.frames_t = []
        self.frame_averaged = None
        self.img_log_ref = None

    def render(self, img) -> np.ndarray:
        self.pl.clear()
        eps = 1e-8

        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        h, w = img.shape[:2]
        t_max = self.t_box_ratio * max(h, w)
        t_delta = t_max / self.max_frames
        # Plot the event data
        if self.show_events:
            # Generate events from image sequence
            if self.img_log_ref is None:
                self.img_log_ref = np.log(img_gray.astype(np.float32) + eps)

            img_log = np.log(img_gray.astype(np.float32) + eps)
            img_log_diff = img_log - self.img_log_ref

            img_pos = img_log_diff >= self.C
            y_pos, x_pos = np.where(img_pos)
            index_choice = np.random.choice(len(x_pos), int(len(x_pos) * self.reduction_ratio))
            x_pos, y_pos = x_pos[index_choice], y_pos[index_choice]
            y_pos = (img.shape[0] - 1) - y_pos
            self.x.append(x_pos)
            self.y.append(y_pos)
            self.t.append(np.random.uniform(0, t_delta, len(x_pos)))
            self.p.append(np.full_like(x_pos, 1))

            img_neg = img_log_diff <= -self.C
            y_neg, x_neg = np.where(img_neg)
            index_choice = np.random.choice(len(x_neg), int(len(x_neg) * self.reduction_ratio))
            x_neg, y_neg = x_neg[index_choice], y_neg[index_choice]
            y_neg = (img.shape[0] - 1) - y_neg
            self.x.append(x_neg)
            self.y.append(y_neg)
            self.t.append(np.random.uniform(0, t_delta, len(x_neg)))
            self.p.append(np.full_like(x_neg, 0))

            # Update the reference image
            img_pos_neg = np.bitwise_or(img_pos, img_neg)
            self.img_log_ref[img_pos_neg] = img_log[img_pos_neg]

            # Concatenate all events
            x = np.concatenate(self.x)
            y = np.concatenate(self.y)
            t = np.concatenate(self.t)
            p = np.concatenate(self.p)

            # Delete old events that beyond the time window
            idx = t < t_max
            x, y, t, p = x[idx], y[idx], t[idx], p[idx]

            xyt = np.column_stack((x, y, t)).astype(np.float32)
            event_color = np.array([self.color_pos, self.color_neg], dtype=np.uint8)
            cloud = pv.PolyData(xyt)
            cloud["colors"] = event_color[p]
            if len(p) > 0:
                self.pl.add_points(cloud, point_size=self.point_size, rgb=True, style="points_gaussian")

        # Show frames
        if self.show_frames:
            if self.average_frames < 1:
                raise ValueError("The average_frames must be greater than or equal to 1.")

            if self.i % self.frame_interval == 0:
                self.frame_averaged = img / self.average_frames
            elif self.i % self.frame_interval < self.average_frames:
                self.frame_averaged += img / self.average_frames

            if self.i % self.frame_interval == self.average_frames - 1:
                self.frames_t.append(0)
                self.frames.append(self.frame_averaged.astype(np.uint8))

            if len(self.frames) > 0:
                for frame_t, frame in zip(self.frames_t, self.frames):
                    texture = pv.numpy_to_texture(frame)
                    plane = pv.Plane(center=(w / 2, h / 2, frame_t), direction=(0, 0, 1), i_size=w, j_size=h)
                    self.pl.add_mesh(plane, texture=texture, lighting=False, opacity=self.frame_opacity)
                    square = pv.Box([0, w, 0, h, frame_t, frame_t])
                    self.pl.add_mesh(square, color=self.pl.theme.edge_color, style="wireframe", line_width=self.box_line_width)

        # Plot the image
        texture = pv.numpy_to_texture(img)
        plane = pv.Plane(center=(w / 2, h / 2, 0), direction=(0, 0, 1), i_size=w, j_size=h)
        self.pl.add_mesh(plane, texture=texture, lighting=False)

        # Plot the bounding box
        box = pv.Box([0, w, 0, h, 0, t_max])
        self.pl.add_mesh(box, color=self.pl.theme.edge_color, style="wireframe", line_width=self.box_line_width)
        if self.show_axes:
            self.pl.show_axes()

        # box_center = box.center
        if self.camera is not None:
            self.pl.camera = self.camera

        img_rendered = self.pl.screenshot(return_img=True)

        # Updat list for the next frame
        if self.show_events:
            self.x = [x]
            self.y = [y]
            self.t = [t + t_delta]
            self.p = [p]

        if self.show_frames:
            self.frames_t = [frame_t + t_delta for frame_t in self.frames_t]
            # Remove out-of-range frames
            self.frames = [frame for frame, frame_t in zip(self.frames, self.frames_t) if frame_t < t_max]
            self.frames_t = [frame_t for frame_t in self.frames_t if frame_t < t_max]

        self.i += 1

        return img_rendered


def main():
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    # Input video parameters
    parser.add_argument("-i", "--input", type=str, default=None)

    parser.add_argument("--rotation_speed", type=int, default=1, help="Rotation speed of the circle")
    parser.add_argument("--rotation_duration", type=int, default=5, help="Duration of the rotation")
    parser.add_argument("--stop_duration", type=int, default=4, help="Duration of the stop")

    # Output video parameters
    parser.add_argument("-o", "--output", type=str, default=f"event_camera_animation_{date}.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=int, default=8, help="Speed of the exported video x(1, 2, 3, ...)")

    # pyvista parameters
    parser.add_argument("--width", type=int, default=1920, help="Width of the plotter")
    parser.add_argument("--height", type=int, default=1080, help="Height of the plotter")
    parser.add_argument("--dark", action="store_true", help="Use the dark theme")
    parser.add_argument("--max_frames", type=int, default=720, help="Maximum number of frames to show")
    parser.add_argument("--t_box_ratio", type=float, default=3, help="Time window ratio to the maximum of the image size")
    parser.add_argument("--box_line_width", type=int, default=2, help="Line width of the bounding box")
    parser.add_argument("--camera", type=int, default=0, help="Camera preset index")

    # Event plot parameters
    parser.add_argument("--hide_events", action="store_false", help="Hide events")
    parser.add_argument("--reduction_ratio", type=float, default=0.5, help="Reduction ratio of the number of events")
    parser.add_argument("--point_size", type=float, default=0.5, help="Point size of the events")
    parser.add_argument("--C", type=float, default=0.25, help="Contrast threshold")
    parser.add_argument("--color_pos", type=int, nargs=3, default=[255, 99, 71], help="Color of the positive events (RGB)")
    parser.add_argument("--color_neg", type=int, nargs=3, default=[0, 0, 255], help="Color of the negative events (RGB)")

    # Frame plot parameters
    parser.add_argument("--show_frames", action="store_true", help="Show frames")
    parser.add_argument("--frame_interval", type=int, default=120, help="Interval of the frames to show")
    parser.add_argument("--frame_opacity", type=float, default=0.8, help="Opacity of the frames")
    parser.add_argument("--average_frames", type=int, default=1, help="Average number of frames (represent the blurred image)")

    args = parser.parse_args()

    if args.input is not None:
        print(f"Load the video from {args.input}")
        cap = cv2.VideoCapture(args.input)
        video_src = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_src.append(frame)
    else:
        print("Generate a rotating circle video")
        print(f"Rotation speed: {args.rotation_speed}")
        print(f"Rotation duration: {args.rotation_duration}")
        print(f"Stop duration: {args.stop_duration}")
        theta_array = np.deg2rad(np.concatenate([np.arange(0, 360 * args.rotation_duration * args.rotation_speed, args.rotation_speed), np.full(360 * args.stop_duration, 0)]))
        video_src = [draw_rot_circle(theta) for theta in theta_array]

        if args.dark:
            video_src = [255 - img for img in video_src]

    theme = pv.themes.DarkTheme() if args.dark else None
    animation = EventCameraAnimation(theme)
    animation.pl.window_size = (args.width, args.height)
    animation.max_frames = args.max_frames
    animation.box_line_width = args.box_line_width
    animation.t_box_ratio = args.t_box_ratio
    animation.show_events = args.hide_events
    animation.reduction_ratio = args.reduction_ratio
    animation.point_size = args.point_size
    animation.C = args.C
    animation.color_pos = args.color_pos
    animation.color_neg = args.color_neg
    animation.show_frames = args.show_frames
    animation.frame_interval = args.frame_interval
    animation.frame_opacity = args.frame_opacity
    animation.average_frames = args.average_frames

    print("Parameters:")
    print("----------------------")
    print("pyvista parameters")
    print(f"  Width: {args.width}")
    print(f"  Height: {args.height}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Box line width: {args.box_line_width}")
    print(f"  Dark theme: {args.dark}")
    print(f"  Camera preset: {args.camera}")
    print("Event plot parameters")
    print(f"  Show events: {args.hide_events}")
    print(f"  Reduction ratio: {args.reduction_ratio}")
    print(f"  Point size: {args.point_size}")
    print(f"  Contrast threshold: {args.C}")
    print(f"  Color of the positive events: {args.color_pos}")
    print(f"  Color of the negative events: {args.color_neg}")
    print("Frame plot parameters")
    print(f"  Show frames: {args.show_frames}")
    print(f"  Frame interval: {args.frame_interval}")
    print(f"  Frame opacity: {args.frame_opacity}")
    print(f"  Average frames: {args.average_frames}")
    print("----------------------")

    H, W = video_src[0].shape[:2]
    T = animation.t_box_ratio * max(H, W)

    camera_presets = []

    # Camera preset 0
    camera = pv.Camera()
    camera.view_angle = 1
    camera.position = np.array([0, 0, T]) / np.arctan(np.deg2rad(camera.view_angle)) * 0.62
    box_center = np.array([W / 2, H / 2, T / 2])
    box_center = box_center + np.array([0, -H * 0.07, 0])
    camera.position = rotate(camera.position, np.deg2rad(-75) * np.array([0, 1, 0]), box_center)
    camera.position = rotate(camera.position, np.deg2rad(-15) * np.array([0, 0, 1]), box_center)
    camera.focal_point = box_center
    camera.up = [0, 1, 0]
    animation.camera = camera
    camera_presets.append(camera)

    # Camera preset 1
    camera = pv.Camera()
    camera.view_angle = 20
    camera.position = np.array([W / 2, H / 2, T * 2 / np.arctan(np.deg2rad(camera.view_angle)) * 0.35])
    camera.focal_point = np.array([W / 2, H / 2, 0])
    camera.up = np.array([0, 1, 0])
    camera_presets.append(camera)

    # Camera preset 2
    camera = pv.Camera()
    camera.view_angle = 5
    camera.position = np.array([W * 3.5, H * 3.5, T * 1.36]) / np.arctan(np.deg2rad(camera.view_angle)) * 0.54
    camera.focal_point = np.array([W / 2, H / 2, T * 0.5])
    camera.up = np.array([0, 0, 1])
    camera_presets.append(camera)

    animation.camera = camera_presets[args.camera]

    # Render the video
    imlist_rendered = []
    for i, img in enumerate(tqdm(video_src)):
        img_rendered = animation.render(img)
        imlist_rendered.append(img_rendered)

    # Save the video
    imlist_rendered = imlist_rendered[:: args.speed]
    height, width = imlist_rendered[0].shape[:2]
    filename = args.output
    print(f"Save the rendered video to {filename}")
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    for img in imlist_rendered:
        writer.write(img)
    writer.release()

    imlist_src_speed = video_src[:: args.speed]
    height, width = imlist_src_speed[0].shape[:2]
    filename = str(Path(args.output).stem + "_src.mp4")
    print(f"Save the source video to {filename}")
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    for img in imlist_src_speed:
        writer.write(img)
    writer.release()


if __name__ == "__main__":
    main()
