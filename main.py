# !/usr/bin/env python
import cv2
import os
import queue
import threading
import traceback
import time
from edgetpu_detection.edgetpu import TFliteDetection

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = "192.168.42.1"
CWD_PATH = os.getcwd()

# Re-trained model
MODEL_DIR = "edgetpu_detection/models"
PATH_TO_MODEL = os.path.join(CWD_PATH, MODEL_DIR, 'ssd_mobilenet_v2_parrot_v1_quant_edgetpu.tflite')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, 'labels.txt')


class Detection(threading.Thread):

    def __init__(self, processed_queue, signal):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        self.frame_queue = queue.Queue()
        self.signal_queue = signal
        self.processed_frame_queue = processed_queue
        self.flush_queue_lock = threading.Lock()
        self.detector = TFliteDetection(PATH_TO_MODEL, PATH_TO_LABELS, count=1)
        super().__init__()
        super().start()

    def start(self):
        # Connect the the drone
        self.drone.connect()

        # Setup callback functions to do some live video processing
        # Yuv frames are passed into set_streaming_callbacks somehow and then
        # we collect the frames using the yuv_frame_cb function
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.start_video_streaming()  # equivalent to pdraw.play

    def stop(self):
        # Properly stop the video stream and disconnect
        self.drone.stop_video_streaming()
        self.drone.disconnect()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.
            :type yuv_frame: olympe.VideoFrame

        Each video frame callback function takes an :py:func:`~olympe.VideoFrame` parameter whose
        lifetime ends after the callback execution. If this video frame is passed to another thread,
        its internal reference count need to be incremented first by calling
        :py:func:`~olympe.VideoFrame.ref`. In this case, once the frame is no longer needed, its
        reference count needs to be decremented so that this video frame can be returned to
        memory pool.
        """
        # Essentially accessing the yuv_frame from this call_back function and then
        # passes it into a queue to be picked up outside the callback function e.g. (object detection function)

        yuv_frame.ref()  # increments the reference counter of the underlying buffer(s) - need to understand

        self.frame_queue.put_nowait(yuv_frame)  # put the yuv_frames into a queue data structure

    def flush_cb(self):
        """
        Video flush callback functions are called when a video stream reclaim all its associated video buffer.
        Every frame that has been referenced
        """
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def convert_yuv_frame(self, yuv_frame):

        # the VideoFrame.info() dictionary contains some useful information
        # such as the video resolution
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        # yuv_frame.vmeta() returns a dictionary that contains additional
        # metadata from the drone (GPS coordinates, battery percentage, ...)

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
        # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

        # Use OpenCV to convert the yuv frame to RGB
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
        return cv2frame

    def terminate(self):
        try:
            signal = self.signal_queue.get(timeout=0.01)
            return signal
        except queue.Empty:
            return None

    def object_detection(self):
        print('entered object detecting')
        while True:
            with self.flush_queue_lock:
                try:
                    yuv_frame = self.frame_queue.get(timeout=0.01)
                    rgb_frame = self.convert_yuv_frame(yuv_frame=yuv_frame)

                    # perform object detection
                    rgb_frame, state = self.detector.detect(rgb_frame)

                    cv2.putText(rgb_frame, "Object Detection frame", (100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                except queue.Empty:
                    if self.frame_queue.empty():
                        continue

                try:
                    self.processed_frame_queue.put_nowait((rgb_frame, state))
                    if self.terminate():
                        break

                except Exception:
                    # We have to continue popping frame from the queue even if
                    # we fail to show one frame
                    traceback.print_exc()

                finally:
                    # Don't forget to unref the yuv frame. We don't want to
                    # starve the video buffer pool
                    yuv_frame.unref()

    # TODO -
    #  Option 1: Change object tracking function to threading.run()
    #  such that object tracking will only be active for duration of flight
    #  will need destroy all windows within the fly function, if not destroyed by keyboard
    #  Option 2: Put this function as another thread

    def fly(self):
        time.sleep(20)

        pass


class VideoManager(object):
    """
    This class handles the displaying of videoframes from the drone post object detection
    """
    def __init__(self, processed_queue, signal):
        self.processed_queue = processed_queue
        thread = threading.Thread(target=self.show, args=(self.processed_queue,))
        self.lock = threading.Lock()
        self.signal_queue = signal
        thread.daemon = True      # when main program quits, any daemon threads are killed automatically)
        thread.start()            # Start the execution
        self._sentinel = False

    def show(self, q):
        """
         Method to show the frames from the drone camera
        """
        while True:
            with self.lock:
                if not q.empty():
                    try:
                        # get the processed frame queue from object detection function in Detection class
                        proc_img, state = q.get_nowait()
                        exit_condition = self.show_yuv_frame(proc_img, state)
                        if exit_condition:
                            self._sentinel = True  # send signal to break the object detection thread
                            print('breaking object detecting thread')
                            break
                    except Exception:
                        continue
                        # We have to continue popping frame from the queue even if
                        # we fail to show one frame
                        # traceback.print_exc()
                else:
                    # print('global queue is empty')
                    pass

        # indicate external completion signal from user (i.e keyboard termination)
        self.signal_queue.put_nowait(self._sentinel)

    def show_yuv_frame(self, rgb_frame, state):
        window_name = "Olympe Streaming Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 2500, 1500)
        # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Display tracker type on frame
        if state:
            cv2.putText(rgb_frame,"Detection Successful",
                        (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        else:
            cv2.putText(rgb_frame, "Detection Failed",
                        (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Use OpenCV to show this frame
        cv2.imshow(window_name, rgb_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyWindow(window_name)
            return True
        else:
            return False


if __name__ == "__main__":

    proc_queue = queue.Queue()

    shutdown_signal = queue.Queue(maxsize=5)

    t2 = VideoManager(proc_queue, shutdown_signal)

    # Entire thread is all the functions below so thread will be kept alive until the entire thread completes
    detection = Detection(proc_queue, shutdown_signal)

    # Start the video stream
    detection.start()

    # Perform some live video processing while the drone is flying
    # detection.fly()

    # comment this out if object_detection is threading.run() (inherited) and uncomment streaming_example.fly()
    detection.object_detection()

    # Stop the video stream
    detection.stop()

    # Recorded video stream postprocessing
    # detection.postprocessing()


