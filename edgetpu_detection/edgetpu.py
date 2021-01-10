import time
import numpy as np

from PIL import Image
from PIL import ImageDraw

import edgetpu_detection.detect as detect
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


class TFliteDetection:

    def __init__(self, model, labels, count=1, threshold=0.8):
        self.count = count
        self.threshold = threshold
        self.labels = labels
        self.model = model
        self.labels = self.load_labels(self.labels) if self.labels else {}
        self.interpreter = self.load_tflite_model(self.model)

    def load_labels(self, path, encoding='utf-8'):
        """Loads labels from file (with or without index numbers).

          Args:
            path: path to label file.
            encoding: label file encoding.
          Returns:
            Dictionary mapping indices to labels.
        """
        with open(path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}

    def make_interpreter(self, model_file):
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                     {'device': device[0]} if device else {})
            ])

    def load_tflite_model(self, model):
        interpreter = self.make_interpreter(model)
        interpreter.allocate_tensors()
        return interpreter

    def draw_objects(self, draw, objs, labels):
        """Draws the bounding box and label for each object."""
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline='red')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                      fill='red')

    def get_image(self, input_img, interpreter):
        image = input_img
        scale = detect.set_input(interpreter, image.size,
                                 lambda size: image.resize(size, Image.ANTIALIAS))
        return image, scale

    def detect(self, input_frame):
        if isinstance(input_frame, np.ndarray):
            input_frame = Image.fromarray(input_frame)  # convert np array to PIL Image
            image, scale = self.get_image(input_frame, self.interpreter)

        else:  # if it is a file
            input_frame = Image.open(input_frame)
            image, scale = self.get_image(input_frame, self.interpreter)

        for _ in range(self.count):
            start = time.perf_counter()
            self.interpreter.invoke()
            detected_objects = detect.get_output(self.interpreter, self.threshold, scale)

        if not detected_objects:
            state = False
        else:
            state = True

        image = image.convert('RGB')
        self.draw_objects(ImageDraw.Draw(image), detected_objects, self.labels)  # output needs to be numpy array
        np_img = np.array(image)
        return np_img, state

