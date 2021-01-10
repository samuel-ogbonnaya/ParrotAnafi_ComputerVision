# Parrot Anafi Autonomous Computer Vision Project
CV project utilising the Parrot Anafi drone for automatically detecting and tracking footballs to generate video footage.

- An Edge TPU MobileSSDV2 model was trained using google-coral. This model allows for very fast inference times (<20ms).
- Addtional models such as YOLO-4 and quantized YOLO are also being experimented with, although these have relatively slowere inference times but high detection performance.

## Requirements
- Software
  - Python >= 3.6
  - [Parrot Olympe SDK](https://developer.parrot.com/docs/olympe/overview.html)
  - Tensorflow lite
  - Open CV 4
- Hardware
  - [Google Coral Edge TPU USB Accelerator](https://coral.ai/products/accelerator/)
  - [Parrot Anafi](https://www.parrot.com/uk/drones/anafi)

## TODO
- [X] Access video from Anafi drone via SDK 
- [X] Add capability to always show video stream regardless of detection status
- [X] COllect data, train TFlite Detection model and evaluate performance
- [ ] Implement Deep Sort Tracking Class to be used in conjunction with the TFLite models
- [ ] Test additional detection models
- [ ] Implement and test drone controller class e.g (take-off, flight, landing etc) using Olympe
  
