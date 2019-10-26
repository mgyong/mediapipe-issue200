# mediapipe-issue200
[MediaPipe](https://mediapipe.dev) example of how to readout Detection proto for [Object Detection example](https://github.com/google/mediapipe/blob/master/mediapipe/docs/object_detection_desktop.md)

Related to MediaPipe [github issue 200](https://github.com/google/mediapipe/issues/200)

2 file changes are reflected at
https://gist.github.com/mgyong/7353474eb3e57ba95621632af274911a
https://gist.github.com/mgyong/be027a075a5e7b082c566a78c3bc0d90

To build this 
```bash
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_out_cpu 
```

To run object detection example
```bash
bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_out_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt


INFO: Initialized TensorFlow Lite runtime.
label: "person"
score: 0.730482936
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.215880126
    ymin: 0.346766353
    width: 0.617606759
    height: 0.647981644
  }
}
```
