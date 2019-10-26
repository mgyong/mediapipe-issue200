### Steps to run the YouTube-8M feature extraction graph

1.  Checkout the mediapipe repository

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    ```

2.  Download the PCA and model data

    ```bash
    mkdir /tmp/mediapipe
    cd /tmp/mediapipe
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
    curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz
    ```

3.  Get the VGGish frozen graph

    Note: To run step 3 and step 4, you must have Python 2.7 or 3.5+ installed
    with the TensorFlow 1.14+ package installed.

    ```bash
    # cd to the root directory of the MediaPipe repo
    cd -
    python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
    ```

4.  Generate a MediaSequence metadata from the input video

    Note: the output file is /tmp/mediapipe/metadata.tfrecord

    ```bash
    python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
      --path_to_input_video=/absolute/path/to/the/local/video/file \
      --clip_start_time_sec=0 \
      --clip_end_time_sec=120
    ```

5.  Run the MediaPipe binary to extract the features

    ```bash
    bazel build -c opt \
      --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
      mediapipe/examples/desktop/youtube8m:extract_yt8m_features

    ./bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
      --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
      --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.tfrecord  \
      --output_side_packets=output_sequence_example=/tmp/mediapipe/output.tfrecord
    ```

### Steps to run the YouTube-8M segement level inference graph with a local video

1.  Make sure you have the output tfrecord from the feature extraction pipeline

2.  Copy the baseline model to local

3.  Build and run the inference binary

  ```
  bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS \
    mediapipe/examples/desktop/youtube8m:segment_level_inference

  bazel-bin/mediapipe/examples/desktop/youtube8m/segment_level_inference \
    --calculator_graph_config_file=mediapipe/graphs/youtube8m/local_video_segment_level_inference.pbtxt \
    --input_side_packets=input_sequence_example_path=/tmp/mediapipe/output.tfrecord,input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/tmp/mediapipe/annotated_video.mp4,segment_size=5,overlap=4
  ```

4.  View the annotated video

### Steps to run the YouTube-8M segement level inference graph with the YT8M dataset

1.  Download the YT8M dataset

  For example, download 1/100-th of the training data use:

  ```
  curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python
  ```

2.  Copy the baseline model to local

3.  Build and run the inference binary

  ```
  bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS \
    mediapipe/examples/desktop/youtube8m:segment_level_inference

  bazel-bin/mediapipe/examples/desktop/youtube8m/segment_level_inference \
    --calculator_graph_config_file=mediapipe/graphs/youtube8m/dataset_segment_level_inference.pbtxt \
    --input_side_packets=tfrecord_path=/tmp/train0093.tfrecord,record_index_str=0 \
    --output_stream=annotation_summary \
    --output_stream_file=/tmp/summary \
    --output_side_packets=yt8m_id \
    --output_side_packets_file=/tmp/yt8m_id
  ```
