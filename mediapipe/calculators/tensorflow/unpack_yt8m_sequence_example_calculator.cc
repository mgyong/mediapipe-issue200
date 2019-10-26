// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iterator>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {
namespace {

const char kId[] = "id";
const char kRgb[] = "rgb";
const char kAudio[] = "audio";
const char kYt8mId[] = "YT8M_ID";
const char kYt8mSequenceExample[] = "YT8M_SEQUENCE_EXAMPLE";
const char kQuantizedRgbFeature[] = "QUANTIZED_RGB_FEATURE";
const char kQuantizedAudioFeature[] = "QUANTIZED_AUDIO_FEATURE";

std::string GetQuantizedFeature(
    const tensorflow::SequenceExample& sequence_example, const std::string& key,
    int index) {
  const auto& bytes_list = sequence_example.feature_lists()
                               .feature_list()
                               .at(key)
                               .feature()
                               .Get(index)
                               .bytes_list()
                               .value();
  CHECK_EQ(1, bytes_list.size());
  return bytes_list.Get(0);
}
}  // namespace

// Unpacks YT8M Sequence Example. Note that the audio feature and rgb feature
// output are quantized. DequantizeByteArrayCalculator can do the dequantization
// for you.
//
// Example config:
// node {
//   calculator: "UnpackYt8mSequenceExampleCalculator"
//   input_side_packet: "YT8M_SEQUENCE_EXAMPLE:yt8m_sequence_example"
//   output_stream: "QUANTIZED_RGB_FEATURE:quantized_rgb_feature"
//   output_stream: "QUANTIZED_AUDIO_FEATURE:quantized_audio_feature"
// }
class UnpackYt8mSequenceExampleCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets()
        .Tag(kYt8mSequenceExample)
        .Set<tensorflow::SequenceExample>();
    cc->Outputs().Tag(kQuantizedRgbFeature).Set<std::string>();
    cc->Outputs().Tag(kQuantizedAudioFeature).Set<std::string>();
    if (cc->OutputSidePackets().HasTag(kYt8mId)) {
      cc->OutputSidePackets().Tag(kYt8mId).Set<std::string>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    const tensorflow::SequenceExample& sequence_example =
        cc->InputSidePackets()
            .Tag(kYt8mSequenceExample)
            .Get<tensorflow::SequenceExample>();
    const std::string& yt8m_id =
        sequence_example.context().feature().at(kId).bytes_list().value().Get(
            0);
    if (cc->OutputSidePackets().HasTag(kYt8mId)) {
      cc->OutputSidePackets().Tag(kYt8mId).Set(
          MakePacket<std::string>(yt8m_id));
    }

    int rgb_feature_list_length =
        sequence_example.feature_lists().feature_list().at(kRgb).feature_size();
    int audio_feature_list_length = sequence_example.feature_lists()
                                        .feature_list()
                                        .at(kAudio)
                                        .feature_size();

    if (rgb_feature_list_length != audio_feature_list_length) {
      return ::mediapipe::FailedPreconditionError(absl::StrCat(
          "Data corruption: the length of audio features and rgb features are "
          "not equal. Please check the sequence example that contains yt8m "
          "id: ",
          yt8m_id));
    }
    feature_list_length_ = rgb_feature_list_length;
    LOG(INFO) << "Reading the sequence example that contains yt8m id: "
              << yt8m_id << ". Feature list length: " << feature_list_length_;
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (current_index_ >= feature_list_length_) {
      return ::mediapipe::tool::StatusStop();
    }
    const tensorflow::SequenceExample& sequence_example =
        cc->InputSidePackets()
            .Tag(kYt8mSequenceExample)
            .Get<tensorflow::SequenceExample>();

    // Uses microsecond as the unit of time. In the YT8M dataset, each feature
    // represents a second.
    const Timestamp timestamp = Timestamp(current_index_ * 1000000);
    cc->Outputs()
        .Tag(kQuantizedRgbFeature)
        .AddPacket(
            MakePacket<std::string>(
                GetQuantizedFeature(sequence_example, kRgb, current_index_))
                .At(timestamp));
    cc->Outputs()
        .Tag(kQuantizedAudioFeature)
        .AddPacket(
            MakePacket<std::string>(
                GetQuantizedFeature(sequence_example, kAudio, current_index_))
                .At(timestamp));
    ++current_index_;
    return ::mediapipe::OkStatus();
  }

 private:
  int current_index_ = 0;
  int feature_list_length_ = 0;
};

REGISTER_CALCULATOR(UnpackYt8mSequenceExampleCalculator);

}  // namespace mediapipe
