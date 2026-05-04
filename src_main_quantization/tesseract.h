// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TESSERACT_DECODER_H
#define TESSERACT_DECODER_H

#include <boost/dynamic_bitset.hpp>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"
#include "utils.h"
#include "visualization.h"

constexpr size_t INF_DET_BEAM = std::numeric_limits<uint16_t>::max();
constexpr int DEFAULT_DET_BEAM = 5;
constexpr size_t DEFAULT_PQLIMIT = 200000;

struct TesseractConfig {
  stim::DetectorErrorModel dem;
  int det_beam = DEFAULT_DET_BEAM;
  bool beam_climbing = false;
  bool no_revisit_dets = true;

  bool verbose = false;
  bool merge_errors = true;
  size_t pqlimit = DEFAULT_PQLIMIT;
  std::vector<std::vector<size_t>> det_orders;
  double det_penalty = 0;
  bool create_visualization = false;

  // Quantization options
  int quant_bits = 0;           // 0 => desactivado, >0 aplica cuantización uniforme
  double quant_max_abs = 20.0;  // Rango simétrico [-quant_max_abs, +quant_max_abs]
  bool quant_auto_max = false;  // Si true, usa el |log-odds| máximo observado del DEM

  std::string str();
};

class Node {
 public:
  AccumInt cost; // int32_t — accumulated sum of CostInt error costs
  size_t num_dets;
  size_t depth;
  int64_t error_chain_idx = -1;

  bool operator>(const Node& other) const;
  std::string str();
};

struct DetectorCostTuple {
  uint32_t error_blocked;
  uint32_t detectors_count;
};

struct ErrorCost {
  CostInt likelihood_cost;
  CostInt min_cost;
};

struct TesseractDecoder {
  TesseractConfig config;
  Visualizer visualizer;

  explicit TesseractDecoder(TesseractConfig config);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events.
  void decode_to_errors(const std::vector<uint64_t>& detections);

  // Clears the predicted_errors_buffer and fills it with the decoded errors for
  // these detection events, using a specified detector ordering index.
  void decode_to_errors(const std::vector<uint64_t>& detections, size_t detector_order,
                        size_t detector_beam);

  // Returns the bitwise XOR of the observables flipped by the errors in the given array, indexed by
  // the original flattened DEM error indices.
  std::vector<int> get_flipped_observables(const std::vector<size_t>& predicted_errors) const;

  // Returns the sum of likelihood costs of the errors in the given array
  AccumInt cost_from_errors(const std::vector<size_t>& predicted_errors) const;

  std::vector<int> decode(const std::vector<uint64_t>& detections);
  void decode_shots(std::vector<stim::SparseShot>& shots,
                    std::vector<std::vector<int>>& obs_predicted);

  bool low_confidence_flag = false;
  std::vector<size_t> predicted_errors_buffer;
  std::vector<size_t> dem_error_to_error;
  std::vector<size_t> error_to_dem_error;
  std::vector<common::Error> errors;
  size_t num_observables;
  size_t num_detectors;

  std::vector<std::vector<int>>& get_eneighbors() {
    return eneighbors;
  }

 private:
  std::vector<std::vector<int>> d2e;
  std::vector<std::vector<int>> eneighbors;
  std::vector<std::vector<int>> edets;
  size_t num_errors;
  std::vector<ErrorCost> error_costs;
  std::vector<common::ErrorChainNode> error_chain_arena;

  double cost_scale = 1.0;        // scale: CostInt = double_cost * cost_scale
  AccumInt det_penalty_int = 0;   // config.det_penalty converted to integer units  

  void initialize_structures(size_t num_detectors);
  AccumInt get_detcost(size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const;
  void flip_detectors_and_block_errors(size_t detector_order, int64_t error_chain_idx,
                                       boost::dynamic_bitset<>& detectors,
                                       std::vector<DetectorCostTuple>& detector_cost_tuples) const;
};

#endif  // TESSERACT_DECODER_H
