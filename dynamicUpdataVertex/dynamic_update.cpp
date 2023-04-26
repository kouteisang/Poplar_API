#include <iostream>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include <popops/codelets.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;

// Shape of the tensor for this example.
constexpr std::size_t m = 300;
constexpr std::size_t n = 300;

void customDynamicUpdate(poplar::Graph &graph, const poplar::Tensor &t,
                         const poplar::Tensor &offset,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {}) {
  // Assumes we have a 2D tensor (a matrix)
  assert(t.rank() == 2);

  // Assumes offset is a 2-vector of coordinates `i`,`j`.
  assert(offset.rank() == 1);
  assert(offset.dim(0) == 2);

  // Work out how to distribute the vertices across the tiles.
  const auto tile_count = graph.getTarget().getTilesPerIPU();
  const auto row_count = t.dim(0);
  const auto rows_per_tile =
      (row_count + tile_count - 1) / tile_count; // ceil(row_count / tile_count)

  // Create the compute set.
  auto compute_set = graph.addComputeSet(debugContext);

  // For each tile, add a vertex and connect the corresponding parts of the
  // tensor.
  for (int tile = 0; tile < row_count / rows_per_tile; ++tile) {
    // We assume the tensor `t` is partitioned across tiles in contiguous rows.
    auto vertex = graph.addVertex(compute_set, "CustomDynamicUpdateScalar");
    graph.setTileMapping(vertex, tile);

    // Connect the corresponding rows of the input tensor.
    graph.connect(vertex["slices"],
                  t.slice(tile * rows_per_tile, (tile + 1) * rows_per_tile, 0));

    // Connect the indices.
    graph.connect(vertex["indices"], offset);

    // Set the vertex state so that the local row offset is known.
    graph.setInitialValue<int>(vertex["local_row"], tile * rows_per_tile);
  }

  // Execute the compute set.
  prog.add(poplar::program::Execute(compute_set));
}

int main() {
  // Get a device ect.
  DeviceManager manager = DeviceManager::createDeviceManager();
  Device device;
  bool success = false;
  for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, 1)) {
    device = std::move(hwDevice);
    std::cerr << "Trying to attach to IPU " << device.getId() << std::endl;
    if ((success = device.attach())) {
      std::cerr << "Attached to IPU " << device.getId() << std::endl;
      break;
    }
  }
  if (!success) {
    std::cerr << "Error attaching to device" << std::endl;
    return -1;
  }
  Target target = device.getTarget();

  // Create the graph and add the poplibs codelets.
  Graph graph(target);
  graph.addCodelets("vertex.cpp");

  // Create the input tensor and map it linearly with a grain size equal to the
  // size of a row.
  poplar::Tensor tensor = graph.addVariable(poplar::FLOAT, {m, n});
  poputil::mapTensorLinearly(graph, tensor, 0, /* grainSize= */ n);

  // Register the host write so we can initialise `tensor`.
  graph.createHostWrite("tensor_write", tensor, true);

  // This can be a variable tensor, just using a constant for illustration
  // purposes.
  poplar::Tensor indices = graph.addConstant<int>(
      poplar::INT, // The element type of the indices.
      {2},         // Shape of the indices.
      {12, 13}     // Values to store in the indices, so we will lookup
                   // the element with coordinates i=12, j=13.
  );
  // Place the indices on tile zero.
  graph.setTileMapping(indices, 0);

  // Create the poplar sequence program.
  poplar::program::Sequence prog;
  customDynamicUpdate(graph, tensor, indices, prog);

  // Print the updated tensor.
  // We should see the value on row 12 column 13 increased by 1.
  prog.add(poplar::program::PrintTensor("updated tensor", tensor));

  // Compile the program.
  Engine engine(graph, prog);
  engine.load(device);

  // Initialise the tensor.
  std::vector<float> initial_values;
  initial_values.resize(m * n);
  std::iota(initial_values.begin(), initial_values.end(), 0.0f);
  engine.writeTensor("tensor_write", initial_values.data(),
                     initial_values.data() + (m * n));

  // Run the program.
  engine.run(0);

  return 0;
}
