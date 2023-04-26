#include <iostream>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include <popops/codelets.hpp>

#include <popops/ElementWise.hpp>
#include <popops/DynamicSlice.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;

// Shape of the tensor for this example.
constexpr std::size_t m = 1024;
constexpr std::size_t n = 1024;

poplar::Tensor dynamicSlice(
  poplar::Graph &graph,
  const poplar::Tensor &t,
  const poplar::Tensor &offset,
  std::vector<std::size_t> dims,
  std::vector<std::size_t> sizes,
  poplar::program::Sequence &prog,
  const poplar::DebugContext &debugContext = {},
  const poplar::OptionFlags &options = {}) {
  // Don't need to do any slicing
  if (dims.empty()) {
    return t;
  }

  // Create a temporary storage location in the "sliceable" layout.
  poplar::Tensor tmp = popops::createSliceableTensor(
    graph,
    t.elementType(),
    t.shape(),
    {dims.back()},
    {sizes.back()},
    0,
    debugContext
  );

  // Copy to the temporary location.
  prog.add(poplar::program::Copy(t, tmp));

  // Slice on the last offset.
  const auto indices_back = sizes.size() - 1;
  poplar::Tensor slice = popops::dynamicSlice(
    graph,
    tmp,
    offset.slice(indices_back, indices_back + 1, 0),
    {dims.back()},
    {sizes.back()},
    prog,
    debugContext,
    options
  );

  // Remove the sliced dimensions from our list.
  dims.pop_back();
  sizes.pop_back();

  // Recurse to do any remaining slices.
  return dynamicSlice(
    graph,
    slice,
    offset.slice(0, indices_back, 0),
    std::move(dims),
    std::move(sizes),
    prog,
    debugContext,
    options
  );
}

void dynamicUpdate(
  poplar::Graph &graph,
  const poplar::Tensor &t,
  const poplar::Tensor &s,
  const poplar::Tensor &offset,
  std::vector<std::size_t> dims,
  std::vector<std::size_t> sizes,
  poplar::program::Sequence &prog,
  const poplar::DebugContext &debugContext = {},
  const poplar::OptionFlags &options = {}) {
  // No dimensions means this is just a copy.
  if (dims.empty()) {
    prog.add(poplar::program::Copy(s, t));
    return;
  }

  // Create a temporary storage location in the "sliceable" layout.
  poplar::Tensor tmp = popops::createSliceableTensor(
    graph,
    t.elementType(),
    t.shape(),
    {dims.back()},
    {sizes.back()},
    0,
    debugContext
  );

  // Copy to the temporary location.
  prog.add(poplar::program::Copy(t, tmp));

  // Slice on the last offset.
  const auto indices_back = sizes.size() - 1;
  const auto dim = dims.back();
  const auto size = sizes.back();
  poplar::Tensor tmp_slice = popops::dynamicSlice(
    graph,
    tmp,
    offset.slice(indices_back, indices_back + 1, 0),
    {dim},
    {size},
    prog,
    debugContext,
    options
  );

  // Remove the sliced dimensions from our list.
  dims.pop_back();
  sizes.pop_back();

  // Recruse to update the slice with `s`.
  dynamicUpdate(
    graph,
    tmp_slice,
    s,
    offset.slice(0, indices_back, 0),
    dims,
    sizes,
    prog,
    debugContext,
    options
  );

  // Put the updated slice back into the temporary storage location.
  popops::dynamicUpdate(
    graph,
    tmp,
    tmp_slice,
    offset.slice(indices_back, indices_back + 1, 0),
    {dim},
    {size},
    prog,
    debugContext,
    options
  );

  // Copy from the temporary storage location back to the input tensor.
  prog.add(poplar::program::Copy(tmp, t));
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
  popops::addCodelets(graph);

  // Optionally create the tensor using `createSliceableTensor`. This tries to
  // distribute the tensor across the tiles to maximise efficiency.
  poplar::Tensor tensor = graph.addVariable(poplar::FLOAT, {m, n});
  poputil::mapTensorLinearly(graph, tensor);

  // Register the host write so we can initialise `tensor`.
  graph.createHostWrite("tensor_write", tensor, true);

  // This can be a variable tensor, just using a constant for illustration
  // purposes.
  poplar::Tensor indices = graph.addConstant<unsigned>(
    poplar::UNSIGNED_INT, // The element type of the indices. 
    {2},                  // Shape of the indices.
    {3, 7}                // Values to store in the indices, so we will lookup
                          // the element with coordinates i=12, j=13.
  );
  // Place the indices on tile zero.
  graph.setTileMapping(indices, 0);

  // Create the poplar sequence program.
  poplar::program::Sequence prog;

  // Slice the elements from tensor.
  // This is equivalent to `slice = tensor[indices[0]][indices[1]]`.
  poplar::Tensor slice = dynamicSlice(
    graph,   // The poplar graph to put the tensor.
    tensor,  // the input tensor being sliced.
    indices, // The indices of the slice.
    {0, 1},  // The dimensions the indices are slicing.
    {1, 1},  // The size of the slice.
    prog     // The poplar program to add this operation to.
  );

  // Create a tensor containing `1.0f` for updating.
  poplar::Tensor one = graph.addConstant<float>(
    poplar::FLOAT, // The constant element type.
    {},            // The shape of the constant, in this case a scalar.
    {1.0f}         // the value of the constant, in this case `1.0f`.
  );
  // Put the update tensor on tile 0.
  graph.setTileMapping(one, 0);

  // Update the slice.
  // This is equivalent to `slice += 1.0`.
  popops::addInPlace(
    graph, // The poplar graph to put the tensor.
    slice, // The in-out tensor to add to.
    one,   // The rhs of the add, in this case 1.0f.
    prog   // The poplar program to add this operation to.
  );

  // Put the updated slice back in the original tensor.
  // This is equivalent to `tensor[indices[0]][indices[1]] = slice`.
  dynamicUpdate(
    graph,   // The poplar graph to put the tensor.
    tensor,  // the input tensor being sliced.
    slice,
    indices,
    {0, 1},  // The dimensions the indices are slicing.
    {1, 1},  // The size of the slice.
    prog     // The poplar program to add this operation to.
  );

  // Print the updated tensor.
  // We should see the value on row 3 column 7 increased by 1.
  prog.add(poplar::program::PrintTensor("updated tensor", tensor));


  // Compile the program.
  Engine engine(graph, prog);
  engine.load(device);

  // Initialise the tensor.
  std::vector<float> initial_values;
  initial_values.resize(m*n);
  std::iota(initial_values.begin(), initial_values.end(), 0.0f);
  engine.writeTensor("tensor_write", initial_values.data(), initial_values.data() + (m*n));

  // Run the program.
  engine.run(0);

  return 0;
}
