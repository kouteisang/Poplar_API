#include <iostream>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include <popops/codelets.hpp>

#include <popops/ElementWise.hpp>
#include <popops/DynamicSlice.hpp>

using namespace poplar;

// Shape of the tensor for this example.
constexpr std::size_t m = 15;
constexpr std::size_t n = 15;

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
  poplar::Tensor tensor = popops::createSliceableTensor(
    graph,          // The poplar graph to put the tensor.
    poplar::FLOAT,  // The element type of the tensor.
    {m, n},         // Shape of the tensor, in this case MxN.
    {0, 1},         // Dimensions which will be sliced.
    {1, 1}          // Size of the slices.
  );

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

  prog.add(poplar::program::PrintTensor("I am the tensor", tensor));

  // Slice the elements from tensor.
  // This is equivalent to `slice = tensor[indices[0]][indices[1]]`.
  poplar::Tensor slice = popops::dynamicSlice(
    graph,   // The poplar graph to put the tensor.
    tensor,  // the input tensor being sliced.
    indices, // The indices of the slice.
    {0, 1},  // The dimensions the indices are slicing.
    {1, 1},  // The size of the slice.
    prog     // The poplar program to add this operation to.
  );

  prog.add(poplar::program::PrintTensor("I am the indices", indices));

  prog.add(poplar::program::PrintTensor("I am the slice", slice));

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
  popops::dynamicUpdate(
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
