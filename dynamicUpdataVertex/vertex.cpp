#include <poplar/Vertex.hpp>

using namespace poplar;

class CustomDynamicUpdateScalar : public Vertex {
public:
  InOut<VectorList<float, VectorListLayout::DELTANELEMENTS>> slices;
  Input<Vector<int>> indices;

  int local_row;

  void compute() {
    const auto i = indices[0] - local_row;
    const auto j = indices[1];

    // Check ith row is within `slices`.
    if (0 <= i && i < slices.size()) {
      // The update is a subtract 1.
      slices[i][j] -= 1.0f;
    }
  }
};