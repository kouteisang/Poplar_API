#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>
#include <vector>

#include <popops/Sort.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>

// g++ --std=c++11 maxmul_api.cpp -lpoplar -lpopops -lpoputil -lpoplin -o matrixMulApi
using namespace std;
using namespace poplin;
using namespace poputil;
using namespace poplar;
using namespace poplar::program;


int main(){

    vector<int> a(9);
    // a[0] = 3;
    // a[1] = 1;
    // a[2] = 2;
    // a[3] = -1;
    // a[4] = -1;
    // a[5] = 1;
    // a[6] = 4;
    // a[7] = 7;
    // a[8] = 1;

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 1;
    a[4] = 2;
    a[5] = 3;
    a[6] = 1;
    a[7] = 2;
    a[8] = 3; 

    auto manager = DeviceManager::createDeviceManager();
    
    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);

    std::cout << "Trying to attach to IPU\n";

    auto it = std::find_if(devices.begin(), devices.end(),
                         [](Device &device) { return device.attach(); });

    if (it == devices.end()) {
        std::cerr << "Error attaching to device\n";
        return 1; // EXIT_FAILURE
    }
    auto device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;

    auto target = device.getTarget();
    // Get the num of tiles in the IPU
    const auto numTiles = target.getNumTiles();

    Graph graph(target);
    popops::addCodelets(graph);

    auto stream = graph.addHostToDeviceFIFO("input_stream", INT, 3);

    Sequence prog;

    Tensor d_a = graph.addVariable(INT, {3}, "d_a");
    Tensor d_b = graph.addVariable(INT, {3}, "d_b");
    Tensor d_c = graph.addVariable(INT, {3}, "d_c"); 
    graph.setTileMapping(d_a, 0);
    graph.setTileMapping(d_b, 0);
    graph.setTileMapping(d_c, 0);
     

    prog.add(Copy(stream, d_a));
    prog.add(Copy(stream, d_b));
    prog.add(Copy(stream, d_c));
     
    prog.add(PrintTensor("before_sort_d_a", d_a)); 
    prog.add(PrintTensor("before_sort_d_b", d_b));
    prog.add(PrintTensor("before_sort_d_c", d_c));
     
    Tensor d_b_after = popops::sortKeyValue(graph, d_a, d_b, 0, prog, "test");
    Tensor d_c_after = popops::sortKeyValue(graph, d_a, d_c, 0, prog, "test");
     

    prog.add(PrintTensor("after_sort_d_a", d_a));
    prog.add(PrintTensor("after_sort_d_b", d_b_after));
    prog.add(PrintTensor("after_sort_d_c", d_c_after));


    Engine engine(graph, prog);
    engine.load(device);
    engine.connectStream("input_stream", a.data(), a.data() + a.size());

    std::cout << "Running program\n";
    engine.run(0);
}