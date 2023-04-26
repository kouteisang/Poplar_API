#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popops/Reduce.hpp> // If we use reduce, we need include this package

// g++ --std=c++11 maxmul_api.cpp -lpoplar -lpopops -lpoputil -lpoplin -o matrixMulApi
using namespace std;
using namespace poplin;
using namespace poputil;
using namespace poplar;
using namespace poplar::program;

int main(){

    // Create the DeviceManager which is used to discover devices
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
    poplin::addCodelets(graph);
    popops::addCodelets(graph); // If we use popops, we need this one to be included
    
    Tensor d_a = graph.addVariable(FLOAT, {300, 300, 300}, "d_a");
    Tensor d_out = graph.addVariable(FLOAT, {300, 300}, "d_out");
    for(int i = 0; i < 300; i ++){
        graph.setTileMapping(d_a[i], i);
        graph.setTileMapping(d_out[i], i);
    }
    auto stream_a = graph.addHostToDeviceFIFO("stream_a", FLOAT, 300*300*300);
    
    std::vector<float> h_a(300*300*300);

    int cnt = 0;
    for(int i = 0; i < 300; i ++){
        for(int j = 0; j < 300; j ++){
            for(int k = 0; k < 300; k ++){
                h_a[cnt] = j*300+k;
                cnt ++;
            }
        }
    }
    std::cout << std::endl;
    program::Sequence prog;
    prog.add(Copy(stream_a, d_a));
    for(int i = 0; i < 300; i ++){
        popops::addInPlace(graph, d_out, d_a[i], prog, "add");
    }
    prog.add(PrintTensor(d_out));

    Engine engine(graph, prog);
    engine.load(device);
    engine.connectStream("stream_a", h_a.data(), h_a.data()+h_a.size());
    std::cout << "Running program\n";
    clock_t startTime, endTime;
    engine.run(0);
    std::cout << "Program complete\n";
}