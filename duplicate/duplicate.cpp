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
#include <poputil/Util.hpp>


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

    Tensor d_a = createMatMulInputLHS(graph, FLOAT, {3, 3}, {3, 3}, "d_a");

    auto stream_a = graph.addHostToDeviceFIFO("stream_a", FLOAT, 3*3);
    
    std::vector<float> h_a(3*3);

    for(int i = 0; i < 3; i ++){
        for(int j = 0; j < 3; j ++){
            h_a[i*3+j] = i*3+j;
        }
    }


    program::Sequence prog;
    prog.add(Copy(stream_a, d_a));
    // d_a[0,0]=2;
    Tensor d_duplicate = poputil::duplicate(graph, d_a, prog, "cloneoperation");
    prog.add(PrintTensor("d_duplicate", d_duplicate));


    Engine engine(graph, prog);
    engine.load(device);
    engine.connectStream("stream_a", h_a.data(), h_a.data()+h_a.size());
    std::cout << "Running program\n";
    clock_t startTime, endTime;
    startTime = clock();
    engine.run(0);
    endTime = clock();
    double timeUse =  (double)(endTime - startTime)/CLOCKS_PER_SEC;
    std::cout << "Time use = " << timeUse << "\n";
    std::cout << "Program complete\n";
}