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
#include <popops/TopK.hpp>
#include <popops/SortOrder.hpp>


// g++ --std=c++11 maxmul_api.cpp -lpoplar -lpopops -lpoputil -lpoplin -o matrixMulApi
using namespace std;
using namespace poplin;
using namespace poputil;
using namespace poplar;
using namespace poplar::program;


int main(){

    int n = 2500;
    vector<int> a(n);

    for(int i = 0; i < n; i ++){
        a[i] = rand() % 25000;
    }

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

    auto stream = graph.addHostToDeviceFIFO("input_stream", INT, n);

    Sequence write;

    Tensor d_a = graph.addVariable(INT, {n}, "d_a");
    graph.setTileMapping(d_a, 0);
    write.add(Copy(stream, d_a)); 
    write.add(PrintTensor("d_a_before", d_a));

    Sequence opt; 

    std::pair<poplar::Tensor, poplar::Tensor> topOne = popops::topKWithPermutation(graph, opt, d_a, popops::TopKParams(1, true, popops::SortOrder::NONE, true), "topK");

    Sequence out;
    out.add(PrintTensor("d_a_after", d_a));  
    out.add(PrintTensor("pairs_first", topOne.first)); 
    out.add(PrintTensor("pairs_second", topOne.second)); 


    Engine engine(graph, {write, opt, out});
    engine.load(device);
    engine.connectStream("input_stream", a.data(), a.data() + a.size());

    std::cout << "Running program\n";
    engine.run(0);

    Engine::TimerTimePoint run_start = engine.getTimeStamp();
    engine.run(1);
    Engine::TimerTimePoint run_end = engine.getTimeStamp();
    string timing_run = engine.reportTiming(run_start, run_end);
    std::cout << "time to run = " << timing_run << "\n";
    
    engine.run(2);
}