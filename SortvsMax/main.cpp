#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>
#include <vector>

#include <popops/Reduce.hpp> 
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

    int n = 512;
    vector<int> a(n);
    for(int i = 0; i < n; i ++){
        a[i] = rand() % 512;
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
    graph.addCodelets("codelets.cpp");

    auto stream = graph.addHostToDeviceFIFO("input_stream", INT, n);

    Sequence write;

    Tensor d_a = graph.addVariable(INT, {n}, "d_a");
    Tensor d_b = graph.addVariable(INT, {n}, "d_b");
    Tensor d_c = graph.addVariable(INT, {n}, "d_b");
    poputil::mapTensorLinearly(graph, d_a);
    poputil::mapTensorLinearly(graph, d_b);
    poputil::mapTensorLinearly(graph, d_c);
     

    write.add(Copy(stream, d_a));
    write.add(Copy(stream, d_b));
    write.add(Copy(stream, d_c));
     
     
    write.add(PrintTensor("before_sort_d_a", d_a)); 
    write.add(PrintTensor("before_sort_d_b", d_b));
    write.add(PrintTensor("before_sort_d_c", d_c));
     

    Tensor row_max = graph.addVariable(INT, {1}, "d_row_max");
    graph.setTileMapping(row_max, 1);

    Tensor row_max_2 = graph.addVariable(INT, {1}, "d_row_max_2");
    graph.setTileMapping(row_max_2, 1); 

    Sequence sort;
    popops::sortInPlace(graph, d_a, 0, sort, "sort_zero_status");

    Sequence max;
    auto reduce_max = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> row_max_cs;
    // The minimum value for each row
    reduceWithOutput(graph, d_b, row_max, {0}, reduce_max, row_max_cs, "max_each_row");
    for(const auto &cs : row_max_cs){
        max.add(Execute(cs));
    }

    Sequence vertex_version;
    ComputeSet cs = graph.addComputeSet("max_cs");
    VertexRef vtx = graph.addVertex(cs, "RowMaxCS");
    graph.connect(vtx["row"], d_c);
    graph.connect(vtx["row_max_2"], row_max_2);
    graph.setTileMapping(vtx, 2);
    vertex_version.add(Execute(cs));

    Sequence out;
    out.add(PrintTensor("row_max", row_max));
    out.add(PrintTensor("row_max_2", row_max_2));
     

    Engine engine(graph, {write, sort, max, vertex_version, out});
    engine.load(device);
    engine.connectStream("input_stream", a.data(), a.data() + a.size());

    std::cout << "Running program\n";
    engine.run(0);
    Engine::TimerTimePoint sort_start = engine.getTimeStamp();
    engine.run(1);
    Engine::TimerTimePoint sort_end = engine.getTimeStamp();
    string timing_sort = engine.reportTiming(sort_start, sort_end);
    std::cout << "time to sort = " << timing_sort << "\n";


    Engine::TimerTimePoint max_start = engine.getTimeStamp();
    engine.run(2);
    Engine::TimerTimePoint max_end = engine.getTimeStamp();
    string timing_max = engine.reportTiming(max_start, max_end);
    std::cout << "time to max = " << timing_max << "\n";

    Engine::TimerTimePoint max_vertex_start = engine.getTimeStamp();
    engine.run(3);
    Engine::TimerTimePoint max_vertex_end = engine.getTimeStamp();
    string timing_vertex_max = engine.reportTiming(max_vertex_start, max_vertex_end);
    std::cout << "time to max = " << timing_vertex_max << "\n";

    engine.run(4);
}