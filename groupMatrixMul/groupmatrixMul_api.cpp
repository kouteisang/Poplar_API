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

    Tensor d_matrix1 = createMatMulGroupedInputLHS(graph, FLOAT, FLOAT,
                                                    {3, 3, 3},
                                                    {3, 3, 3},
                                                    "d_matrix1");

    // Tensor d_matrix2 = graph.addVariable(FLOAT, {blocks, blockRow, blockCol}, "d_matrix2");
    Tensor d_matrix2 = createMatMulGroupedInputRHS(graph, FLOAT, FLOAT,    
                                                    {3, 3, 3},
                                                    {3, 3, 3}, "d_matrix2");

    // Define the temp result and the final result
    // Tensor dtemp_out = graph.addVariable(FLOAT, {blocks, blockRow, blockCol}, "dtemp_out");
    // Tensor dtemp_out = createMatMulGroupedOutput(graph, FLOAT, FLOAT, 
    //                                                 {3, 3, 3},
    //                                                 {3, 3, 3}, "dtemp_out");
    Tensor dtemp_out = graph.addVariable(FLOAT, {1, 3, 3, 3}, "dtemp_out");
    graph.setTileMapping(dtemp_out, 0);
    auto stream_a = graph.addHostToDeviceFIFO("stream_a", FLOAT, 3*3*3);
    auto stream_b = graph.addHostToDeviceFIFO("stream_b", FLOAT, 3*3*3);
    
    std::vector<float> h_a(27);
    std::vector<float> h_b(27);

    int cnt = 0;
    for(int i = 0; i < 3; i ++){
        for(int j = 0; j < 3; j ++){
            for(int k = 0; k < 3; k ++){
                h_a[cnt] = cnt;
                cnt ++;
            }
        }
    }


    cnt = 0;
    for(int i = 0; i < 3; i ++){
        for(int j = 0; j < 3; j ++){
            for(int k = 0; k < 3; k ++){
                h_b[cnt] = cnt;
                cnt ++;
            }
        }
    }

    program::Sequence prog;
    prog.add(Copy(stream_a, d_matrix1));
    prog.add(Copy(stream_b, d_matrix2));
    // prog.add(PrintTensor("d_a", d_matrix1));
    // prog.add(PrintTensor("d_b", d_matrix2));
    Tensor res = poplin::matMulGrouped(graph, d_matrix1, d_matrix2, prog, FLOAT, "dtemp_out");
    prog.add(PrintTensor("res", res));
    prog.add(Copy(res, dtemp_out[0]));
    prog.add(PrintTensor("dtemp_out", dtemp_out[0]));
     

    Engine engine(graph, prog);
    engine.load(device);
    engine.connectStream("stream_a", h_a.data(), h_a.data()+h_a.size());
    engine.connectStream("stream_b", h_b.data(), h_b.data()+h_b.size());
    std::cout << "Running program\n";
    engine.run(0);
    std::cout << "Program complete\n";
}