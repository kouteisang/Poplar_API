rm reduce
g++ --std=c++11 reduceWithOutput.cpp -lpoplar -lpopops -lpoputil -lpoplin -o reduce
 ./reduce
 POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report_api_reduce"}'