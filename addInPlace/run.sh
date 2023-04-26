rm addInPlace
g++ --std=c++11 addInPlace.cpp -lpoplar -lpopops -lpoputil -lpoplin -o addInPlace
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report_api_addInPlace"}' ./addInPlace