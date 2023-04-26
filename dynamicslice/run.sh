rm dynamicslice
g++ --std=c++11 dynamicslice.cpp -lpoplar -lpopops -lpoputil -lpoplin -o dynamicslice
./dynamicslice
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report"}'