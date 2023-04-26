rm gteq
g++ --std=c++11 gteq.cpp -lpoplar -lpopops -lpoputil -lpoplin -o gteq
./gteq
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report"}'