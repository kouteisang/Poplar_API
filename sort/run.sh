rm sort
g++ --std=c++11 sort.cpp -lpoplar -lpopops -lpoputil -lpoplin -o sort
./sort
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report"}'