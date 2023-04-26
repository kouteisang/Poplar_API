#include <poplar/Vertex.hpp>
#include <print.h>
using namespace poplar;

// Subtract the minimum value for each row and col, this the codelet for step 1
class RowMaxCS : public Vertex {
public:

    Input<Vector<int>> row;
    Output<Vector<int>> row_max_2;

    bool compute() {
        int res = 0;
        int n = row.size();
        for(int i = 0; i < n; i ++){
            if(row[i] > res){
                res = row[i];
            }
        }
        row_max_2[0] = res;
        return true;
    }
};
