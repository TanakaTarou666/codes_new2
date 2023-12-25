#include "recom.h"

#ifndef __FMBASE__
#define __FMBASE__

class FMBase : virtual public Recom {
   public:
    FMBase(int missing_count);
    double predict_y(SparseVector &x, double w0, Vector w, Matrix &v);
    SparseVector make_one_hot_data(int user_index, int item_index);
};

#endif