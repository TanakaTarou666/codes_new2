#include "recom.h"

#ifndef __FMBASE__
#define __FMBASE__

class FMBase : virtual public Recom {
   protected:
    int sum_users_items= rs::num_users+rs::num_items;
    
   public:
    FMBase(int missing_count);
    double predict_y(SparseVector &x, double w0, Vector w, Matrix &v);
};

#endif