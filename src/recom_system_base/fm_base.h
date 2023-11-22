#include "recom.h"

#ifndef __FMBASE__
#define __FMBASE__

class FMBase : virtual public Recom {
   protected:
    
   public:
    FMBase(int missing_count);
    double predict_y(SparseVector &x, double w0, Vector w, Matrix &v);
    virtual void precompute();
    void train() override;
};

#endif