#include "recom.h"

class FMBase : virtual public Recom {
   protected:
    
   public:
    FMBase(int missing_count);
    double predict_y(SparseVector &x, double w0, Vector w, Matrix &v);
};