#include "../math_utils/tensor.h"
#include "recom.h"

class TFCRecom : virtual public Recom {
   protected:
    Matrix membership_, prev_membership_;
    Matrix dissimilarities_;
    int cluster_size_;
    double fuzzifier_em_,fuzzifier_Lambda_;

   public:
    TFCRecom(int missing_count);
    virtual void calculate_dissimilarities(Tensor predict_correct_data);
    virtual void calculate_membership();
};