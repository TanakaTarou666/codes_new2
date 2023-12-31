#include "recom.h"

#ifndef __TFCRECOM__
#define __TFCRECOM__

class TFCRecom : virtual public Recom {
   protected:
    Matrix membership_, prev_membership_;
    Matrix dissimilarities_;
    int cluster_size_;
    double fuzzifier_em_,fuzzifier_lambda_;

   public:
    TFCRecom(int missing_count);
    virtual void calculate_membership();
    
};

#endif