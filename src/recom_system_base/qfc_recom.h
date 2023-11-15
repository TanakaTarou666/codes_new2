#include "../math_utils/tensor.h"
#include "tfc_recom.h"

class QFCRecom : virtual public TFCRecom {
   protected:
   Vector cluster_size_adjustments_,prev_cluster_size_adjustments_;

   public:
    QFCRecom(int missing_count);
    virtual void calculate_membership() override;
    virtual void calculate_cluster_size_adjustments();
};