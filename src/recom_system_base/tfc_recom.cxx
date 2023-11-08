#include "tfc_recom.h"

TFCRecom::TFCRecom(int missing_pattern) : Recom(missing_pattern), membership_(), prev_membership_(), dissimilarities_() {}

void TFCRecom::calculate_membership() {
    prev_membership_ = membership_;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            if (dissimilarities_(c,i) != 0.0) {
                double denominator = 0.0;
                for (int j = 0; j < cluster_size_; j++) {
                    denominator += pow((1 - fuzzifier_Lambda_ * (1 - fuzzifier_em_) * dissimilarities_[c][i]) /
                                           (1 - fuzzifier_Lambda_ * (1 - fuzzifier_em_) * dissimilarities_[j][i]),
                                       1.0 / (fuzzifier_em_ - 1));
                }
                membership_(c,i)= 1.0 / denominator;
            }
        }
    }
    return;
}
