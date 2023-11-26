#include "qfc_recom.h"

QFCRecom::QFCRecom(int missing_pattern)
    : TFCRecom(missing_pattern), Recom(missing_pattern), cluster_size_adjustments_(), prev_cluster_size_adjustments_() {}

void QFCRecom::calculate_membership() {
    prev_membership_ = membership_;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            if (dissimilarities_(c, i) != 0.0) {
                double denominator = 0.0;
                for (int j = 0; j < cluster_size_; j++) {
                    denominator += cluster_size_adjustments_[j] / cluster_size_adjustments_[c] *
                                   pow((1 - fuzzifier_lambda_ * (1 - fuzzifier_em_) * dissimilarities_(j, i)) /
                                           (1 - fuzzifier_lambda_ * (1 - fuzzifier_em_) * dissimilarities_(c, i)),
                                       1.0 / (1 - fuzzifier_em_));
                }
                membership_(c, i) = 1.0 / denominator;
            }
        }
    }
    return;
}

void QFCRecom::calculate_cluster_size_adjustments() {
    prev_cluster_size_adjustments_ = cluster_size_adjustments_;
    double denominator_factors[cluster_size_] = {};
    for (int c = 0; c < cluster_size_; c++) {
        double tmp_denominator_factor = 0.0;
        for (int k = 0; k < rs::num_users; k++) {
            if ((dissimilarities_(c, k) != 0.0) && (membership_(c, k) != 0.0)) {
                tmp_denominator_factor +=
                    pow(membership_(c, k), fuzzifier_em_) * (1 - fuzzifier_lambda_ * (1 - fuzzifier_em_) * dissimilarities_(c, k));
            }
        }
        denominator_factors[c] = tmp_denominator_factor;
    }
    for (int c = 0; c < cluster_size_; c++) {
        double denominator = 0.0;
        for (int j = 0; j < cluster_size_; j++) {
            denominator += pow(denominator_factors[j] / denominator_factors[c], 1 / (fuzzifier_em_));
        }
        cluster_size_adjustments_[c] = 1 / denominator;
    }
}
