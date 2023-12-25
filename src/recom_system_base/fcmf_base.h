#include "recom.h"

#ifndef __FCMFBASE__
#define __FCMFBASE__

class FCMFBase : virtual public Recom {
   protected:
    // 正則化パラメータ
    double reg_parameter_;
    // 学習率
    double learning_rate_;
    // 潜在次元
    int latent_dimension_;
    // ユーザー行列とアイテム行列
    Tensor user_factors_, item_factors_;
    Tensor prev_user_factors_, prev_item_factors_;
    double *user_factor_values_, *item_factor_values_;
    // クラスタリング
    Matrix membership_, prev_membership_;
    Matrix dissimilarities_;
    int cluster_size_;
    double fuzzifier_em_, fuzzifier_lambda_;

   public:
    FCMFBase(int missing_count);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda, double reg_parameter,
                        double learning_rate);
    void set_initial_values(int seed) override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};

#endif