#files=("mf" "tfcmf") #
files=("tfcfm_sgd" "qfcfm_sgd") #
#files=("tfcfm_sgd") #
# 制御用のFIFOファイルを作成
# 2つのコアで並列処理を実行
core_count=6
for file in "${files[@]}"; do                                           #
  {                                                                     #
    for i in {0..2}; do                                                 #
      {                                                                 #
        rm -f ".out/${file}_$((core_count + i))"                        #
        make ".out/${file}_$((core_count + i))" ARG=$((core_count + i)) #
        # コア指定
        taskset -c $((core_count + i))-$((core_count + i)) ".out/${file}_$((core_count + i))" $i $i #
      } &                                                                                           #
    done                                                                                            #
    core_count=$((core_count + 3))                                                                  #
  }                                                                                                 #
done                                                                                                #

# 全てのジョブが完了するまで待機
wait #