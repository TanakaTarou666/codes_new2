files=("tfcwnmf" "qfcwnmf") #
#files=("tf_sgd" "qfcfm_sgd") #
#files=("tfcfm_sgd") #
# 制御用のFIFOファイルを作成
# 2つのコアで並列処理を実行
core_count=0                                                            #
dimension=3                                                             #
index=0                                                                 #
for file in "${files[@]}"; do                                           #
  {                                                                     #
    ((index++))                                                         #
    for i in {0..2}; do                                               #
      {                                                                 #
        rm -f ".out/${file}_$((core_count + i))"                        #
        make ".out/${file}_$((core_count + i))" ARG=$((core_count + i)) #
        # コア指定
        if ((index % 2 != 0)); then                                                                                                           #
          taskset -c $((core_count + i))-$((core_count + i)) ".out/${file}_$((core_count + i))" $i $i                                         #
        else                                                                                                                                  #
          taskset -c $((core_count + i))-$((core_count + i)) ".out/${file}_$((core_count + i))" $((dimension - 1 - i)) $((dimension - 1 - i)) #
        fi                                                                                                                                    #
      } &                                                                                                                                     #
    done                                                                                                                                      #
    #core_count=$((core_count + 3))                                                                  #
  }  #
done #
# 全てのジョブが完了するまで待機
wait #
