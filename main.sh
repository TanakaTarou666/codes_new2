files=("tfcfm_sgd") #
# 制御用のFIFOファイルを作成
# 2つのコアで並列処理を実行
for file in "${files[@]}"; do #
  { #
    for i in {0..7}; do #
      { #
        rm -f ".out/${file}_${i}" #
        make ".out/${file}_${i}" ARG=${i} #
        # コア指定
        taskset -c ${i}-${i} ".out/${file}_${i}" ${i} ${i} #
      } & #
    done #
    wait #
  } #
done #

# 全てのジョブが完了するまで待機
wait #
