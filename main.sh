files=("mf" "tfcmf")

# 制御用のFIFOファイルを作成
# 2つのコアで並列処理を実行
for file in "${files[@]}";
do
  {
    make ".out/$file${i}.out"
    # コア指定
    taskset -c 0-1 ".out/$file"
  } &
done

# 全てのジョブが完了するまで待機
wait
