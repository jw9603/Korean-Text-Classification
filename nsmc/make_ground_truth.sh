mkdir metric
awk -F'\t' '{print $1}' ./data/review.sorted.refined.tok.shuf.test.tsv > ./metric/ground_truth.result.txt