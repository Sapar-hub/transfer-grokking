for op in add mult; do
  for seed in 42 43 44 45 46; do
    echo "[run] eval_l31_perplexity seed=$seed op=$op"
    python eval_l31_perplexity.py "$seed" "$op"
  done
done
