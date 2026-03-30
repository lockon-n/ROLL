tar czf kernel_cache.tar.gz ~/.triton/cache/ ~/.cache/vllm/
hdfs dfs -put -f kernel_cache.tar.gz hdfs://harunava/home/byte_malia_gcp_aiic/user/junlongli/cache/
echo "Cache updated!"