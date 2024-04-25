1. 将adbpg目录拷贝至ann-benchmark项目下的ann_benchmarks/algorithms/中

2. 构建用于测试的镜像
python install.py --proc 16 --algorithm adbpg

2. 根据实例信息以及测试内容，配置ann_benchmarks/algorithms/config.yml中的参数

导入参数
arg_groups: [
{
    host: 'gp-xxxxxxxxxx-master.gpdb.rds.aliyuncs.com',
    port: 5432,
    user: 'username',
    password: 'passwd',
    dbname: 'database_name',
    M: 16,
    efConstruction: 600,
    parallel_build: 4, #并行构建数，等于每个节点的cpu核心数
    external_storage: 1, #是否使用mmap
    pq_enable: 1 #是否开启PQ
}
]

查询参数
query_args: [[
{
    ef_search: 400,
    max_scan_points: 2000, #最大扫描的点个数
    pq_amp: 10, #pq放大系数
    parallel: 8 #并行查询数,开启--batch时生效
}
]]

3. 使用ann-benchmark框架进行测试
建议打开batch并行测试
example: python run.py --algorithm adbpg --runs 1 --batch
