import multiprocessing
import struct
import time
import numpy as np
from psycopg import _struct
from psycopg.types.array import _pack_head, _pack_dim
from psycopg.types.array import ListBinaryDumper
import psycopg
from psycopg.types import TypeInfo


from ..base.module import BaseANN


class TestBinaryDumper(ListBinaryDumper):
    def dump(self, obj):
        return bytes(obj)


def float4_dump(obj: np.ndarray):
    data = b""  # placeholders to avoid a resize
    dim: int = obj.shape[0]
    hasnull = 0

    _ele_len = _struct.pack_len(4)
    out = []
    for i in range(dim):
        out.append(_ele_len)
        out.append(struct.pack("!f", obj[i]))
    data = data + b"".join(out)

    data = _pack_head(1, hasnull, psycopg.postgres.types["float4"].oid) + _pack_dim(dim, 1) + data
    return np.frombuffer(data, dtype=np.uint8)


class ArrayBinaryDumper(ListBinaryDumper):
    def dump(self, obj: np.ndarray):
        data = b""
        dim: int = obj.shape[0]
        hasnull = 0

        _ele_len = _struct.pack_len(4)
        out = []
        for i in range(dim):
            out.append(_ele_len)
            out.append(struct.pack("!f", obj[i]))
        data = data + b"".join(out)

        data = _pack_head(1, hasnull, psycopg.postgres.types["float4"].oid) + _pack_dim(dim, 1) + data
        return data


def register_array(conn: psycopg.connection):
    info = TypeInfo.fetch(conn, 'float4[]')
    if info is None:
        raise psycopg.ProgrammingError('vector type not found in the database')
    info.register(conn)

    # add oid to anonymous class for set_types
    test_dumper = type('', (TestBinaryDumper,), {'oid': info.oid})

    adapters = conn.adapters
    adapters.register_dumper('numpy.ndarray', test_dumper)


class ADBPG(BaseANN):
    def __init__(self, metric, method_param):
        print("method_param: ", method_param)
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._host = 'gp-xxxxxxx-master.gpdb.rds.aliyuncs.com'
        self._port = 5432
        self._dbname = 'xxxxx'
        self._user = 'xxxxx'
        self._password = 'xxxxx'
        self._parallel_build_num = method_param['parallel_build']
        self._external_storage = method_param['external_storage']
        self._pq_enable = method_param['pq_enable']
        self._pq_segments = method_param['pq_segments']
        self._insert_parallel = 15
        self._query_curs = []
        self._cur = None
        self._insert_data = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def parallel_insert(self, base, end):
        conn = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=True,
        )
        register_array(conn)

        cur = conn.cursor()
        for i in range(base, end):
            tmp = float4_dump(self._insert_data[i])
            cur.execute(f"INSERT INTO {self._table_name} VALUES (%s, %s)", (i, tmp), binary=True, prepare=True)

    def fit(self, X):
        conn = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=True,
        )
        self._insert_data = X
        self._insert_size = X.shape[0]
        dim = X.shape[1]
        print("dim =", dim)
        register_array(conn)
        cur = conn.cursor()
        # self._cur = cur
        # return
        table_name = f"base_{self._insert_size}_{dim}"
        self._table_name = table_name
        cur.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}')")
        exist = cur.fetchone()[0]
        if exist:
            cur.execute(f"SELECT COUNT() FROM {table_name}")
            cnt = cur.fetchone()[0]
            if cnt == self._insert_size:
                print("base table exists and data is already loaded")
            else:
                exist = False

        if not exist:
            print("table not exists, creating table...")
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            cur.execute(f"CREATE TABLE {table_name} (id int, embedding float4[])")
            print("inserting base table...")
            procs = []
            start = time.time()
            shared_size = self._insert_size // self._insert_parallel + 1
            for i in range(self._insert_parallel):
                base = shared_size * i
                end = min(self._insert_size, base + shared_size)
                worker = multiprocessing.Process(target=self.parallel_insert, args=(base, end))
                worker.start()
                procs.append(worker)

            for proc in procs:
                proc.join()
            end = time.time()
            print("insert data cost %.2f s" % (end - start))


        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE UNLOGGED TABLE items (id int, embedding float4[])")
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying from base table")
        cur.execute(f"INSERT INTO items SELECT * FROM {table_name}")


        print("creating index...")
        cur.execute("SET statement_timeout to 0")
        cur.execute("SET idle_in_transaction_session_timeout to 0")
        cur.execute("SET fastann.build_parallel_processes to %d" % self._parallel_build_num)
        cur.execute("SET fastann.codebook_trainer_nthreads to %d" % self._parallel_build_num)
        cur.execute(f"select count(1) from gp_dist_random('pg_ann_codebooks') where index = 'items_embedding_idx_{dim}' and enable = true")
        codebook_cnt = cur.fetchone()[0]
        if codebook_cnt > 0:
            print("code book already enabled, set codebook_trainer_start_before_building_index to OFF")
            cur.execute("SET fastann.codebook_trainer_start_before_building_index TO off")
        start = time.time()
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX items_embedding_idx_%d ON items USING ann(embedding) WITH (dim=%d,hnsw_m=%d,external_storage=%d,distancemeasure=cosine,hnsw_ef_construction=%d,pq_enable=%d,pq_segments=%d)" % (dim, dim, self._m, self._external_storage, self._ef_construction, self._pq_enable, self._pq_segments)
            )
        elif self._metric == "euclidean":
            cur.execute(
                "CREATE INDEX items_embedding_idx_%d ON items USING ann(embedding) WITH (dim=%d,hnsw_m=%d,external_storage=%d,distancemeasure=l2,hnsw_ef_construction=%d,pq_enable=%d,pq_segments=%d)" % (dim, dim, self._m, self._external_storage, self._ef_construction, self._pq_enable, self._pq_segments)
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        end = time.time()
        print("create index cost %.2f s" % (end - start))

        cur.execute("ANALYZE items")

        print("done!")
        self._cur = cur

    def set_query_arguments(self, params):
        self._ef_search = params["ef_search"]
        self._hnsw_max_scan_points = params["max_scan_points"]
        self._pq_amp = params["pq_amp"]
        self._query_parallel = params["parallel"]

        sql = "SET fastann.hnsw_max_scan_points = %d;" % self._hnsw_max_scan_points
        print(sql)
        self._cur.execute(sql)

        sql = "SET fastann.hnsw_ef_search = %d;" % self._ef_search
        print(sql)
        self._cur.execute(sql)

        sql = "SET fastann.pq_amp = %d;" % self._pq_amp
        print(sql)
        self._cur.execute(sql)

        self._cur.execute("SET optimizer to off;")
        self._cur.execute("SET rds_ann_struct_first_table_size_threshold = %d;" % 0)
        self._cur.execute("SET rds_ann_struct_first_row_threshold = %d;" % 0)
        self._cur.execute("SET rds_ann_struct_first_selectivity_threshold = %f;" % 0)

    def set_query_arguments_for_cur(self, cur):
        sql = "SET fastann.hnsw_max_scan_points = %d;" % self._hnsw_max_scan_points
        cur.execute(sql)

        sql = "SET fastann.hnsw_ef_search = %d;" % self._ef_search
        cur.execute(sql)

        sql = "SET fastann.pq_amp = %d;" % self._pq_amp
        cur.execute(sql)

        cur.execute("SET optimizer to off;")
        cur.execute("SET rds_ann_struct_first_table_size_threshold = %d;" % 0)
        cur.execute("SET rds_ann_struct_first_row_threshold = %d;" % 0)
        cur.execute("SET rds_ann_struct_first_selectivity_threshold = %f;" % 0)

    def query(self, v, n):
        tmp = float4_dump(v)
        self._cur.execute(self._query, (tmp, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def parallel_query(self, n, proc_id):
        conn = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=True,
        )
        register_array(conn)
        cursor = conn.cursor()
        self.set_query_arguments_for_cur(cursor)

        base = (self._query_size // self._query_parallel) * proc_id
        end = min(self._query_size, base + self._query_size // self._query_parallel)
        local_cost_list = [0] * (end - base)

        _s = time.time()
        for i in range(base, end):
            tmp = float4_dump(self._query_data[i].copy())
            start = time.time()
            cursor.execute(self._query, (tmp, n), binary=True, prepare=True)
            local_cost_list[i-base] = time.time() - start

        print("total_time:", time.time()-_s)

        total_cost = sum(local_cost_list)
        # report qps for this worker
        print("worker %d cost %.2f s, qps %.2f, mean rt %.5f, p99 rt %.5f" %
              (proc_id, total_cost, (end - base) / total_cost, total_cost/len(local_cost_list), np.percentile(local_cost_list, 99)))
        conn.close()
        return

    def batch_query(self, X, n: int) -> None:
        n = 10
        amp_factor = 3
        self._query_data = X.repeat(amp_factor*self._query_parallel, axis=0)
        self._query_size = X.shape[0] * amp_factor *self._query_parallel
        procs = []

        print("query using %d parallel" % self._query_parallel)

        for i in range(self._query_parallel):
            worker = multiprocessing.Process(target=self.parallel_query, args=(n, i))
            procs.append(worker)

        start = time.time()
        for worker in procs:
            worker.start()

        for proc in procs:
            proc.join()

        end = time.time()

        # 计算并打印QPS
        qps = self._query_size/(end-start)
        print("QPS: %.3f" % qps)

        self.res = []

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        # self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return 0

    def __str__(self):
        return f"ADBPG(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}, max_scan_pint={self._hnsw_max_scan_points}, pq_amp={self._pq_amp})"

