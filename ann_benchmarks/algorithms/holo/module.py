import multiprocessing
import time
import numpy as np
import os
import psycopg
import json

from ..base.module import BaseANN

class Relyt(BaseANN):
    def __init__(self, metric, method_param):
        print("method_param: ", method_param)
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._host = os.environ.get('holo_host')
        self._port = os.environ.get('holo_port')
        self._dbname = os.environ.get('holo_dbname')
        self._user = os.environ.get('holo_user')
        self._password = os.environ.get('holo_password')
        self._parallel_build_num = method_param['parallel_build']
        self._external_storage = method_param['external_storage']
        self._pq_enable = method_param['pq_enable']
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
        cur = conn.cursor()
        for i in range(base, end):
            tmp = self._insert_data[i]
            #print('str(tmp)', str(tmp))
            tmp_list = tmp.tolist() if isinstance(tmp, np.ndarray) else tmp
            # 使用转换后的列表进行 JSON 序列化
            cur.execute(f"INSERT INTO {self._table_name} (id, embedding) VALUES ({i}, ARRAY{json.dumps(tmp_list)}::real[] )")

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
            sql = f"""CREATE TABLE {table_name} (
                    id bigint,
                    embedding real[] CHECK(array_ndims(feature_col) = 1 AND array_length(feature_col, 1) = {dim})
                );"""
            cur.execute(sql)
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
        cur.execute(f"CREATE UNLOGGED TABLE items (id int, embedding vectors.vector({dim})) USING heap;")
        print("copying from base table")
        cur.execute(f"INSERT INTO items SELECT * FROM {table_name}")


        print("creating index...")
        cur.execute("SET statement_timeout to 0")
        cur.execute("SET idle_in_transaction_session_timeout to 0")
        cur.execute('SET vectors.pgvector_compatibility=on')
        cur.execute("SET fastann.build_parallel_processes to %d" % self._parallel_build_num)
        cur.execute("SET fastann.codebook_trainer_nthreads to %d" % self._parallel_build_num)
        # cur.execute(f"select count(1) from gp_dist_random('pg_ann_codebooks') where index = 'items_embedding_idx_{dim}' and enable = true")
        # codebook_cnt = cur.fetchone()[0]
        # if codebook_cnt > 0:
        #     print("code book already enabled, set codebook_trainer_start_before_building_index to OFF")
        #     cur.execute("SET fastann.codebook_trainer_start_before_building_index TO off")
        start = time.time()

        if self._metric == "angular":

            cur.execute("""CALL set_table_property(
                                            'items',
                                            'INDEX items_embedding_idx_%d',
                                            '{"feature_col":{"algorithm":"Graph",
                                            "distance_method":"InnerProduct",
                                            "builder_params":{"min_flush_proxima_row_count" : 1000,
                                            "min_compaction_proxima_row_count" : 1000,
                                            "max_total_size_to_merge_mb" : 2000}}}')
                            """% (dim)
            )
        elif self._metric == "euclidean":
            cur.execute("""CALL set_table_property(
                                                        'items',
                                                        'INDEX items_embedding_idx_%d',
                                                        '{"feature_col":{"algorithm":"Graph",
                                                        "distance_method":"SquaredEuclidean",
                                                        "builder_params":{"min_flush_proxima_row_count" : 1000,
                                                        "min_compaction_proxima_row_count" : 1000,
                                                        "max_total_size_to_merge_mb" : 2000}}}')
                                        """ % (dim))
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
        tmp = v
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
        cursor = conn.cursor()
        self.set_query_arguments_for_cur(cursor)

        base = (self._query_size // self._query_parallel) * proc_id
        end = min(self._query_size, base + self._query_size // self._query_parallel)
        local_cost_list = [0] * (end - base)

        _s = time.time()
        for i in range(base, end):
            tmp = self._query_data[i].copy()
            tmp = tmp.tolist() if isinstance(tmp, np.ndarray) else tmp
            tmp = json.dumps(tmp)
            start = time.time()
            cursor.execute(self._query, (tmp, n), prepare=True)
            local_cost_list[i-base] = time.time() - start

        print("total_time:", time.time()-_s)

        total_cost = sum(local_cost_list)
        # report qps for this worker
        print("worker %d cost %.2f s, qps %.2f, mean rt %.5f, p50 rt %.5f, p95 rt %.5f, p99 rt %.5f" %
              (proc_id, total_cost, (end - base) / total_cost, total_cost/len(local_cost_list), np.percentile(local_cost_list, 50), np.percentile(local_cost_list, 95), np.percentile(local_cost_list, 99)))
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
        return f"RELYT(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}, max_scan_pint={self._hnsw_max_scan_points}, pq_amp={self._pq_amp})"
