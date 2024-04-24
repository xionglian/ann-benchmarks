import psycopg2
import uuid
import random
import time
from ann_benchmarks.algorithms.base import BaseANN

class Holo(BaseANN):
    def __init__(self, user, password, host, port, dbname):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.conn = None
        self.connect()

    def connect(self):
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

    def __str__(self):
        return "Holo"

    def done(self):
        if self.conn:
            self.conn.close()

    def fit(self, X):
        # Holo might not need a fit method if data is pre-loaded or loaded outside the benchmarking context
        pass

    def query(self, v, n):
        cur = self.conn.cursor()
        search_id = str(uuid.uuid4())
        sql = f"""
        SELECT
            doc_id,
            pm_approx_squared_euclidean_distance(vector::real[], %s::real[]) AS similarity
        FROM
            your_holo_table
        WHERE
            project_id = ANY(%s)
        ORDER BY
            similarity asc
        LIMIT {n};
        """
        cur.execute(sql, (v, [8]))  # Example project_id; adjust as necessary
        results = cur.fetchall()
        cur.close()
        return [result[0] for result in results]  # Assuming the first column is the doc_id

# Example usage within ann-benchmarks context
def run_holo_benchmarks():
    holo = Holo("user", "password", "host", "port", "dbname")
    n = 10
    total_time = 0

    for i in range(n):
        query_vector = [random.random()]*1024  # Vector size needs to match the database schema
        start_time = time.time()
        results = holo.query(query_vector, 100)
        duration = time.time() - start_time
        total_time += duration
        print(f'Query {i+1}, Duration: {duration}s, Results: {len(results)}')

    print(f'Average duration: {total_time/n}s')

    holo.done()
