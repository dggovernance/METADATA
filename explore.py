# %%
import csv

def get_measure_columns(measure_name, dax_csv_path):
    result = []
    with open(dax_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['nameMeasure'].strip().lower() == measure_name.strip().lower():
                result.append((row['table_name_in_DAX'].strip().upper(), 
                               row['column_name_in_DAX'].strip().upper(), 
                               row['exist_measure'].strip()))
    return result

def trace_measures(measure_name, dax_csv_path, visited=None):
    if visited is None:
        visited = set()
    all_relations = []
    stack = [measure_name]
    while stack:
        current_measure = stack.pop()
        if current_measure.lower() in visited or not current_measure:
            continue
        visited.add(current_measure.lower())
        rows = get_measure_columns(current_measure, dax_csv_path)
        for table, column, exist_measure in rows:
            all_relations.append({'measure': current_measure, 'table': table, 'column': column})
            if exist_measure:  # Nếu có liên kết đến measure khác thì đi tiếp
                for next_measure in [x.strip() for x in exist_measure.split(',') if x.strip()]:
                    if next_measure.lower() not in visited:
                        stack.append(next_measure)
    return all_relations

# Ví dụ sử dụng:
# relations = trace_measures("#Acct L2M", "C:/Users/hungpv6/Desktop/Metric_ZeroFee/table_dax.csv")
# for rel in relations:
#     print(rel)

# %%
def get_relationships(relationship_csv_path):
    rels = []
    with open(relationship_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rels.append( (row['fromTable'].strip().upper(), row['fromColumn'].strip().upper(),
                          row['toTable'].strip().upper(), row['toColumn'].strip().upper()) )
    return rels

# Ví dụ:
# relationships = get_relationships("C:/Users/hungpv6/Desktop/Metric_ZeroFee/relationship_table.csv")
# print(relationships)

# %%
import json
import re

def get_table_lineage_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    table_lineage = {}
    for tbl, info in data.items():
        content = info.get('content', '')
        # Tìm các bảng nguồn trong FROM/JOIN
        sources = set()
        for m in re.findall(r"(FROM|JOIN)\s+([A-Z0-9_\.\@]+)", content, re.I):
            sources.add(m[1].split()[0].upper())
        table_lineage[tbl.upper()] = list(sources)
    return table_lineage

# Ví dụ:
# table_lineage = get_table_lineage_from_json("C:/Users/hungpv6/Desktop/Metric_ZeroFee/output.json")
# print(table_lineage)

# %%
from collections import defaultdict

class Node:
    def __init__(self, table, column=None, node_type="column"):
        self.table = table
        self.column = column
        self.type = node_type
        self.id = f"{table}.{column}" if column else table

class LineageGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)
    def add_edge(self, parent: Node, child: Node):
        self.edges[parent.id].append(child.id)
        self.nodes[parent.id] = parent
        self.nodes[child.id] = child
    def get_children(self, node: Node):
        return [self.nodes[child_id] for child_id in self.edges.get(node.id,[])]
    
def build_graph(measure_cols, relationships, table_lineage):
    # print(measure_cols)
    # print(relationships)
    # print(table_lineage)

    import time
    time.sleep(1)
    g = LineageGraph()
    for tbl, col, _ in measure_cols:
        g.nodes[f"{tbl}.{col}"] = Node(tbl, col)
    # relationship: từ from → to (tức là from là nguồn, to là đích)
    for fromT, fromC, toT, toC in relationships:
        if fromC and toC:
            g.add_edge(Node(fromT, fromC), Node(toT, toC)) # sửa chiều
       

    # lineage bảng: từ src (nguồn) đến dest (đích)
    for dest, sources in table_lineage.items():
        for src in sources:
            g.add_edge(Node(src), Node(dest)) # sửa chiều
    return g


# %%
from collections import deque

def bfs_collect_lineage(graph, start_nodes, max_depth=10):
    visited = set()
    edges = []
    queue = deque([(n, 0) for n in start_nodes])
    while queue:
        node, depth = queue.popleft()
        if node.id in visited or depth > max_depth:
            continue
        visited.add(node.id)
        for child in graph.get_children(node):
            if (node.id, child.id) not in edges:
                edges.append( (node.id, child.id) )
            queue.append((child, depth + 1))
    return edges

# Ví dụ:
measure_cols = get_measure_columns("#Acct L2M", "C:/Users/hungpv6/Desktop/Metric_ZeroFee/table_dax.csv")
relationships = get_relationships("C:/Users/hungpv6/Desktop/Metric_ZeroFee/relationship_table.csv")
table_lineage = get_table_lineage_from_json("C:/Users/hungpv6/Desktop/Metric_ZeroFee/output.json")

g = build_graph(measure_cols, relationships, table_lineage)
print(g)
start_nodes = [Node(tbl, col) for tbl, col, _ in measure_cols]
edges = bfs_collect_lineage(g, start_nodes)
print(edges)  # [(cha, con), ...]

# %%
print("Nodes:")
for node_id, node in g.nodes.items():
    print(f"  {node_id}: {node}")

print("\nEdges:")
for parent_id, child_ids in g.edges.items():
    print(f"  {parent_id} -> {child_ids}")

# %%
# Có thể dùng dict lồng dict để lưu trie nếu muốn build tất cả đường đi từ gốc đến lá.
def build_trie_from_edges(edges):
    trie = {}
    for parent, child in edges:
        trie.setdefault(parent, []).append(child)
    return trie

# %%
import pandas as pd

def lineage_to_dataframe(edges):
    df = pd.DataFrame(edges, columns=["Parent", "Child"])
    return df

df = lineage_to_dataframe(edges)
print(df)


