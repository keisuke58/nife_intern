#!/usr/bin/env python3
"""
push_to_neo4j.py — CSVを読んでローカルNeo4j Desktopに一括インポート。

使い方:
  pip install neo4j
  python push_to_neo4j.py                        # パスワード: neo4j (デフォルト)
  python push_to_neo4j.py --password <your_pw>   # パスワード指定
  python push_to_neo4j.py --wipe                 # 既存データを全削除してから投入

接続先: bolt://localhost:7687 (Neo4j Desktop デフォルト)
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent

def run(driver, cypher, params=None):
    with driver.session() as s:
        s.run(cypher, **(params or {}))

def run_many(driver, cypher, rows, batch=500):
    total = 0
    with driver.session() as s:
        for i in range(0, len(rows), batch):
            s.run(cypher, rows=rows[i:i+batch])
            total += min(batch, len(rows)-i)
    return total

def load_csv(filename):
    with open(HERE / filename, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri',      default='bolt://localhost:7687')
    parser.add_argument('--user',     default='neo4j')
    parser.add_argument('--password', default='neo4j')
    parser.add_argument('--wipe',     action='store_true',
                        help='既存ノード・エッジを全削除してから投入')
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j ドライバー未インストール: pip install neo4j")
        return

    print(f"接続: {args.uri}  ユーザー: {args.user}")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    print("接続OK")

    # --- Wipe ---
    if args.wipe:
        print("既存データ削除中...")
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("  削除完了")

    # --- Constraints ---
    print("制約作成...")
    run(driver, "CREATE CONSTRAINT species_id IF NOT EXISTS FOR (s:Species) REQUIRE s.id IS UNIQUE")
    run(driver, "CREATE CONSTRAINT substrate_id IF NOT EXISTS FOR (s:Substrate) REQUIRE s.id IS UNIQUE")

    # --- Species nodes ---
    print("Species ノード投入...")
    rows = [
        {
            'id':    r[':ID'],
            'name':  r['name:STRING'],
            'HMT':   r['HMT:STRING'],
            'eHOMD': r['eHOMD:STRING'],
            'cls':   r['class:STRING'],
        }
        for r in load_csv('nodes_species.csv')
    ]
    n = run_many(driver,
        "UNWIND $rows AS row "
        "MERGE (s:Species {id: row.id}) "
        "SET s.name=row.name, s.HMT=row.HMT, s.eHOMD=row.eHOMD, s.class=row.cls",
        rows)
    print(f"  {n} ノード")

    # --- Substrate nodes ---
    print("Substrate ノード投入...")
    rows = [
        {
            'id':       r[':ID'],
            'name':     r['name:STRING'],
            'sub_type': r['sub_type:STRING'],
            'HMDB_ID':  r['HMDB_ID:STRING'],
            'HMDB_url': r['HMDB_url:STRING'],
            'EC':       r['EC:STRING'],
            'KEGG':     r['KEGG:STRING'],
        }
        for r in load_csv('nodes_substrate.csv')
    ]
    n = run_many(driver,
        "UNWIND $rows AS row "
        "MERGE (s:Substrate {id: row.id}) "
        "SET s.name=row.name, s.sub_type=row.sub_type, "
        "    s.HMDB_ID=row.HMDB_ID, s.HMDB_url=row.HMDB_url, "
        "    s.EC=row.EC, s.KEGG=row.KEGG",
        rows)
    print(f"  {n} ノード")

    # --- Species→Substrate relationships (8 types, one query per type) ---
    print("Species→Substrate エッジ投入...")
    ss_rows = load_csv('rels_species_substrate.csv')
    # group by type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in ss_rows:
        by_type[r[':TYPE']].append({
            'src':      r[':START_ID'],
            'dst':      r[':END_ID'],
            'evidence': r['evidence:STRING'],
            'figure':   r['figure:STRING'],
            'pubmed1':  r['pubmed1:STRING'],
            'pubmed2':  r['pubmed2:STRING'],
            'refs':     r['references:STRING'],
            'comments': r['comments:STRING'],
        })
    total_edges = 0
    for rel_type, rows in by_type.items():
        n = run_many(driver,
            f"UNWIND $rows AS row "
            f"MATCH (a:Species  {{id: row.src}}) "
            f"MATCH (b:Substrate{{id: row.dst}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r.evidence=row.evidence, r.figure=row.figure, "
            f"    r.pubmed1=row.pubmed1, r.pubmed2=row.pubmed2, "
            f"    r.references=row.refs, r.comments=row.comments",
            rows)
        total_edges += n
        print(f"  {rel_type}: {n}")
    print(f"  合計 {total_edges} エッジ")

    # --- Species→Species SUPPORTS (MOESM3) ---
    print("SUPPORTS エッジ投入 (MOESM3)...")
    rows = [
        {
            'src':     r[':START_ID'],
            'dst':     r[':END_ID'],
            'oral':    r['oral:STRING'],
            'strain':  r['strain_details:STRING'],
            'cocult':  r['co_culture:STRING'],
            'comment': r['comments:STRING'],
            'ref':     r['reference:STRING'],
        }
        for r in load_csv('rels_species_supports.csv')
    ]
    n = run_many(driver,
        "UNWIND $rows AS row "
        "MATCH (a:Species{id: row.src}) "
        "MATCH (b:Species{id: row.dst}) "
        "MERGE (a)-[r:SUPPORTS]->(b) "
        "SET r.oral=row.oral, r.strain_details=row.strain, "
        "    r.co_culture=row.cocult, r.comments=row.comment, "
        "    r.reference=row.ref",
        rows)
    print(f"  {n} エッジ")

    # --- Summary ---
    print("\n=== 確認 ===")
    with driver.session() as s:
        result = s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
        for rec in result:
            print(f"  {rec['label']}: {rec['cnt']}")
        result = s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY cnt DESC")
        for rec in result:
            print(f"  [{rec['t']}]: {rec['cnt']}")

    driver.close()
    print("\n完了!")

if __name__ == '__main__':
    main()
