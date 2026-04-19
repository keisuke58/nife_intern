// ============================================================
// Oral Biofilm Knowledge Graph — Neo4j Desktop LOAD CSV import
// Source: Dieckow 2025 npj Biofilms SI + Szafranski 2025 Nat Comms MOESM3
//
// Step 1: Copy these 4 files to your Neo4j import folder:
//   nodes_species.csv
//   nodes_substrate.csv
//   rels_species_substrate.csv
//   rels_species_supports.csv
//
// Step 2: Run each block below in Neo4j Browser (one at a time).
// ============================================================

// --- Block 1: Constraints ---
CREATE CONSTRAINT species_id IF NOT EXISTS FOR (s:Species) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT substrate_id IF NOT EXISTS FOR (s:Substrate) REQUIRE s.id IS UNIQUE;

// --- Block 2: Species nodes ---
LOAD CSV WITH HEADERS FROM 'file:///nodes_species.csv' AS row
MERGE (s:Species {id: row.`:ID`})
SET s.name   = row.`name:STRING`,
    s.HMT    = row.`HMT:STRING`,
    s.eHOMD  = row.`eHOMD:STRING`,
    s.class  = row.`class:STRING`;

// --- Block 3: Substrate nodes ---
LOAD CSV WITH HEADERS FROM 'file:///nodes_substrate.csv' AS row
MERGE (s:Substrate {id: row.`:ID`})
SET s.name     = row.`name:STRING`,
    s.sub_type = row.`sub_type:STRING`,
    s.HMDB_ID  = row.`HMDB_ID:STRING`,
    s.HMDB_url = row.`HMDB_url:STRING`,
    s.EC       = row.`EC:STRING`,
    s.KEGG     = row.`KEGG:STRING`;

// --- Block 4: Species→Substrate relationships ---
LOAD CSV WITH HEADERS FROM 'file:///rels_species_substrate.csv' AS row
MATCH (a:Species  {id: row.`:START_ID`})
MATCH (b:Substrate{id: row.`:END_ID`})
CALL apoc.merge.relationship(a, row.`:TYPE`,
  {evidence: row.`evidence:STRING`,
   figure:   row.`figure:STRING`,
   pubmed1:  row.`pubmed1:STRING`,
   pubmed2:  row.`pubmed2:STRING`,
   references: row.`references:STRING`,
   comments: row.`comments:STRING`},
  {}, b) YIELD rel RETURN count(rel);

// --- Block 4 (no APOC fallback) ---
// If APOC is not installed, run this instead of Block 4:
// LOAD CSV WITH HEADERS FROM 'file:///rels_species_substrate.csv' AS row
// MATCH (a:Species  {id: row.`:START_ID`})
// MATCH (b:Substrate{id: row.`:END_ID`})
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'PRODUCES'       THEN [1] ELSE [] END |
//   MERGE (a)-[:PRODUCES       {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'USES'           THEN [1] ELSE [] END |
//   MERGE (a)-[:USES           {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'IS_INHIBITED_BY' THEN [1] ELSE [] END |
//   MERGE (a)-[:IS_INHIBITED_BY{evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'DEGRADES'        THEN [1] ELSE [] END |
//   MERGE (a)-[:DEGRADES       {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'DEPENDS_ON'      THEN [1] ELSE [] END |
//   MERGE (a)-[:DEPENDS_ON     {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'HYDROLYSES'      THEN [1] ELSE [] END |
//   MERGE (a)-[:HYDROLYSES     {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'RELEASES'        THEN [1] ELSE [] END |
//   MERGE (a)-[:RELEASES       {evidence:row.`evidence:STRING`}]->(b))
// FOREACH (_ IN CASE row.`:TYPE` WHEN 'IS_A_HOST_FOR'   THEN [1] ELSE [] END |
//   MERGE (a)-[:IS_A_HOST_FOR  {evidence:row.`evidence:STRING`}]->(b));

// --- Block 5: Species→Species SUPPORTS (MOESM3) ---
LOAD CSV WITH HEADERS FROM 'file:///rels_species_supports.csv' AS row
MATCH (a:Species{id: row.`:START_ID`})
MATCH (b:Species{id: row.`:END_ID`})
MERGE (a)-[r:SUPPORTS]->(b)
SET r.oral           = row.`oral:STRING`,
    r.strain_details = row.`strain_details:STRING`,
    r.co_culture     = row.`co_culture:STRING`,
    r.comments       = row.`comments:STRING`,
    r.reference      = row.`reference:STRING`;

// --- Verification queries ---
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count;
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC;
