// memory.js — Smart memory retrieval for LadybugDB api-server
//
// Architecture:
//   RAM buffer (HebbianTracker) ←→ LadybugDB (Concept + COOCCURS)
//   Write-behind: flush every FLUSH_INTERVAL_MS
//   Read: keyword search + Hebbian boost from RAM
//
// Endpoints:
//   POST /memory/retrieve  — keyword search + hebbian boost + 1-hop
//   POST /memory/decay     — decay weights, prune weak, flush
//   GET  /memory/stats     — tracker statistics

const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();
const database = require("./explorer/src/server/utils/Database");
const logger = require("./explorer/src/server/utils/Logger");
const snowball = require("snowball-stemmers");

// Snowball stemmers (Russian + English)
const stemmerRu = snowball.newStemmer("russian");
const stemmerEn = snowball.newStemmer("english");

// ─── Translation Dictionary (Wiktionary seed + auto-supplement) ─────────────

const DICT_PATH = path.join(__dirname, "dict_ru_en.json");
const CLUSTER_MAX_HOPS = 1;
const CLUSTER_MAX_SIZE = 8;

let translationDict = {};
let clusterMap = new Map(); // word → Set of cluster members

try {
  translationDict = JSON.parse(fs.readFileSync(DICT_PATH, "utf8"));
  logger.info(`Translation dict loaded: ${Object.keys(translationDict).length} keys`);
} catch (err) {
  logger.warn("Translation dict not found, starting empty: " + err.message);
}

// Build clusters: just direct translations (1-hop), no transitive closure.
// Transitive expansion happens at query time via Hebbian.
function buildClusters() {
  clusterMap = new Map();

  for (const word of Object.keys(translationDict)) {
    if (clusterMap.has(word)) continue;

    // Collect: word + all its direct translations
    const cluster = new Set([word]);
    for (const t of (translationDict[word] || [])) {
      cluster.add(t.toLowerCase());
    }

    // Also add reverse: for each translation, add ITS direct translations
    const directTranslations = [...cluster];
    for (const t of directTranslations) {
      for (const rev of (translationDict[t] || [])) {
        if (cluster.size < CLUSTER_MAX_SIZE) cluster.add(rev.toLowerCase());
      }
    }

    // Assign same cluster to all members
    for (const member of cluster) {
      if (!clusterMap.has(member)) {
        clusterMap.set(member, cluster);
      }
    }
  }

  logger.info(`Clusters built: ${clusterMap.size} words mapped`);
}

buildClusters();

// Get cluster for a word (returns array of all variants)
function getCluster(word) {
  const low = word.toLowerCase();
  const cluster = clusterMap.get(low);
  return cluster ? [...cluster] : [low];
}

// ─── Stem Index (morphological normalization for dict lookups) ───────────────

const stemIndex = new Map(); // stem → [original dict keys]
const trigramIndex = new Map(); // trigram → [key indices]
let dictKeysArray = [];

// Trigram extraction (Dice coefficient fuzzy matching)
function trigrams(str) {
  const s = "  " + str.toLowerCase() + " ";
  const result = [];
  for (let i = 0; i < s.length - 2; i++) result.push(s.substring(i, i + 3));
  return result;
}

function buildStemIndex() {
  stemIndex.clear();
  trigramIndex.clear();
  dictKeysArray = Object.keys(translationDict);

  for (let i = 0; i < dictKeysArray.length; i++) {
    const key = dictKeysArray[i];
    // Stem index
    const s = isCyrillic(key) ? stemmerRu.stem(key) : stemmerEn.stem(key);
    if (!stemIndex.has(s)) stemIndex.set(s, []);
    stemIndex.get(s).push(key);
    // Trigram index
    for (const tg of trigrams(key)) {
      if (!trigramIndex.has(tg)) trigramIndex.set(tg, []);
      trigramIndex.get(tg).push(i);
    }
  }

  logger.info(`Stem index: ${stemIndex.size} stems, trigram index: ${trigramIndex.size} trigrams from ${dictKeysArray.length} keys`);
}

buildStemIndex();

// Trigram fuzzy search (language-agnostic, works for both ru and en typos)
function fuzzyTrigramLookup(word) {
  const qTg = trigrams(word);
  const scores = new Map();
  for (const tg of qTg) {
    const idxs = trigramIndex.get(tg);
    if (!idxs) continue;
    for (const idx of idxs) scores.set(idx, (scores.get(idx) || 0) + 1);
  }

  let bestIdx = -1, bestScore = 0;
  for (const [idx, overlap] of scores) {
    const keyTgLen = trigrams(dictKeysArray[idx]).length;
    const score = (2 * overlap) / (qTg.length + keyTgLen); // Dice coefficient
    if (score > bestScore) { bestScore = score; bestIdx = idx; }
  }

  if (bestIdx >= 0 && bestScore >= 0.4) {
    return translationDict[dictKeysArray[bestIdx]] || [];
  }
  return [];
}

// Local dict lookup: exact → stem → trigram fuzzy (all local, sync, no network)
function dictLookup(word) {
  const low = word.toLowerCase();

  // Layer 1: exact match (O(1))
  if (translationDict[low]) return translationDict[low];

  // Layer 2: stem match — handles morphological variants (O(1))
  const s = isCyrillic(low) ? stemmerRu.stem(low) : stemmerEn.stem(low);
  const stemCandidates = stemIndex.get(s);
  if (stemCandidates && stemCandidates.length > 0) {
    const translations = new Set();
    for (const key of stemCandidates) {
      for (const t of (translationDict[key] || [])) translations.add(t);
    }
    return [...translations];
  }

  // Layer 3: trigram fuzzy — handles typos in any language (~2ms over 91k keys)
  if (low.length >= 4) {
    return fuzzyTrigramLookup(low);
  }

  return [];
}

// ─── Config ─────────────────────────────────────────────────────────────────

const MAX_KEYWORDS = 50;
const SEARCH_LIMIT = 5;
const SNIPPET_LEN = 300;
const MIN_ALPHA_CHARS = 4;
const MIN_WORD_COUNT = 1;
const HOP_LIMIT = 10;

const HEBBIAN_LR = 0.05;
const HEBBIAN_DECAY_RATE = 0.001;
const HEBBIAN_EDGE_THRESHOLD = 0.01;
const CONCEPT_CAP = 500;
const FLUSH_INTERVAL_MS = 15_000; // 15 seconds

// ─── Stop Words (ru + en) ───────────────────────────────────────────────────

// Function words only: prepositions, conjunctions, particles, pronouns.
// NO content words here — even common verbs/nouns can be search targets.
const STOP_WORDS = new Set([
  // ru 4+ char: предлоги, союзы, частицы, местоимения
  "если","через","чтобы","когда","тоже","только","потом","потому",
  "этот","этого","этом","этой","этих","есть","нету",
  "того","тому","свой","свою","своё","своего","свои","своим","своей",
  "твой","твою","твоё","твоего","твои","твоим","твоей","твоем",
  "себя","себе","тебя","тебе",
  // en 4+ char: prepositions, conjunctions, pronouns
  "that","this","these","those","with","from","into","over","after",
  "before","between","under","above","below","been","were","does",
  "have","will","would","could","should","them","they","their","there",
  "here","some","each","every","both","than","what","which","where",
  "also","just","only","very","then","else",
]);

// ─── Quality Filter ─────────────────────────────────────────────────────────

function isWorthSearching(text) {
  const stripped = (text || "").trim();
  if (!stripped) return false;
  // Check if text has NO letters at all (only digits/punctuation/spaces)
  if (!(stripped.match(/\p{L}/u))) return false;
  const alphaChars = (stripped.match(/\p{L}/gu) || []).length;
  if (alphaChars < MIN_ALPHA_CHARS) return false;
  const words = stripped.split(/\s+/);
  if (words.length < MIN_WORD_COUNT) return false;
  return true;
}

// ─── Transliteration (ru↔en, bidirectional) ────────────────────────────────

const RU_TO_EN = {
  "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo","ж":"zh",
  "з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o",
  "п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts",
  "ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya",
};

// Build reverse map (en→ru) from multi-char sequences first (greedy match)
const EN_TO_RU_MAP = {};
for (const [ru, en] of Object.entries(RU_TO_EN)) {
  if (en) EN_TO_RU_MAP[en] = ru;
}
// Sort by length descending for greedy matching
const EN_TO_RU_KEYS = Object.keys(EN_TO_RU_MAP).sort((a, b) => b.length - a.length);

function translitRuToEn(text) {
  let result = "";
  for (const ch of text) {
    result += RU_TO_EN[ch] || ch;
  }
  return result;
}

function translitEnToRu(text) {
  let result = "";
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (const key of EN_TO_RU_KEYS) {
      if (text.substring(i, i + key.length) === key) {
        result += EN_TO_RU_MAP[key];
        i += key.length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      result += text[i];
      i++;
    }
  }
  return result;
}

function isCyrillic(text) {
  return /[\u0400-\u04ff]/u.test(text);
}

const HEBBIAN_EXPAND_THRESHOLD = 0.2;
const HEBBIAN_EXPAND_TOP_K = 2;
const VOTING_LIMIT = 20; // results per variant for voting

// Snowball stemmer (proper morphological stemming for ru + en)
function stem(word) {
  if (word.length < 4) return word;
  return isCyrillic(word) ? stemmerRu.stem(word) : stemmerEn.stem(word);
}

// Full pipeline: translate → stem → deduplicate
// Returns array of STEMS for voting search
async function expandKeyword(kw) {
  // Cache check
  const cached = expandCache.get(kw);
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) return cached.stems;

  // Step 1: get cluster (transitive closure from dict)
  let cluster = getCluster(kw);

  // Step 2: if not in dict at all, try translit
  if (cluster.length <= 1) {
    if (isCyrillic(kw)) {
      const en = translitRuToEn(kw);
      cluster = [...new Set([...cluster, ...getCluster(en), en])];
    } else {
      const ru = translitEnToRu(kw);
      cluster = [...new Set([...cluster, ...getCluster(ru), ru])];
    }
  }

  // Step 3: if dict had nothing for this word, try local fuzzy lookup
  const hasDictEntry = translationDict[kw] && translationDict[kw].length > 0;
  if (!hasDictEntry && kw.length >= 4) {
    const localHits = dictLookup(kw);
    for (const t of localHits) {
      cluster = [...new Set([...cluster, t, ...getCluster(t)])];
    }
  }

  // Step 4: Hebbian COOCCURS — check ALL cluster words as seeds
  const assocs = hebbian.getAssociatedKeywords(cluster, HEBBIAN_EXPAND_TOP_K)
    .filter((a) => a.weight >= HEBBIAN_EXPAND_THRESHOLD);
  for (const { keyword: assocKw } of assocs) {
    if (cluster.includes(assocKw)) continue;
    const assocCluster = getCluster(assocKw);
    cluster = [...new Set([...cluster, ...assocCluster])];
  }

  // Step 5: STEM every word in cluster → deduplicate stems
  const stems = new Set();
  for (const word of cluster) {
    stems.add(stem(word));
  }

  const result = [...stems];
  expandCache.set(kw, { stems: result, ts: Date.now() });
  return result;
}

// Build a single-variant Cypher query (one word, both content+id)
function buildSingleWordQuery(word, limit = VOTING_LIMIT) {
  const w = esc(word);
  return `
    MATCH (n:Core) WHERE lower(n.content) CONTAINS "${w}" OR lower(n.id) CONTAINS "${w}"
    RETURN 'Core' AS tbl, n.id AS id LIMIT ${limit}
    UNION ALL
    MATCH (n:POI) WHERE lower(n.content) CONTAINS "${w}" OR lower(n.id) CONTAINS "${w}" OR lower(n.situation) CONTAINS "${w}"
    RETURN 'POI' AS tbl, n.id AS id LIMIT ${limit}
    UNION ALL
    MATCH (n:Daily) WHERE lower(n.content) CONTAINS "${w}" OR lower(n.id) CONTAINS "${w}"
    RETURN 'Daily' AS tbl, n.id AS id LIMIT ${limit}
    UNION ALL
    MATCH (n:Entity) WHERE lower(n.data) CONTAINS "${w}" OR lower(n.id) CONTAINS "${w}" OR lower(n.name) CONTAINS "${w}"
    RETURN 'Entity' AS tbl, n.id AS id LIMIT ${limit};`;
}

// ─── Query Cache ────────────────────────────────────────────────────────────

const CACHE_TTL_MS = 60_000; // 60 seconds
const expandCache = new Map(); // keyword → {stems, ts}
const stemCache = new Map();   // stem → {rows, ts}

function getCachedStemResult(stem) {
  const entry = stemCache.get(stem);
  if (entry && Date.now() - entry.ts < CACHE_TTL_MS) return entry.rows;
  return null;
}

function setCachedStemResult(stem, rows) {
  stemCache.set(stem, { rows, ts: Date.now() });
  // Evict old entries periodically
  if (stemCache.size > 2000) {
    const now = Date.now();
    for (const [k, v] of stemCache) {
      if (now - v.ts > CACHE_TTL_MS) stemCache.delete(k);
    }
  }
}

// Voting: run variant queries in batches, with per-stem caching
const QUERY_BATCH_SIZE = 8;

async function votingSearch(variants, topK = SEARCH_LIMIT * 4) {
  // Split: cached vs uncached stems
  const results = [];
  const uncached = [];
  for (const word of variants) {
    const cached = getCachedStemResult(word);
    if (cached) {
      results.push(cached);
    } else {
      uncached.push(word);
    }
  }

  // Run uncached in batches
  for (let i = 0; i < uncached.length; i += QUERY_BATCH_SIZE) {
    const batch = uncached.slice(i, i + QUERY_BATCH_SIZE);
    const batchResults = await Promise.all(
      batch.map(async (word) => {
        try {
          const rows = await runQuery(buildSingleWordQuery(word));
          setCachedStemResult(word, rows);
          return rows;
        } catch {
          return [];
        }
      })
    );
    results.push(...batchResults);
  }

  // IDF-weighted voting: rare stems = heavy votes, common stems = light votes
  const votes = new Map(); // id → {tbl, score}
  for (let i = 0; i < results.length; i++) {
    const rows = results[i];
    const seen = new Set();
    // Dedup within one variant, count unique hits = IDF denominator
    const uniqueIds = new Set(rows.map((r) => r.id));
    // IDF weight capped: min denominator 2 prevents single-hit typo dominance
    const weight = uniqueIds.size > 0 ? 1.0 / Math.max(uniqueIds.size, 2) : 0;
    for (const r of rows) {
      if (seen.has(r.id)) continue;
      seen.add(r.id);
      const entry = votes.get(r.id) || { tbl: r.tbl, score: 0 };
      entry.score += weight;
      votes.set(r.id, entry);
    }
  }

  // Sort by score DESC, return top K
  return Array.from(votes.entries())
    .map(([id, { tbl, score }]) => ({
      id,
      tbl,
      count: Math.round(score * 100) / 100,
      source: "keyword",
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topK);
}

// ─── Keyword Extraction ─────────────────────────────────────────────────────

function extractKeywords(text, max = MAX_KEYWORDS) {
  // Extract all words 4+ chars that contain at least one letter
  const words = (text.match(/[\p{L}\p{N}_-]{4,}/gu) || [])
    .filter((w) => /\p{L}/u.test(w));  // must have at least one letter

  // Count frequency per word (skip function words)
  const freq = new Map();
  for (const w of words) {
    const low = w.toLowerCase();
    if (STOP_WORDS.has(low)) continue;
    freq.set(low, (freq.get(low) || 0) + 1);
  }

  // Tier 1: core (3+ occurrences) — sorted by freq DESC
  const core = [...freq.entries()]
    .filter(([, c]) => c >= 3)
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w);

  // Tier 2: signal (2x) — sorted by length DESC (longer = more specific)
  const signal = [...freq.entries()]
    .filter(([, c]) => c === 2)
    .sort((a, b) => b[0].length - a[0].length)
    .map(([w]) => w);

  // Tier 3: unique (1x) — sorted by length DESC, top few
  const unique = [...freq.entries()]
    .filter(([, c]) => c === 1)
    .sort((a, b) => b[0].length - a[0].length)
    .map(([w]) => w);

  // Combine: core first (most frequent), then signal, then unique — hard cap at max
  const combined = [...core, ...signal, ...unique];
  return combined.slice(0, max);
}

// ─── Cypher Helpers ─────────────────────────────────────────────────────────

function esc(str) {
  return String(str).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

// Build CONTAINS condition for one keyword with pre-expanded variants
function kwCond(variants, field) {
  return variants.map((v) => `lower(${field}) CONTAINS "${esc(v)}"`).join(" OR ");
}

// Expand all keywords (async — may call Google Translate on miss)
async function expandAllKeywords(keywords) {
  const expanded = new Map(); // keyword -> variants[]
  await Promise.all(
    keywords.map(async (kw) => {
      expanded.set(kw, await expandKeyword(kw));
    })
  );
  return expanded;
}

function buildKeywordQuery(expanded, limit = SEARCH_LIMIT) {
  const keywords = Array.from(expanded.keys());
  const contentConds = keywords
    .map((kw) => {
      const v = expanded.get(kw);
      return `${kwCond(v, "n.content")} OR ${kwCond(v, "n.id")}`;
    })
    .join(" OR ");
  const entityConds = keywords
    .map((kw) => {
      const v = expanded.get(kw);
      return `${kwCond(v, "n.data")} OR ${kwCond(v, "n.id")} OR ${kwCond(v, "n.name")}`;
    })
    .join(" OR ");
  const poiExtra = keywords[0]
    ? ` OR ${kwCond(expanded.get(keywords[0]), "n.situation")}`
    : "";
  return `
    MATCH (n:Core) WHERE ${contentConds}
    RETURN 'Core' AS tbl, n.id AS id, substring(n.content, 0, ${SNIPPET_LEN}) AS snippet
    LIMIT ${limit}
    UNION ALL
    MATCH (n:POI) WHERE ${contentConds}${poiExtra}
    RETURN 'POI' AS tbl, n.id AS id, substring(COALESCE(n.situation, n.content), 0, ${SNIPPET_LEN}) AS snippet
    LIMIT ${limit}
    UNION ALL
    MATCH (n:Daily) WHERE ${contentConds}
    RETURN 'Daily' AS tbl, n.id AS id, substring(n.content, 0, ${SNIPPET_LEN}) AS snippet
    LIMIT ${limit}
    UNION ALL
    MATCH (n:Entity) WHERE ${entityConds}
    RETURN 'Entity' AS tbl, n.id AS id, substring(COALESCE(n.data, n.name), 0, ${SNIPPET_LEN}) AS snippet
    LIMIT ${limit};`;
}

function buildHopQuery(nodeIds) {
  const idList = nodeIds.map((id) => `"${esc(id)}"`).join(", ");
  return `
    MATCH (n)-[r]->(m) WHERE n.id IN [${idList}]
    RETURN n.id AS src, label(r) AS rel, m.id AS target
    LIMIT ${HOP_LIMIT};`;
}

// ─── Query Executor ─────────────────────────────────────────────────────────

async function runQuery(cypher) {
  const conn = database.getConnection();
  try {
    const result = await conn.query(cypher);
    const rows = await result.getAll();
    result.close();
    return rows;
  } finally {
    database.releaseConnection(conn);
  }
}

// ─── Hebbian Concept Tracker (RAM buffer + DB persistence) ──────────────────

class HebbianTracker {
  constructor() {
    // Concept data: keyword -> {weight, accessCount, lastAccessed}
    this.concepts = new Map();
    // COOCCURS edges: "kwA|kwB" (sorted) -> {weight, count}
    this.edges = new Map();
    // Dirty tracking for write-behind
    this.dirtyConcepts = new Set();
    this.dirtyEdges = new Set();
    this.totalQueries = 0;
    this._flushing = false;
  }

  // Record co-occurring keywords from a single retrieve
  recordKeywords(keywords) {
    if (!keywords.length) return;
    this.totalQueries++;
    const now = Date.now();

    // Update concept weights
    for (const kw of keywords) {
      const c = this.concepts.get(kw) || {
        weight: HEBBIAN_EDGE_THRESHOLD,
        accessCount: 0,
        lastAccessed: now,
      };
      c.accessCount++;
      c.weight += HEBBIAN_LR * (1.0 - c.weight);
      c.lastAccessed = now;
      this.concepts.set(kw, c);
      this.dirtyConcepts.add(kw);
    }

    // Update COOCCURS between keyword pairs
    for (let i = 0; i < keywords.length; i++) {
      for (let j = i + 1; j < keywords.length; j++) {
        const key = [keywords[i], keywords[j]].sort().join("|");
        const edge = this.edges.get(key) || {
          weight: HEBBIAN_EDGE_THRESHOLD,
          count: 0,
        };
        edge.weight += HEBBIAN_LR * (1.0 - edge.weight);
        edge.count++;
        this.edges.set(key, edge);
        this.dirtyEdges.add(key);
      }
    }

    // Eviction check
    if (this.concepts.size > CONCEPT_CAP) {
      this._evictWeakest();
    }
  }

  // Get associated keywords not in the input set
  getAssociatedKeywords(keywords, topK = 3) {
    const kwSet = new Set(keywords);
    const scores = new Map();
    for (const kw of keywords) {
      for (const [key, edge] of this.edges) {
        const [a, b] = key.split("|");
        let assoc = null;
        if (a === kw && !kwSet.has(b)) assoc = b;
        else if (b === kw && !kwSet.has(a)) assoc = a;
        if (!assoc) continue;
        const existing = scores.get(assoc) || 0;
        scores.set(assoc, Math.max(existing, edge.weight));
      }
    }
    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([kw, weight]) => ({ keyword: kw, weight }));
  }

  // Decay all weights, prune weak
  applyDecay() {
    let prunedEdges = 0;
    let prunedConcepts = 0;
    for (const [key, edge] of this.edges) {
      edge.weight *= 1 - HEBBIAN_DECAY_RATE;
      if (edge.weight < HEBBIAN_EDGE_THRESHOLD) {
        this.edges.delete(key);
        this.dirtyEdges.add(key); // mark for deletion in DB
        prunedEdges++;
      }
    }
    for (const [kw, c] of this.concepts) {
      c.weight *= 1 - HEBBIAN_DECAY_RATE;
      if (c.weight < HEBBIAN_EDGE_THRESHOLD && c.accessCount <= 1) {
        this.concepts.delete(kw);
        this.dirtyConcepts.add(kw); // mark for deletion in DB
        prunedConcepts++;
      }
    }
    return { prunedEdges, prunedConcepts };
  }

  _evictWeakest() {
    const now = Date.now();
    const scored = Array.from(this.concepts.entries()).map(([kw, c]) => {
      const ageMs = Math.max(1, now - (c.lastAccessed || now));
      const ageSec = ageMs / 1000;
      const score = c.weight * (1 + c.accessCount) / Math.log1p(ageSec);
      return { kw, score };
    });
    scored.sort((a, b) => a.score - b.score);
    const toRemove = scored.slice(0, this.concepts.size - CONCEPT_CAP);
    for (const { kw } of toRemove) {
      this.concepts.delete(kw);
      this.dirtyConcepts.add(kw);
      // Remove all edges involving this concept
      for (const key of this.edges.keys()) {
        const [a, b] = key.split("|");
        if (a === kw || b === kw) {
          this.edges.delete(key);
          this.dirtyEdges.add(key);
        }
      }
    }
    logger.info(`Evicted ${toRemove.length} weak concepts`);
  }

  // ─── DB Persistence ─────────────────────────────────────────────────────

  async loadFromDB() {
    try {
      const concepts = await runQuery(
        "MATCH (c:Concept) RETURN c.id AS id, c.weight AS weight, c.access_count AS ac;"
      );
      for (const r of concepts) {
        this.concepts.set(r.id, {
          weight: r.weight || 0.01,
          accessCount: r.ac || 0,
          lastAccessed: Date.now(),
        });
      }
      const edges = await runQuery(
        "MATCH (a:Concept)-[r:COOCCURS]->(b:Concept) RETURN a.id AS src, b.id AS dst, r.weight AS w, r.count AS cnt;"
      );
      for (const r of edges) {
        const key = [r.src, r.dst].sort().join("|");
        this.edges.set(key, { weight: r.w || 0.01, count: r.cnt || 0 });
      }
      logger.info(
        `Hebbian loaded: ${this.concepts.size} concepts, ${this.edges.size} edges`
      );
    } catch (err) {
      logger.error("Hebbian loadFromDB failed: " + err.message);
    }
  }

  async flushToDB() {
    if (this._flushing) return;
    if (this.dirtyConcepts.size === 0 && this.dirtyEdges.size === 0) return;
    this._flushing = true;
    const conceptsBatch = new Set(this.dirtyConcepts);
    const edgesBatch = new Set(this.dirtyEdges);
    this.dirtyConcepts.clear();
    this.dirtyEdges.clear();

    let flushedC = 0, flushedE = 0, failedC = 0, failedE = 0;
    // Flush concepts (per-item error isolation)
    for (const kw of conceptsBatch) {
      try {
        const c = this.concepts.get(kw);
        if (c) {
          await runQuery(
            `MERGE (c:Concept {id: "${esc(kw)}"})` +
              ` SET c.weight = ${c.weight}, c.access_count = ${c.accessCount},` +
              ` c.last_accessed = current_timestamp();`
          );
        } else {
          await runQuery(
            `MATCH (c:Concept {id: "${esc(kw)}"}) DETACH DELETE c;`
          );
        }
        flushedC++;
      } catch (err) {
        failedC++;
        if (this.concepts.has(kw)) this.dirtyConcepts.add(kw);
      }
    }
    // Flush COOCCURS edges (per-item error isolation)
    for (const key of edgesBatch) {
      try {
        const [a, b] = key.split("|");
        const edge = this.edges.get(key);
        if (edge) {
          await runQuery(
            `MATCH (a:Concept {id: "${esc(a)}"}), (b:Concept {id: "${esc(b)}"})` +
              ` MERGE (a)-[r:COOCCURS]->(b)` +
              ` SET r.weight = ${edge.weight}, r.count = ${edge.count};`
          );
        } else {
          await runQuery(
            `MATCH (a:Concept {id: "${esc(a)}"})-[r:COOCCURS]->(b:Concept {id: "${esc(b)}"}) DELETE r;`
          );
        }
        flushedE++;
      } catch (err) {
        failedE++;
        if (this.edges.has(key)) this.dirtyEdges.add(key);
      }
    }
    if (flushedC + flushedE > 0) {
      logger.info(
        `Hebbian flushed: ${flushedC} concepts, ${flushedE} edges` +
          (failedC + failedE > 0 ? ` (failed: ${failedC}c, ${failedE}e)` : "")
      );
    }
    if (failedC + failedE > 0 && flushedC + flushedE === 0) {
      logger.error(`Hebbian flush: all failed (${failedC}c, ${failedE}e)`);
    }
    this._flushing = false;
  }

  getStats() {
    return {
      concepts: this.concepts.size,
      edges: this.edges.size,
      dirtyConcepts: this.dirtyConcepts.size,
      dirtyEdges: this.dirtyEdges.size,
      totalQueries: this.totalQueries,
    };
  }
}

const hebbian = new HebbianTracker();

// Load state from DB on startup (deferred — DB might not be ready yet)
setTimeout(async () => {
  await hebbian.loadFromDB();
}, 2000);

// Write-behind flush timer
setInterval(async () => {
  await hebbian.flushToDB();
}, FLUSH_INTERVAL_MS);

// ─── POST /memory/retrieve ──────────────────────────────────────────────────

router.post("/retrieve", async (req, res) => {
  const start = Date.now();
  const text = req.body.text || "";

  if (!isWorthSearching(text)) {
    return res.json({ results: [], keywords: [], reason: "filtered", ms: 0 });
  }

  const keywords = extractKeywords(text);
  if (!keywords.length) {
    return res.json({ results: [], keywords: [], reason: "no_keywords", ms: 0 });
  }

  try {
    // 1. Expand all keywords into variant clusters
    const allVariants = new Set();
    for (const kw of keywords) {
      const expanded = await expandKeyword(kw);
      for (const v of expanded) allVariants.add(v);
    }
    const variants = [...allVariants];

    // 2. Voting search: each variant queries independently, results ranked by vote count
    const results = await votingSearch(variants, SEARCH_LIMIT * 4);
    const foundIds = results.map((r) => r.id).filter(Boolean);

    // 3. 1-hop traversal for top results
    let hops = [];
    if (foundIds.length > 0) {
      try {
        const hopRows = await runQuery(buildHopQuery(foundIds.slice(0, 10)));
        hops = hopRows.map((r) => ({
          src: r.src,
          rel: r.rel,
          target: r.target,
        }));
      } catch (err) {
        logger.warn("Hop traversal failed: " + err.message);
      }
    }

    // 4. Record keywords for Hebbian learning (RAM only, flushed async)
    hebbian.recordKeywords(keywords);

    const ms = Date.now() - start;
    return res.json({
      results,
      hops,
      keywords,
      variants: variants.length,
      ms,
    });
  } catch (err) {
    logger.error("Memory retrieve error: " + err.message);
    return res.status(400).json({ error: err.message });
  }
});

// ─── POST /memory/decay ─────────────────────────────────────────────────────

router.post("/decay", async (_req, res) => {
  const { prunedEdges, prunedConcepts } = hebbian.applyDecay();
  await hebbian.flushToDB();
  return res.json({
    prunedEdges,
    prunedConcepts,
    remainingConcepts: hebbian.concepts.size,
    remainingEdges: hebbian.edges.size,
  });
});

// ─── POST /memory/flush ─────────────────────────────────────────────────────

router.post("/flush", async (_req, res) => {
  await hebbian.flushToDB();
  return res.json({
    concepts: hebbian.concepts.size,
    edges: hebbian.edges.size,
    dirty: hebbian.dirtyConcepts.size + hebbian.dirtyEdges.size,
  });
});

// ─── GET /memory/stats ──────────────────────────────────────────────────────

router.get("/stats", (_req, res) => {
  const stats = hebbian.getStats();
  const topConcepts = Array.from(hebbian.concepts.entries())
    .sort((a, b) => b[1].accessCount - a[1].accessCount)
    .slice(0, 10)
    .map(([kw, c]) => ({ keyword: kw, accessCount: c.accessCount, weight: c.weight }));
  return res.json({ ...stats, topConcepts });
});

module.exports = router;
