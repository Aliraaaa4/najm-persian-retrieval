# Engineering Decisions

## 1. Objective and scope

This project implements a local retrieval system over a selected subset of the
OpenITI PER0675AH corpus. The indexed scope contains works by Baba Afzal, Ibn
As'ad Hanati, Jalal al-Din Rumi, and Nasir al-Din Tusi.

The primary goal was not to build a generic keyword search engine or a
generative question-answering system. The goal was to retrieve relevant source
passages from historical Persian and Islamic texts and return them with clear
provenance, including author, work, version, document identifier, passage
identifier, and a readable snippet.

The submitted system is designed for local execution, uses only open-source
software and models, and requires no paid service or proprietary API.

## 2. Development constraints

The project was developed on a Windows personal computer with an Intel
Core i5-13420H processor, 8 GB of RAM, and no required GPU. These constraints
strongly influenced the architecture.

The main design priorities were:

- CPU-compatible execution;
- bounded memory usage;
- deterministic and inspectable artifacts;
- simple local storage;
- reproducible tests;
- no external database server;
- no dependency on paid embedding or language-model APIs.

A larger embedding model, cross-encoder reranker, vector database, or
generative model could potentially improve some results, but would increase
memory use, latency, installation complexity, and operational risk on the
target hardware.

## 3. Overall architecture

The system is divided into independent layers:

1. Corpus manifest and source-file audit
2. OpenITI parsing and metadata extraction
3. Text normalization and structural analysis
4. Logical-unit and passage construction
5. Lexical index construction
6. Dense embedding artifact construction
7. Hybrid retrieval
8. Scope, paratext, and answerability analysis
9. Trusted abstention
10. Passage-store enrichment
11. FastAPI REST service
12. Persian right-to-left demonstration interface

This separation makes it possible to test parsing, indexing, retrieval,
decision logic, storage, and API behavior independently.

The runtime service loads three main artifacts:

- a SQLite FTS5 lexical index;
- a SQLite passage store;
- a dense artifact containing normalized NumPy embeddings and passage
  metadata.

The API returns source passages only. It does not ask a large language model to
generate an answer, which reduces hallucination risk and keeps retrieved
evidence directly inspectable.

## 4. Corpus selection and version handling

The project uses the specified OpenITI subset rather than indexing the entire
collection. A corpus manifest records the selected works and versions.

OpenITI repositories may contain multiple text versions with different
quality levels. Where relevant, one version was treated as the primary
canonical source and another version was retained as a reference-only source.
This avoided silently mixing incompatible versions while preserving useful
material for audit and comparison.

The parser keeps document and version identifiers throughout the pipeline so
that every public result can be traced back to its source.

## 5. Data preprocessing

Historical OpenITI texts contain YAML metadata, mARkdown annotations,
headings, poetry markers, prose, Arabic quotations, Qur'anic citations,
paratext, and in some cases OCR noise.

The parser was therefore designed to be conservative and structure-aware.
Instead of aggressively stripping every marker, the pipeline first separates
metadata, primary text, structural markers, and non-primary material.

Unicode normalization standardizes common Persian and Arabic character
variants and spacing behavior while avoiding destructive normalization of
historically meaningful content. Arabic passages are preserved rather than
removed, because they may be genuine parts of the selected Persian works.

Parsing behavior was evaluated on representative samples covering structured
poetry, mixed prose, and noisy OCR-like text. Separate parser evaluation
reports and golden-sample guidance are included in the repository.

## 6. Passage construction

A fixed character-window splitter would be simple, but it can break verses,
headings, sentences, and semantically connected units. The implemented
pipeline first identifies logical units and structural boundaries and then
builds retrieval passages and contextual windows from those units.

This approach improves result readability and source presentation. It also
allows the system to label passages by kind, including primary text, mixed
content, and paratext-related material.

The trade-off is additional preprocessing complexity. Boundary rules must be
tested against different OpenITI structures and cannot guarantee perfect
segmentation for every possible document in the wider corpus.

## 7. Lexical retrieval

Lexical retrieval uses SQLite FTS5. SQLite was selected because it is
open-source, available locally, easy to distribute, and sufficient for the
size of the selected corpus.

Lexical retrieval is valuable for:

- exact names and titles;
- distinctive historical terms;
- quotations;
- rare words;
- cases where exact token overlap is important.

A server-based search platform such as Elasticsearch would provide more
operational features and scaling options, but would be unnecessarily heavy for
the target corpus and personal-computer requirement.

## 8. Dense retrieval

Dense retrieval uses the open-source
`intfloat/multilingual-e5-small` sentence-embedding model.

This model was chosen as a compromise between multilingual semantic quality,
CPU usability, memory requirements, and artifact size. Passage embeddings are
stored as normalized NumPy arrays and query-time ranking uses cosine
similarity.

Exact NumPy search is acceptable for the current corpus size and keeps the
runtime simple. For a substantially larger corpus, an approximate nearest
neighbor index such as FAISS or HNSW would be more appropriate.

The model is multilingual rather than specifically trained for classical
Persian, historical orthography, Persian poetry, mixed Persian-Arabic text, or
OpenITI OCR noise. This is an important quality limitation.

## 9. Hybrid ranking

Neither lexical nor dense retrieval is sufficient on its own.

Lexical search can miss semantically relevant passages when the query and
source use different wording. Dense retrieval can overlook exact source names,
rare terminology, or precise quotations and may retrieve semantically broad
but insufficiently supported passages.

The system therefore combines lexical and dense candidates using weighted
reciprocal-rank fusion. Rank fusion was selected because lexical BM25 values
and dense cosine scores are not directly comparable or calibrated on the same
scale.

This hybrid design provides a practical balance between exact matching and
semantic similarity without requiring a learned ranking model.

A lightweight cross-encoder reranker was considered but not included because
it would increase latency, RAM use, model downloads, and deployment
complexity on the target machine.

## 10. Trusted abstention

A retrieval system should not return confident-looking results when the
available evidence is weak or when the question explicitly refers to a source
outside the indexed corpus.

The system therefore includes an abstention layer. Its decision uses evidence
such as:

- explicit in-corpus and out-of-corpus author or work mentions;
- source-attribution conflicts;
- lexical and dense agreement;
- hybrid evidence strength;
- top-result passage kind;
- paratext or mixed-content risk.

When evidence is insufficient, the public API returns an abstention response
with no public result passages. Diagnostic candidates remain internal and are
not exposed as trusted evidence.

This design favors precision and source trust over maximum answer coverage.
The cost is that some difficult but potentially answerable queries may be
rejected.

## 11. Query suggestions

For abstained requests, the API may return deterministic query suggestions.
Suggestions are generated only from the frozen corpus scope catalog.

They do not override the abstention decision, do not expose diagnostic
passages, and do not invent works outside the indexed collection. Their purpose
is to help the user reformulate a query toward an available author, work, or
primary-text scope.

The current implementation is rule- and catalog-based. It does not yet include
fuzzy entity resolution, typo correction, or learned query rewriting.

## 12. Evaluation strategy

Evaluation was separated from implementation as much as possible.

The project includes:

- parser golden samples;
- parser pilot evaluation;
- manually adjudicated retrieval queries;
- lexical, dense, and hybrid comparisons;
- answerable and out-of-corpus cases;
- hard negatives and source-attribution traps;
- abstention calibration and frozen validation;
- regression tests for API and runtime behavior.

Calibration and validation records were kept separate so that abstention
thresholds were not repeatedly tuned against the final validation set.

The current evaluation set is still small relative to the possible range of
historical Persian questions. It does not replace a larger multi-annotator
relevance study.

## 13. Runtime delivery and reproducibility

Large generated indexes are not committed to Git. The validated runtime
artifacts are distributed as a GitHub Release ZIP with a SHA-256 checksum.

The bundle includes the lexical database, passage store, dense embeddings,
metadata, passage records, and evaluation metadata. Paths inside the ZIP are
portable rather than Windows-specific.

The first dense query on a new computer may require downloading and caching
the open-source embedding model. After the model is cached, the system can run
with local-files-only mode enabled.

The repository includes commands for corpus auditing and parsing, deterministic
builders for the lexical index and passage store, and a runtime-bundle builder.
However, the current version does not yet expose the entire raw-corpus-to-dense-
artifact pipeline as one public end-to-end CLI command. The submitted release
therefore prioritizes a verified runtime bundle for reliable handoff.

## 14. Testing and software quality

The project currently contains 451 automated tests. They cover parsing,
normalization, passage construction, index behavior, hybrid retrieval,
abstention, passage storage, runtime configuration, API responses, and static
UI delivery.

A fresh-clone test was also performed in a new virtual environment. Installing
the project in editable mode and running the full test suite produced 451
passing tests.

SQLite artifact validation, dense-artifact hashes, corpus identifiers, schema
versions, and runtime paths are checked before serving retrieval requests.

## 15. Challenges encountered

The main engineering challenges were:

- inconsistent structures across OpenITI text versions;
- preserving poetry and prose boundaries;
- handling mixed Persian and Arabic material;
- separating primary text from paratext;
- avoiding overconfident results for out-of-corpus questions;
- comparing lexical and dense systems with different score scales;
- producing a portable runtime artifact on Windows;
- keeping memory and installation requirements suitable for an 8 GB machine;
- distinguishing reproducible source code from large generated artifacts.

These challenges led to conservative parsing, explicit manifests, hybrid
ranking, trusted abstention, deterministic packaging, and extensive validation.

## 16. Known limitations

The current version has the following limitations:

- the corpus scope is fixed to the selected works;
- the dense model is not fine-tuned for classical Persian;
- exact dense search will not scale efficiently to millions of passages;
- no cross-encoder reranker is used;
- fusion weights and abstention thresholds are manually calibrated rather than
  learned;
- the human-labeled evaluation set is limited;
- parsing rules may not cover every OpenITI tagging variation;
- indexing is not incremental;
- the first run may need internet access to download the embedding model;
- the build pipeline is not yet available as one end-to-end command;
- the API has no authentication, rate limiting, or production observability;
- the UI is a demonstration interface rather than a complete search product;
- full cross-platform runtime testing has primarily been performed on Windows.

## 17. What I would improve with more time

The first priority would be reproducibility and build automation:

1. Add one configuration-driven command that audits, parses, normalizes,
   chunks, embeds, indexes, validates, and packages the corpus.
2. Expose dense-index construction as a supported CLI.
3. Add automated release-asset verification in CI.
4. Add Windows, Linux, and macOS test jobs.
5. Provide Docker and an optional fully offline model package.

The second priority would be retrieval quality:

1. Expand the human-labeled evaluation set.
2. Use multiple annotators and measure inter-annotator agreement.
3. Add a lightweight cross-encoder reranker.
4. Fine-tune or adapt embeddings for historical Persian.
5. Perform hard-negative mining.
6. Calibrate fusion and abstention on larger frozen splits.

The third priority would be scale and maintainability:

1. Introduce FAISS or HNSW for larger corpora.
2. Support incremental and content-addressed indexing.
3. Regenerate embeddings only for changed passages.
4. Add atomic artifact switching and rollback.
5. Track complete artifact lineage across corpus versions.

For production deployment, I would also add authentication, rate limiting,
structured logs, metrics, tracing, pagination, work and author filters,
highlighting, accessibility review, and monitoring.

## 18. Summary

The final design intentionally prioritizes trustworthy source retrieval,
local execution, reproducibility, and maintainability over maximum model size
or generative functionality.

The system demonstrates that a small, CPU-compatible hybrid architecture can
provide useful semantic retrieval over historical Persian texts while
remaining inspectable and explicit about uncertainty.
