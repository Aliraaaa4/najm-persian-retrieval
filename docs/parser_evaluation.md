# Parser Evaluation Plan

## 1. Goal

The parser converts selected OpenITI text versions into structured,
traceable, deterministic, and loss-preserving blocks for later passage
construction and semantic retrieval.

The parser must preserve the exact source body while extracting useful
document structure such as:

- page markers
- image references
- milestones
- headings
- sections
- verses
- paragraphs

The parser must never silently delete, rewrite, or invent source
content.

## 2. Supported parsing profiles

### 2.1 structured_poetry

Used for:

- Baba Afzal, Diwan
- Jalal al-Din Rumi, Diwan
- Jalal al-Din Rumi, Mathnawi

Important structures:

- page markers
- milestones
- poem or genre headings
- verses
- hemistich separators
- continuation lines
- Mathnawi daftar boundaries

Important characteristics:

- a verse usually contains the `%~%` separator
- one verse may continue onto a following line beginning with `~~`
- page and milestone markers may appear at the end of a verse or
  continuation line
- headings usually begin with `###`

### 2.2 mixed_prose_ocr

Used for:

- Ibn As'ad Hanati, Masalik AOCP
- Jalal al-Din Rumi, Majalis-i Sab'a
- Nasir al-Din Tusi, Akhlaq-i Muhtashami

Important structures:

- page markers
- image references
- milestones
- headings
- paragraphs
- OCR continuation lines
- quoted poetry
- Majalis council boundaries
- inline structural markers

Important characteristics:

- one logical paragraph may span several physical source lines
- lines beginning with `#` are not necessarily verses
- quoted poetry is identified more reliably by `%~%`
- page, image, and milestone markers may appear inside content lines
- OCR noise must not be silently corrected by the parser

### 2.3 raw_ocr_reference

Used for:

- Ibn As'ad Hanati, Masalik Kraken

This profile is parsed conservatively.

The parser extracts definite structures such as:

- page markers
- milestones
- blank lines
- raw text regions

The parser must not aggressively infer headings or paragraph boundaries
when evidence is weak.

This version is retained as a reference version and is not included in
the main semantic index.

## 3. Shared parser architecture

All parser candidates must use the same output contract and shared
OpenITI core.

The architecture consists of:

- source reader
- header and body boundary detector
- character-offset tracker
- marker extractor
- diagnostic collector
- exact reconstruction checker
- profile-specific handler

Profile-specific handlers are:

- structured poetry handler
- mixed prose and OCR handler
- conservative raw OCR handler

A separate complete parser is not created for every book. Book-specific
rules must remain small, explicit, and configurable.

## 4. Character-level loss preservation

### 4.1 Primary lossless unit

OpenITI page and milestone markers may appear inside content lines.

For example, a source line may contain both text and a page marker:

    ~~تیز را PageV1P001

A source line may also contain a page marker and milestone:

    PageV1P5269 ms2105

Therefore, the primary unit of lossless coverage is the source
character span, not the physical source line.

### 4.2 Coverage rule

Every source-body character, including newline characters, must be
covered exactly once by the ordered parsed blocks.

Multiple parsed blocks may share the same physical source line when
their character spans do not overlap.

For an inline-marker line, the parser may produce blocks such as:

1. content fragment
2. page marker
3. milestone marker
4. remaining content or newline

### 4.3 No overlap

Two blocks may share line numbers, but their character spans must not
overlap.

The following condition is mandatory:

    overlapping source characters = 0

### 4.4 No uncovered content

The following condition is mandatory:

    uncovered source characters = 0

Unknown or ambiguous content must be stored as a RAW block and
accompanied by a diagnostic.

Unknown content must never be silently removed.

### 4.5 Exact reconstruction

When all block `raw_text` values are concatenated in source order, the
result must be exactly equal to the original source body:

    parsed_document.reconstruct_body() == original_source_body

Exact reconstruction includes:

- original characters
- original spaces
- original marker text
- original line endings
- blank lines
- malformed OCR text

## 5. Header and body policy

The OpenITI header is read for metadata but is not emitted as body
content blocks.

The source body begins immediately after:

    #META#Header#End#

The parser must store:

- one-based first body line
- zero-based first body character offset

Header content must never appear in:

- display text
- retrieval text
- passage construction
- semantic embeddings

## 6. Source spans

Every parsed block must contain a `SourceSpan`.

Line numbering is:

- one-based
- inclusive at both ends

Character offsets are:

- zero-based
- half-open

The character interval is:

    [char_start, char_end)

For every block:

    raw_text == source_text[char_start:char_end]

Source spans are used for:

- exact reconstruction
- overlap detection
- uncovered-character detection
- debugging
- citations
- API traceability

## 7. Block types

All parser candidates emit the same block types:

- PAGE_MARKER
- IMAGE_REFERENCE
- MILESTONE
- HEADING
- SECTION
- VERSE
- PARAGRAPH
- BLANK
- RAW

A block must not be assigned more than one primary block type.

Additional information is stored in block attributes.

Examples of attributes include:

- heading level
- genre
- verse number
- first hemistich
- second hemistich
- daftar number
- council number
- milestone number
- continuation status

## 8. Text representations

Every block contains three text representations.

### 8.1 raw_text

The exact source substring.

It must never be normalized or corrected.

### 8.2 display_text

Text suitable for API display.

Display processing may remove structural prefixes or normalize
presentation whitespace, but it must not silently correct historical
spelling or OCR errors.

### 8.3 retrieval_text

Controlled text used later for passage construction and embeddings.

Structural-only blocks must have empty retrieval text.

These block types normally have empty retrieval text:

- PAGE_MARKER
- IMAGE_REFERENCE
- MILESTONE
- BLANK

Persian normalization is implemented separately from structural
parsing.

## 9. Marker extraction policy

Markers must be detected before content classification.

Marker priority is:

1. image reference
2. page marker
3. milestone
4. heading or section marker
5. verse or paragraph content

This priority applies even when multiple structures occur on the same
physical line.

### 9.1 Page markers

Page markers follow forms such as:

    PageV1P001
    PageV01P191
    PageV06P140

The parser must support variable-length volume and page numbers.

Recommended pattern:

    PageV(?P<volume>\d+)P(?P<page>\d+)

The raw marker must always be preserved.

### 9.2 Milestones

Milestones follow forms such as:

    ms27
    ms313
    ms2105

Recommended pattern:

    (?<![A-Za-z0-9_])ms(?P<number>\d+)(?![A-Za-z0-9_])

Milestones may appear:

- at the beginning of a line
- inside a heading
- inside prose
- after a page marker
- at the end of a document

The marker must be extracted without deleting surrounding content.

### 9.3 Image references

Image references commonly use Markdown image syntax:

    ![image filename](./page_image.png)

The parser must:

- preserve the complete raw marker
- extract the target path when possible
- extract an image identifier when possible
- retain source offsets
- attach the active image reference to following content blocks
- report malformed image references

## 10. Image-reference policy

Image references connect OCR text with the source page image.

The first project version does not:

- download images
- store image files
- embed images
- run OCR again
- use multimodal models
- include image markers in retrieval text

The first project version does:

- preserve image markers
- extract image metadata
- attach image context to content blocks
- expose image references for traceability
- use image-marker counts in parser regression tests

A missing image reference must not cause the parser to crash.

Page markers and image references are evaluated independently because
their counts are not guaranteed to be identical.

## 11. Context attachment

Content blocks may inherit structural context from earlier markers.

Possible context includes:

- current page
- current image
- current heading
- current section
- current daftar
- current council

A content block must be attached to the most recent applicable context
that precedes its first source character.

Context changes that occur inside a physical line must be applied from
the exact character position of the marker onward.

## 12. Diagnostics

The parser must produce explicit diagnostics for uncertain or malformed
input.

Example diagnostic codes include:

- unrecognized_content
- malformed_page_marker
- malformed_image_reference
- malformed_milestone
- orphan_verse_continuation
- verse_without_hemistich_separator
- heading_without_content
- unexpected_marker_order
- uncovered_source_characters
- overlapping_source_characters
- reconstruction_mismatch

Diagnostics must include a source span whenever possible.

A diagnostic does not automatically imply parser failure. Hard-gate
diagnostics such as overlap or reconstruction mismatch do imply failure.

## 13. Determinism

The same source file, configuration, and parser version must always
produce:

- the same ordered blocks
- the same source spans
- the same attributes
- the same diagnostics
- the same serialized output
- the same output hash

The structural parser must not depend on:

- remote APIs
- paid services
- nondeterministic LLM output
- random sampling

## 14. Candidate parsing approaches

### 14.1 Candidate A: flat_regex

Each source line is processed primarily through ordered regular
expressions.

Purpose:

- establish a simple baseline
- provide fast execution
- expose the value of document context
- remain easy to debug

Expected weaknesses:

- weak handling of inline markers
- weak verse continuation handling
- weak OCR paragraph reconstruction
- weak section and context attachment

### 14.2 Candidate B: profile_state_machine

The parser maintains document state such as:

- current page
- current image
- current heading
- current section
- current daftar
- current council
- previous content type

Expected strengths:

- better structural attachment
- better verse reconstruction
- better paragraph boundaries
- explainable deterministic behavior
- correct handling of profile differences

### 14.3 Candidate C: buffered_hybrid

A profile-aware state machine with limited look-ahead and line
buffering.

Possible uses:

- resolve OCR continuation lines
- reduce false heading detection
- improve paragraph boundaries
- resolve ambiguous verse continuations

Candidate C is implemented only when Candidate B shows a measurable and
specific weakness on the development golden dataset.

## 15. Hard gates

A parser candidate is rejected if any of the following conditions fail:

- no crash on all seven configured versions
- source-character coverage equals 100 percent
- uncovered source characters equal zero
- overlapping source characters equal zero
- exact source-body reconstruction succeeds
- all recognized page markers are preserved
- all recognized image references are preserved
- all recognized milestones are preserved
- OpenITI header content is excluded from body blocks
- repeated execution produces identical serialized output
- raw corpus files remain unchanged

Hard gates are evaluated before quality scoring.

A candidate that fails one hard gate cannot be selected, even when its
classification score is high.

## 16. Quality metrics

### 16.1 Block-type metrics

Measure precision, recall, and F1 for:

- heading
- section
- verse
- paragraph
- page marker
- image reference
- milestone
- blank
- raw

### 16.2 Boundary metrics

Measure whether the predicted block start and end character offsets
match the golden annotations.

Both exact boundaries and partial overlap may be reported, but final
selection prioritizes exact boundary correctness.

### 16.3 Marker metrics

Measure precision, recall, and F1 for:

- page markers
- image references
- milestones

Marker raw text and source offsets must also match.

### 16.4 Poetry metrics

Measure:

- verse detection accuracy
- hemistich extraction accuracy
- continuation-line attachment accuracy
- orphan continuation rate
- verse-number extraction accuracy
- daftar attachment accuracy

### 16.5 Prose and OCR metrics

Measure:

- paragraph-boundary accuracy
- heading precision, recall, and F1
- OCR continuation attachment accuracy
- council attachment accuracy
- false verse rate in prose
- RAW line and character rate

### 16.6 Context-attachment metrics

Measure:

- page attachment accuracy
- image attachment accuracy
- heading attachment accuracy
- section-path accuracy
- daftar attachment accuracy
- council attachment accuracy

### 16.7 Lossless metrics

Measure:

- character coverage
- uncovered characters
- overlapping characters
- exact reconstruction
- marker preservation
- deterministic output

### 16.8 Performance metrics

Measure:

- runtime
- peak memory usage
- blocks produced per second

Performance is secondary to correctness but remains important for
typical personal computers.

## 17. Weighted quality score

Only candidates passing every hard gate receive a final score.

Weights are:

- block-type F1: 30 points
- exact boundary F1: 25 points
- verse and paragraph structure: 15 points
- page, image, marker, and section attachment: 15 points
- RAW or unknown rate: 10 points
- runtime and peak memory: 5 points

Total:

    100 points

A lower RAW rate is better only when it does not increase false
classifications.

Conservative RAW output is preferred over incorrect invented structure.

## 18. Score aggregation policy

Large works such as Rumi's Diwan must not dominate the corpus score.

Scores are aggregated in three stages:

1. calculate metrics for each sample
2. calculate the macro average for each version
3. calculate the macro average across versions

The final corpus score is not calculated by placing every source line
from every work into one global pool.

This gives each configured version meaningful representation.

## 19. Profile-specific selection

The project is not required to select one identical handler for all
profiles.

The shared core and output contract remain identical, but the best
candidate may differ by profile.

For example:

- structured poetry may use a profile state machine
- mixed prose OCR may require buffered hybrid logic
- raw OCR reference may use a conservative state machine

Each profile-specific choice must be supported by the same evaluation
policy and golden annotations.

## 20. Golden dataset plan

The golden dataset contains approximately 28 representative samples:

- Baba Afzal, Diwan: 3 samples
- Rumi, Diwan: 5 samples
- Rumi, Mathnawi: 6 samples, one from each daftar
- Masalik AOCP: 4 samples
- Majalis-i Sab'a: 4 samples
- Akhlaq-i Muhtashami: 4 samples
- Masalik Kraken: 2 samples

Each sample should normally contain approximately 40 to 70 source
lines.

Samples should include difficult structures such as:

- inline page markers
- inline milestones
- heading boundaries
- verse continuation lines
- quoted poetry inside prose
- OCR-broken paragraphs
- page and image transitions
- daftar boundaries
- council boundaries
- noisy raw OCR

## 21. Development and holdout split

Golden samples are divided into:

- development samples
- holdout samples

Development samples are used to:

- inspect parser errors
- design rules
- tune regular expressions
- improve state transitions

At least one sample from each configured version is reserved for
holdout evaluation.

Holdout samples must not be used to tune parser rules.

After final holdout evaluation, changing parser rules requires recording
that the previous holdout evaluation is no longer final.

## 22. Known full-document regression counts

These counts are regression expectations and do not replace manual
golden evaluation.

### Baba Afzal, Diwan

- page markers: 201
- headings: 201
- verse marker lines: 484
- milestones: 27
- image references: 0

### Masalik AOCP

- page markers: 380
- image references: 380
- raw headings: 151
- milestones: 313

### Masalik Kraken

- page markers: 637
- standard headings: 0
- milestones: 449
- image references: 0

### Rumi, Diwan

- page markers: 5269
- headings: 5269
- verse marker lines: 40164
- milestones: 2105
- image references: 0

### Majalis-i Sab'a

- page markers: 123
- image references: 123
- raw headings: 69
- councils: 7
- milestones: 118

### Rumi, Mathnawi

- daftar sections: 6
- page markers: 972
- verse marker lines: 25635
- milestones: 1130
- image references: 0

### Akhlaq-i Muhtashami

- page markers: 572
- image references: 572
- raw headings: 333
- milestones: 347

Regression counts must be generated with explicit detector patterns and
stored alongside the evaluation report.

A count mismatch requires investigation but does not automatically prove
that the parser is wrong.

## 23. Selection process

The final parser selection process is:

1. create and review pilot samples
2. finalize the golden annotation schema
3. validate character-level golden coverage
4. expand the golden dataset
5. implement and evaluate `flat_regex`
6. implement and evaluate `profile_state_machine`
7. inspect errors separately by work and profile
8. implement `buffered_hybrid` only when justified
9. run final holdout evaluation
10. verify full-document regression counts
11. select the simplest candidate that passes every hard gate and
    provides the best reliable quality

## 24. Complexity preference

When two candidates both pass every hard gate, the simpler candidate is
preferred when the score improvement of the more complex candidate is
less than one point.

Example:

    profile_state_machine: 91.8
    buffered_hybrid: 92.3

The state-machine candidate is preferred because the improvement is
less than one point.

Example:

    profile_state_machine: 88.4
    buffered_hybrid: 92.1

The buffered candidate may be selected because the improvement is
meaningful, provided that:

- all hard gates pass
- the improvement also appears on holdout samples
- the improvement is not limited to one work
- runtime and memory remain acceptable