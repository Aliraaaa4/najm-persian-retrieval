# Parser Evaluation Plan

## 1. Goal

The parser converts selected OpenITI text versions into structured,
traceable, and loss-preserving blocks for later passage construction
and semantic retrieval.

The parser must preserve the source structure without silently dropping
or inventing content.

## 2. Supported profiles

### structured_poetry

Used for:

- Baba Afzal, Diwan
- Jalal al-Din Rumi, Diwan
- Jalal al-Din Rumi, Mathnawi

Important structures:

- page markers
- milestones
- headings
- verses
- hemistichs
- Mathnawi daftar boundaries

### mixed_prose_ocr

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
- Majalis council boundaries

### raw_ocr_reference

Used for:

- Ibn As'ad Hanati, Masalik Kraken

This profile is parsed conservatively. Definite markers are extracted,
but uncertain headings and paragraph boundaries are not aggressively
inferred.

## 3. Image-reference policy

Image-reference markers are preserved because they connect OCR text to
the source page image.

The first project version does not:

- download images
- store image files
- embed images
- use multimodal models
- insert image markers into retrieval text

The parser does:

- preserve the raw image marker
- extract an image identifier or URL when possible
- keep the source line and character offsets
- attach the active image reference to following content blocks
- report malformed image references

## 4. Output principles

### Loss preservation

Every body line must be covered exactly once by a parsed block.

Unrecognized content must be stored as a RAW block and accompanied by
a diagnostic instead of being deleted.

### Text representations

Each block contains:

- raw_text: exact source text
- display_text: text suitable for API display
- retrieval_text: controlled text used later for embeddings

Structural markers have an empty retrieval_text.

### Source traceability

Source line numbers are one-based and inclusive.

Character offsets are zero-based and half-open:

    [char_start, char_end)

### Determinism

The same file and parser configuration must always produce the same
ordered blocks and serialized output.

## 5. Candidate approaches

### Candidate A: flat_regex

Each line is classified independently using ordered regular expressions.

Purpose:

- simple baseline
- fast execution
- easy debugging

Expected weakness:

- poor context handling
- weak OCR paragraph reconstruction
- weak verse pairing

### Candidate B: profile_state_machine

The parser maintains document context:

- current page
- current image
- current heading
- current section
- current daftar
- current council
- previous block type

Expected strength:

- correct structural attachment
- better verse and paragraph boundaries
- deterministic and explainable

### Candidate C: buffered_hybrid

A profile-aware state machine with limited look-ahead and line buffering.

Purpose:

- resolve ambiguous OCR line continuations
- avoid false headings
- improve paragraph boundaries

Candidate C is implemented only if Candidate B shows measurable
weaknesses on the golden dataset.

## 6. Hard gates

A parser candidate is rejected if any of these conditions fail:

- no crash on all seven configured versions
- body-line coverage equals 100 percent
- overlapping source spans equal zero
- marker preservation equals 100 percent
- output is deterministic
- header metadata is excluded from body blocks
- raw source files remain unchanged

## 7. Quality metrics

- block-type precision, recall, and F1
- boundary precision, recall, and F1
- verse-pair accuracy
- orphan-verse-line rate
- heading precision, recall, and F1
- page-attachment accuracy
- image-attachment accuracy
- section-attachment accuracy
- raw or unknown line rate
- reconstruction accuracy
- runtime
- peak memory

## 8. Weighted score

Only candidates passing every hard gate receive a final score.

- block-type F1: 30 points
- boundary F1: 25 points
- verse and paragraph structure: 15 points
- page, image, heading, and section attachment: 15 points
- raw or unknown rate: 10 points
- runtime and peak memory: 5 points

Total: 100 points.

If a more complex candidate improves the score by less than one point,
the simpler candidate is preferred.

## 9. Golden dataset plan

The golden dataset must include approximately 28 representative samples:

- Baba Afzal, Diwan: 3 samples
- Rumi, Diwan: 5 samples
- Rumi, Mathnawi: 6 samples, one from each daftar
- Masalik AOCP: 4 samples
- Majalis-i Sab'a: 4 samples
- Akhlaq-i Muhtashami: 4 samples
- Masalik Kraken: 2 samples

Each sample should contain approximately 40 to 70 source lines and
include difficult structures when possible.

## 10. Known full-document regression counts

These counts are regression expectations and do not replace manual
golden evaluation.

- Baba Afzal, Diwan:
  - page markers: 201
  - headings: 201
  - verses: 484
  - milestones: 27

- Masalik AOCP:
  - page markers: 380
  - image references: 380
  - raw headings: 151
  - milestones: 313

- Masalik Kraken:
  - page markers: 637
  - standard headings: 0
  - milestones: 449

- Rumi, Diwan:
  - page markers: 5269
  - headings: 5269
  - verses: 40164
  - milestones: 2105

- Majalis-i Sab'a:
  - page markers: 123
  - image references: 123
  - raw headings: 69
  - councils: 7
  - milestones: 118

- Rumi, Mathnawi:
  - daftar sections: 6
  - page units: 972
  - verses: 25635
  - milestones: 1130

- Akhlaq-i Muhtashami:
  - page markers: 572
  - image references: 571
  - raw headings: 333
  - milestones: 347

## 11. Selection process

1. Create and review golden samples.
2. Implement and evaluate flat_regex.
3. Implement and evaluate profile_state_machine.
4. Inspect errors by work and profile.
5. Implement buffered_hybrid only when justified.
6. Select the simplest candidate that passes all hard gates and provides
   the best reliable quality.