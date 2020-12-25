# Cocke-Kasami-Younger (CKY) algorithm

## Introduction

This assignment implements the CKY algorithm for bottom-up CFG parsing and applies it to the word and the parsing problem of English.
Developed in 1960, the CKY algorithm is the most used chart parser for CFGs (context-free grammars ) in CNF (Chomsky normal-form). It uses a dynamic programming algorithm to tell whether a string is in the language of grammar.

## Requirements

- Python: `3.8.3 [GCC 7.3.0]`
- NLTK: `3.5`
- Texttable: `texttable==1.6.3` (`pip install texttable`)

## Project file structure

```
├── atis
│   ├── atis-grammar-cnf.cfg
│   ├── atis-grammar-original.cfg
│   ├── atis-test-sentences.txt
│   └── other_bad_sentences.txt
├── cky.py
├── README.md
└── results
    ├── summary_bad_sentences.txt
    ├── summary_tree_counts.txt
    ├── ten_sents_cyk_chart.txt
    └── ten_sents_parsed_trees.txt
```

## Usage

- **Help**: for instructions on how to run the script with appropriate arguments.

  ```
  python cky.py --help
  ```

- **Run CYK parser**: Given CNF grammar and set of test sentences, check if these sentences are in the language of grammar and also display counts of all possible CKY parsed tress.

  ```
  python cky.py atis/atis-grammar-cnf.cfg atis/atis-test-sentences.txt
  ```

- Run and test the parser on some self-made sentences that are ungrammatical (i.e. not in the language of given CFG)

  ```
  python cky.py atis/atis-grammar-cnf.cfg atis/other_bad_sentences.txt
  ```

## Runtime

- **Total** runtime: 20.51 s
- **CYK parser** runtime: 17.76 s
- **Backpointer** runtime: 0.015 s

However, if you use optional arguments `-show_chart` or `-show_tree`, the total runtime is as follows:

- Total runtime: `-show_chart`: 23.67 s
- Total runtime: `-show_tree`: 285.27 s

## Results Contents

- `summary_tree_counts.txt`: Summary table of given ATIS test set with 98 sentences. Display if the sentence is in the language of CFG and counts of all possible CYk parse trees.
- `ten_sents_parsed_trees.txt`: Shows CYK parsed trees of the first 10 sentences from the ATIS test-set
- `ten_sents_cyk_chart.txt`: Shows CYK chart of first 10 sentences from the ATIS test-set
- `summary_bad_sentences.txt` : (Summary table of some self-made sentences) Shows if the sentences are in the language of CFG and counts of the parse tree for each.

## Glimpse of results

While all results can be found in `results`, here is a glimpse:

- CYK tree

```
( 1 ) show availability .

Given sentence is in the language of CFG

              SIGMA
   ┌────────────┴───────────┐
   │                       JWB
   │            ┌───────────┴──────────┐
NOUN_NN      AVPNP_NN             pt_char_per
   │            │                      │
  show     availability                .

            SIGMA
  ┌───────────┴───────────┐
  │                      JXI
  │           ┌───────────┴──────────┐
NP_NN      NOUN_NN              pt_char_per
  │           │                      │
 show    availability                .

              SIGMA
   ┌────────────┴───────────┐
   │                       JKA
   │            ┌───────────┴──────────┐
VERB_VB       NP_NN               pt_char_per
   │            │                      │
  show     availability                .

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
```

- CYK chart

```
( 1 ) show availability .

Given sentence is in the language of CFG

+---+-----------------------------+----------------------------+---------------+
|   |            show             |        availability        |       .       |
+===+=============================+============================+===============+
|   | 1                           | 2                          | 3             |
+---+-----------------------------+----------------------------+---------------+
| 4 | {DDO, GIH, AXQ, GNY, JKA,   | {DDO, GIH, DDC, JKA, BDV,  | {pt_char_per} |
|   | BDV, GKM, DDC, CYP, DTG,    | DTG, CYP, DRN, JVX, GTF,   |               |
|   | DRN, GTF, BEG, BGD, GXV,    | GXV, BQP, BRN, GIB, HMN,   |               |
|   | GTW, AWW, CVX, BQP, DEJ,    | AYT, GQX, DRT, DMR, GHV,   |               |
|   | BRN, DPN, GIB, AYT, GQX,    | GCT, GNZ, EBU, AWP, GOT,   |               |
|   | DRT, DMR, GIC, GNJ, GHV,    | AXB, GGF, LP, DBY, OR,     |               |
|   | GCT, GNZ, DRS, CYJ, EBU,    | HMV, SD, CPY, JIC, DNV,    |               |
|   | OR, GOT, AWP, LP, GGF, DBY, | JXI, GMP, GPB, GKR, DLT,   |               |
|   | DRM, DNI, AXB, AYL, GOU,    | CWK, BJF, GEB, HEZ, SIGMA, |               |
|   | SD, JCV, CPY, DOS, JIC,     | GTX, GFT, DSO, DZG, DNA,   |               |
|   | DNV, GMP, GPB, KQG, GKR,    | GFL, FFF, GCP, AQX, DOT,   |               |
|   | DLT, JLA, AVB, CWK, BJF,    | DNJ, GUX, AQT, CFV, GIR,   |               |
|   | DLS, DBX, GEB, DAR, HEZ,    | JWB, DQY, JXV, AXR, GJD,   |               |
|   | SIGMA, ATV, GTX, CXC, GFT,  | GUH, CZW, GJX, HOV, GOV,   |               |
|   | DSO, DZG, DNA, GFL, FFF,    | AWX, GCX, GTP, GOP, GVX,   |               |
|   | GCP, DOJ, KSE, GGL, AQX,    | GKV, GLF, DFE, BCP, APR,   |               |
|   | DOT, DNJ, GUX, CFV, AUT,    | CVJ, JIH, BGL, CZN, GWZ,   |               |
|   | GIR, AQT, JWB, ARQ, DQY,    | DDF, AYP, APN, DEY, DOK,   |               |
|   | KSH, AXR, GJD, BGW, GUH,    | DXK, EME, DIQ, GDH, BKL,   |               |
|   | GTN, CZW, GJX, DAX, GNK,    | JYF, GEN, JVI, JVN, EKO,   |               |
|   | GOV, AXY, AWX, AOP, GCX,    | ALB, DGI, CQN, JJO, EYE,   |               |
|   | GVX, GTP, GKQ, GOP, AVV,    | CZQ, AXZ, BY, GKN, ARR,    |               |
|   | GKV, GLF, AWO, BCP, DFE,    | GJJ, BHV, DAU, GIX, HKZ,   |               |
|   | APR, CVJ, JIH, BGL, DPK,    | DPL, GIL, GBT, JUO, GID,   |               |
|   | CZN, GWZ, AWT, DDF, AYP,    | JR, BGX, CVP, DBP, JYI,    |               |
|   | GUW, APN, DEY, DOK, GLW,    | DHJ, CTT, HEX, GDP, GRN,   |               |
|   | DXK, GOS, AVR, BDU, DQX,    | DRW, GGB, GVH, EDB, JIK,   |               |
|   | DIQ, JEF, CWZ, GDH, GVG,    | CSY, HOR, CVM, DAC, JXN,   |               |
|   | BKL, GUQ, HKY, HIV, AOX,    | GSL, EA, BHR, CZH, BEH,    |               |
|   | DBO, GHJ, GEN, ALB, EKO,    | DPC, CTW, GLX, DRZ, ALN,   |               |
|   | DGI, CQN, JJO, FXF, EYE,    | GWF, NS, ANF, JXZ, HNH,    |               |
|   | BY, AVE, CZQ, AXZ, GKN,     | HPB, DFZ, ESZ, GHX, GNL,   |               |
|   | ARR, GJJ, GUG, DIP, KSL,    | GPV, CVY, AAA, AVF, NP_NN, |               |
|   | IMPR_VB, FFL, DPB, CXO,     | DBG, DPO, GOR, HPH, GCZ,   |               |
|   | BHV, DAU, GIX, DPL,         | GJH, GUR, DAI, DXB, ANR,   |               |
|   | INFCL_VB, DSL, DFY, JJU,    | JVU, RJ, CZK, DPI, JJM,    |               |
|   | NR, AXA, GIL, IZP, GHP,     | GFB}                       |               |
|   | FFR, DES, GBT, BHQ, CYG,    |                            |               |
|   | AVJ, DNU, GKU, GID, JR,     |                            |               |
|   | DLE, BGK, CXX, BGX, ESY,    |                            |               |
|   | DBP, CVP, DMZ, JJI, AYO,    |                            |               |
|   | CYD, DHJ, CTT, IZL, OH,     |                            |               |
|   | HEX, GOQ, GDP, KPS, BGT,    |                            |               |
|   | HOQ, GRN, DRW, GGB, GOO,    |                            |               |
|   | GVH, EDB, HWZ, GGR, JIK,    |                            |               |
|   | CSY, NN, CVM, DAC, GJR,     |                            |               |
|   | GSL, EA, BHR, CZH, BEH,     |                            |               |
|   | DPC, CTW, IEX, HPG, ALN,    |                            |               |
|   | GLX, DRZ, VP_VB, CYA, GWF,  |                            |               |
|   | NS, DMQ, ND, FFI, ANF, DFZ, |                            |               |
|   | GNL, GHX, GPV, CVY, AAA,    |                            |               |
|   | DGH, GGV, DDN, GLE, JJN,    |                            |               |
|   | AVF, NP_NN, DBG, FTT, GOR,  |                            |               |
|   | DAI, AQH, GCZ, GJH, AXN,    |                            |               |
|   | GUR, DPO, ANR, GGJ, DXB,    |                            |               |
|   | GNX, RJ, CXU, GMO, DEX,     |                            |               |
|   | DPI, DXW, CZK, DLN, JJM,    |                            |               |
|   | GFB, AUX}                   |                            |               |
+---+-----------------------------+----------------------------+---------------+
| 3 | {AVPNP_NN, KPS, JYA, JXJ,   | {pt_noun_nn, NOUN_NN,      | .             |
|   | JGT, KSE, FII, FXF, KSH,    | NP_NN, AVPNP_NN, SIGMA}    |               |
|   | IEX, KSL, FFL, FYZ, VP_VB,  |                            |               |
|   | HLL, IMX, FYX, INFCL_VB,    |                            |               |
|   | HLB, IZP, JCV, FFR, IKJ,    |                            |               |
|   | HGF, NP_NN, KQG, JVK, FTT,  |                            |               |
|   | FMS, JLA, KIY, JVA, IKB,    |                            |               |
|   | IZL, JEF, SIGMA, JUR}       |                            |               |
+---+-----------------------------+----------------------------+---------------+
| 2 | {NOUN_NN, VERB_VB, VP_VB,   | .                          | .             |
|   | NP_NN, AVPNP_NN, SIGMA,     |                            |               |
|   | show, INFCL_VB}             |                            |               |
+---+-----------------------------+----------------------------+---------------+

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
```
