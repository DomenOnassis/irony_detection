# Long text parsing

This directory includes files which are used to parse and normalize long form sentences into a standardized format which is used
for irony clasification.

## Methodology
- Long text is split into normalized sentences (text between two dots and starts with uppercase, 
the punctuations and special characters were left in the text as they provide usefull semantics for irony detection)
- Word n-grams are parsed from sentences and stored inside .txt files (each corpus has its own folder and one file per sentence). Those files will be used for irony classification.

### .txt file format
This is a test sentence for my project. This is another one :).

**2-gram** example

File: sentence_1.txt
```txt
sentence for
for my
my project
project .
. This
This is
is another
another one
one :)
:) .
```

File: sentence_2.txt
```txt
This is
is another
another one
one :)
:) .
```
