# Dafny encoder

A Dafny function's signature includes 
function name, 
input arguments (and their types),
requires clauses (assumptions on the function's input, aka precondition), and
ensures clauses (properties of the function's output, aka postcondition).

Based on a function's signature,
our model recommends the most relevant Dafny functions.




# Data Collection

While there are a few dataset for Dafny,
the size of dataset is small and/or the contained dafny functions are tiny.

As a part of this project, I developed a data collection tool at this repository: 
https://github.com/chanheec/dafny-extractor





# Training Methodology

At a high level, this project's methodology resembles to [CodeBERT paper](https://arxiv.org/abs/2002.08155)'s methodology.
CodeBERT is a bimodal pre-trained model for programming language and natural language `~\cite{CodeBERT}`.

CodeBERT uses a program and its comment (natural language).
Comment is essentially a specification of a function written in natural language.
Proof goal (precondition and postcondition) is a specification of a function written in mathematical formula.

At a high level, instead of (comment, program) pair, we utilize (proof goal, proof) pair.



##  Pre-training for Dafny using masked language modeling (MLM)

We train our model based on `codebert-base-mlm` (https://huggingface.co/microsoft/codebert-base-mlm).

First, as in CodeBERT/Roberta, we merge the input pair (proof goal, program) as a single string using separator.
Next, we randomly select 15% of the tokens in a data.
For each selected token, it is replaced with [MASK] (80% probability),
replaced with a random token (10% probability), 
or kept unchanged (10% probability).

The model is trained with the goal of predicting the original token of 15% of selected tokens.


<!-- TODO: consider using special charactor as in CodeBERT
[CLS], w1, w2, ..wn, [SEP ], c1, c2, ..., cm, [EOS]. One segment is natural lan- guage text, and another is code from a certain pro- gramming language. [CLS] is a special token in front of the two segments, whose final hidden repre- sentation is considered as the aggregated sequence representation for classification or ranking. Follow- ing the standard way of processing text in Trans- former, we regard a natural language text as a se- quence of words, and split it as WordPiece (Wu et al., 2016). We regard a piece of code as a se- quence of tokens. -->

See `mlm.py` for more details.




## Task-specific training (fine-tuning)

We (currently) have only one task, which is lemma (function) recommendation.
For this recommendation task, we fine-tune the pre-trained model using classification task.


For training, I simply utilized each function to connect proof goal and proof body.
The rationale is that, a function's proof goal is clearly related to the function's proof body.


TODO: Write how I did it


We utilized our model as a unified encoder (embedding query and candidate as a whole).


See `recommend.py` for more details.








# Evaluation

1. For each proof in library, extract function signature, which contains preconditions and postconditions.


2. Based on the goal, get top 10 recommend lemmas out of 1000 lemmas
We utilized our model as a unified encoder (embedding query and candidate as a whole).




3. Check if there is a lemma utilized in the proof body
4. Successful recommendation for 47% of proofs 


  
Q: CodeBERT-mlm, CodeBERT, DafnyBERT-mlm, DafnyBERT-rec  before dafny training    


It seems that there is no prior work on Dafny regarding lemma/function recommendation.


See `recommend-quant.py` for more details.



# Discussion: embedding proof goal and lemma: together VS separately

As in CodeBERT, we utilized our model as a unified encoder (embedding query and candidate as a whole).
However, we can consider applying "late fusion" (i.e., pre-calculate all embedding for all lemma, and only embed query upon request, and then calculate cosine similarity).
By embedding proof goal and lemma in a shared vector space, 
the response can be a lot faster. 
It might be the way to go if we want some IDE integration. 



# Future Work

- Contrastive learning for semantic-preserving transformations (augmentations)
- Graph-based predicate embedding (cf. https://arxiv.org/abs/2009.08366)
 

