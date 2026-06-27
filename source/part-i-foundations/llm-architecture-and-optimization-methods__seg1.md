<!-- page 35 -->
Chapter 1
LLM Architecture and Optimization
Methods
This section covers the foundational architecture of large language models and the key optimization
techniques that make training and inference efficient. Topics are ordered as a curriculum: we begin
with the transformer itself, then cover how to train it efficiently, how to adapt it cheaply, how to
compress it, how to scale it, and how to accelerate its inference.
1.1
How LLMs Work: An Intuitive Overview
Before diving into architectural details, let us build intuition for how a large language model transforms
text into text. The entire process follows a simple pipeline: text →tokens →representations →
tokens →text.
Raw
Text
Tokenizer
Token
IDs
Embedding
Layer
Transformer
Layers (×L)
Vocab
Logits
Decode
Output
Text
autoregressive loop (append token to input)
Figure 1.1: The LLM pipeline: text is tokenized into subword units, converted to integer IDs, embedded as
dense vectors, processed through transformer layers, projected to vocabulary logits, and decoded back to text.
The dashed loop shows autoregressive generation—each output token is appended to the input for the next
forward pass.
The Four Key Stages
1. Tokenization: Raw text is split into subword pieces (not characters, not full words) using a
learned vocabulary. “unhappiness” might become [“un”, “happiness”] or [“unhapp”, “iness”].
2. Embedding: Each token ID indexes into a learned embedding table, producing a dense
vector in Rd (typically d = 4096). These vectors capture semantic meaning—similar words
get similar vectors.
3. Contextual Processing: The transformer stack processes all embeddings in parallel, using
self-attention to let each position “read” from all other positions. After L layers, each
position’s hidden state encodes rich contextual information.
4. Prediction: The final hidden state is projected to a probability distribution over the full
vocabulary, and a decoding strategy selects the next token.
1.2
Tokenization
Tokenization is the critical first step that converts raw text into the discrete symbols a language
model operates on. The choice of tokenizer directly affects model quality, multilingual capability,
and computational efficiency.
35


<!-- page 36 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Why Subwords?
Character-level models need very long sequences (expensive attention). Word-level models cannot
handle rare or novel words. Subword tokenization strikes the ideal balance: common words are
single tokens (“the” →[the]), rare words decompose into known pieces (“cryptocurrency” →
[“crypt”, “ocur”, “rency”]), and the vocabulary stays manageable (32K–128K tokens).
1.2.1
Why Not Characters or Words?
Table 1.1: Trade-offs of different tokenization granularities.
Granularity
Vocab Size
Seq Length
Issues
Character
∼256
Very long
Attention cost O(n2); hard to learn long-range semantics
Word
∼500K+
Short
Cannot handle rare/novel words; huge embedding table
Subword
32K–128K
Moderate
Best trade-off: short sequences, open vocabulary
1.2.2
Byte-Pair Encoding (BPE)
BPE [24] is the dominant tokenization algorithm used by GPT, Llama, Mistral, and most modern
LLMs.
BPE Algorithm
1. Start with a vocabulary of individual characters (bytes)
2. Count all adjacent symbol pairs in the training corpus
3. Merge the most frequent pair into a new symbol
4. Repeat steps 2–3 for k iterations (until desired vocabulary size)
Figure 1.2: BPE tokenization example: starting from characters, the algorithm iteratively merges the most
frequent adjacent pairs until the word becomes a single token or the vocabulary budget is exhausted.
36


<!-- page 37 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.2.3
Other Tokenization Methods
Table 1.2: Comparison of subword tokenization algorithms.
Method
Used By
Key Idea
BPE
GPT-4
[23],
Llama-
3 [25], Mistral [26]
Bottom-up merging of frequent pairs; de-
terministic
WordPiece
BERT
[27],
Distil-
BERT [28]
Similar to BPE but maximizes likelihood
of training data
Unigram LM
SentencePiece (T5 [29],
XLNet [30])
Top-down: start with large vocab, prune
by likelihood impact
Byte-level BPE
GPT-2 [31]+
BPE on raw bytes (no unknown tokens
possible); 256 base vocab
1.2.4
Tokenization Best Practices
1. Vocabulary size matters: 32K is minimal; 128K enables better multilingual coverage and
code handling. Llama-3 uses 128K tokens.
2. Special tokens: Always include <bos>, <eos>, <pad>, <unk>. For instruction-tuned models,
add role markers (<|user|>, <|assistant|>).
3. Fertility: Measure tokens-per-word across languages. High fertility (many tokens per word)
indicates poor coverage for that language.
4. Never tokenize across boundaries: Spaces, punctuation, and digits should be handled
consistently. Most modern tokenizers prepend a space marker (“the”) to distinguish word-initial
vs. continuation tokens.
5. Numbers: Consider digit-level tokenization for arithmetic tasks. “2024” as [“2”,“0”,“2”,“4”]
enables digit-by-digit reasoning.
6. Code: Ensure whitespace (indentation) is tokenized efficiently. Llama-3 tokenizes runs of
spaces as single tokens.
1.2.5
Tokenization in Practice: HuggingFace Example
The transformers library provides a unified interface for all tokenizers. The following demonstrates
encoding and decoding with a modern LLM tokenizer:
from
transformers
import
AutoTokenizer
# Load Llama -3 tokenizer
(128K vocabulary , byte -level BPE)
tokenizer = AutoTokenizer. from_pretrained ("meta -llama/Meta -Llama -3-8B")
text = " Reinforcement
learning
optimizes long -term
rewards."
# Encode: text -> token IDs
token_ids = tokenizer.encode(text)
print(token_ids)
# [128000 , 29934 , 262, 11008 , 4815 , 6900 , 1317 , 9860 , 21845 , 13]
# Decode
individual
tokens to see subword
splits
tokens = tokenizer. convert_ids_to_tokens (token_ids)
print(tokens)
# [’<| begin_of_text|>’, ’Re ’, ’inforce ’, ’ment ’, ’ learning ’,
#
’ optimizes ’, ’ long ’, ’-term ’, ’ rewards ’, ’.’]
# Decode
back to text (round -trip)
reconstructed = tokenizer.decode(token_ids , skip_special_tokens =True)
37


<!-- page 38 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
assert
reconstructed == text
# Perfect
reconstruction
# Tokenize
with
attention
mask (for
batched
inputs
with
padding)
batch = tokenizer(
["Short
text.", "A much
longer
input
sentence
for
comparison."],
padding=True , return_tensors ="pt"
)
print(batch.keys ())
# dict_keys ([’ input_ids ’, ’attention_mask ’])
Listing 1.1: Tokenization encode/decode with HuggingFace Transformers.
1.2.6
Special Tokens and Structured Prompts
Special tokens are reserved vocabulary entries that carry structural meaning rather than linguistic
content. They are critical for controlling model behavior.
Table 1.3: Common special tokens across LLM families.
Token
Alias
Purpose
<bos> / <|begin_of_text|>
BOS
Marks start of sequence
<eos> / <|end_of_text|>
EOS
Marks end of sequence; stops generation
<|user|>
—
Marks start of user turn in chat
<|assistant|>
—
Marks start of assistant turn in chat
<pad>
PAD
Fills batch to uniform length; masked in attention
<unk>
UNK
Out-of-vocabulary placeholder (rare with BPE)
[SEP]
SEP
Separates segments (BERT-style)
[CLS]
CLS
Classification token (BERT)
[MASK]
MASK
Masked token for MLM pretraining
Role Markers for Instruction-Tuned Models.
Modern chat models use special tokens to
delineate conversational structure. These are not trained to carry semantic meaning—they are
structural delimiters that the model learns to parse:
# Llama -3 chat
template
messages = [
{"role": "system", "content": "You are a helpful
assistant."},
{"role": "user", "content": "Explain
PPO in one
sentence."},
]
# apply_chat_template
handles
all
special
token
insertion
prompt = tokenizer. apply_chat_template (messages , tokenize=False)
print(prompt)
# <| begin_of_text |><| start_header_id |>system <| end_header_id |>
#
# You are a helpful
assistant .<| eot_id |><| start_header_id |>user <| end_header_id |>
#
# Explain PPO in one
sentence .<| eot_id |><| start_header_id |>assistant <|
end_header_id|>
#
#
Listing 1.2: Chat template with special tokens (Llama-3 format).
Special Token Best Practices
• Never split special tokens: They must be atomic—ensure your tokenizer treats them as
single units, not character sequences.
• Mask loss on special tokens: During SFT, do not compute loss on structural tokens (role
markers, separators). The model should not “learn” to predict formatting.
38

