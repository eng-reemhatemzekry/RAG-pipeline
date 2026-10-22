# RAG From Zero --- A Beginner-Friendly Python Walkthrough
This README explains how to build and understand a Retrieval-Augmented
Generation (RAG) pipeline from the ground up. It is intended to
accompany a Python script such as rag_from_scratch.py.

The goal is not just to run RAG, but to understand what every stage
does, why it is needed, and how information moves through the system.
# 1. What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that lets a
Large Language Model (LLM) answer questions using information retrieved
from your own documents.

A normal LLM answers from patterns learned during training and from the
prompt you provide. It may not know your private documents, may have
outdated knowledge, or may generate details that are not supported by
the source.

RAG adds a retrieval step before generation:
1. Prepare documents --- load information from files or other
   sources.
2. Split documents into chunks --- divide long text into smaller,
   manageable pieces.
3. Create embeddings --- convert each chunk into a numerical
   representation of its meaning.
4. Index the embeddings --- store them in a searchable vector
   index.
5. Receive a question --- the user asks something.
6. Retrieve relevant chunks --- find the chunks whose meanings are
   closest to the question.
7. Build a prompt --- provide the question and retrieved context to
   the LLM.
8. Generate an answer --- ask the LLM to answer using that context.
9. Inspect the result --- check whether the answer is supported by
   the retrieved text.

2. Important concepts
Large Language Model (LLM)
A model that generates text from an input prompt. In a RAG system, the
LLM uses the retrieved text as context for answering a question. The LLM
does not perform the vector search itself unless the application
explicitly gives it a tool to do so.
Document
The original source of information, such as a PDF, text file, webpage,
or database record.
Chunk
A smaller section of a document. Chunking makes retrieval more focused
and helps keep the prompt within the LLM's context limit.
Chunk size
The approximate maximum size of each chunk, often measured in characters
or tokens. Larger chunks preserve more surrounding context but may
include irrelevant text. Smaller chunks can be more precise but may lose
useful context.
Chunk overlap
The amount of text shared by neighboring chunks. Overlap can preserve
context when an important sentence falls across a chunk boundary. Too
much overlap creates redundant chunks and increases storage and
retrieval cost.
Embedding
A vector (a list of numbers) produced by an embedding model. Text with
similar meanings tends to have vectors that are close under the chosen
similarity measure.

An embedding is not a summary or a set of human-readable keywords.
It is a numerical representation used for similarity search.
Vector store / vector index
A system that stores vectors and associated text or metadata, and
searches for vectors similar to a query vector. Examples include FAISS,
Chroma, Qdrant, and vector databases. A simple educational script may
use a local index.
Similarity search
A search that ranks stored vectors by how similar they are to the query
vector. With cosine similarity, a common score is:
`\frac{q\cdot d}{\|q\|\|d\|}`{=tex} \]

Here, `q` is the question embedding and `d` is a document-chunk
embedding. Higher cosine similarity usually means greater directional
similarity. Scores are not automatically probabilities or guarantees of
correctness.

### Top-k

The number of retrieved chunks to return. If `k = 4`, the retriever
returns up to four chunks. A larger `k` may provide more context but can
also add noise, cost, and irrelevant text.

### Context

The retrieved text inserted into the LLM prompt. Context is evidence the
model can use; it does not guarantee that the model will follow it
perfectly.

### Grounded generation

Prompting the LLM to answer using the retrieved context and to say when
the context does not contain enough information. This reduces
unsupported answers but does not eliminate hallucinations.

------------------------------------------------------------------------

## 3. The two phases of RAG

It helps to separate RAG into two phases.

### Phase A --- Indexing

This phase prepares your knowledge base. It usually runs when documents
are added or changed, rather than every time a user asks a question.

-   Read documents.
-   Extract text.
-   Clean or normalize text when necessary.
-   Split text into chunks.
-   Generate one embedding per chunk.
-   Store vectors, chunk text, and metadata.

### Phase B --- Querying

This phase runs whenever the user asks a question.

-   Receive the question.
-   Generate an embedding for the question using the same compatible
    embedding model.
-   Search the index.
-   Retrieve the top-k chunks.
-   Construct a prompt containing the retrieved context and question.
-   Call the LLM.
-   Return the answer, ideally with source references.

**Important:** If the embedding model or its settings change, stored
document embeddings may need to be regenerated. Query embeddings and
document embeddings must live in a compatible vector space.

------------------------------------------------------------------------

## 4. Walkthrough of the Python script

The sections below describe the responsibilities a teaching script
should implement. Match each section to the corresponding function or
code block in `rag_from_scratch.py`.

### Step 1 --- Import libraries

Typical libraries may include:

-   `pathlib` or `os` for file paths.
-   A PDF library such as `pypdf` for extracting text from PDF files.
-   A text splitter, or a small custom function, for chunking.
-   An embedding library or API client for generating embeddings.
-   A vector index such as FAISS, or a vector database client.
-   An LLM client for generating the final answer.

Imports only make functionality available. They do not load documents,
create embeddings, or call a model by themselves.

### Step 2 --- Configure paths and parameters

Common settings include:

``` python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
```

These are **example values**, not universal best settings.

-   `CHUNK_SIZE` controls the approximate chunk length according to the
    splitter's unit.
-   `CHUNK_OVERLAP` controls how much text neighboring chunks share.
-   `TOP_K` controls how many chunks the retriever returns.

Use a valid document path and configure credentials through environment
variables when an external API is required. Do not hard-code API keys in
source code or commit them to Git.

### Step 3 --- Load and extract documents

For a PDF, a loader typically opens the file and reads each page. Each
page's text is extracted and kept with metadata such as its page number
and source filename.

Why preserve metadata? When an answer is produced, metadata helps
identify which document or page supplied the evidence.

PDF text extraction can be imperfect. Scanned PDFs may require OCR, and
tables, columns, headers, and footers may be extracted in the wrong
order. Always inspect a few extracted pages before continuing.

### Step 4 --- Split the text into chunks

The splitter turns long document text into smaller pieces. A chunk
should be large enough to preserve a useful idea but small enough for
retrieval to isolate relevant information.

For example, with a character-based splitter, `CHUNK_SIZE = 1000` means
approximately 1,000 characters per chunk---not 1,000 words or tokens.

The overlap helps avoid losing context at boundaries. If chunks are too
small, definitions may be separated from explanations. If they are too
large, retrieval may return a lot of unrelated text.

A useful debug step is to print: - The number of source documents or
pages. - The number of chunks. - A short preview of a few chunks. -
Their source and page metadata.

### Step 5 --- Generate document embeddings

Each chunk is passed to an embedding model:

``` text
Chunk text → Embedding model → [0.12, -0.08, 0.31, ...]
```

The numbers shown are illustrative only. Real vector dimensions depend
on the selected model.

Embedding every chunk can take time and may incur API cost. For a larger
knowledge base, save the embeddings and index so that they do not need
to be recomputed every time the script starts.

### Step 6 --- Build or load the vector index

The vector index stores chunk vectors and lets the application search
for nearby vectors efficiently. A practical implementation also keeps
the original chunk text and metadata linked to each vector.

If the script rebuilds the index on every run, it is easier to
understand but slower. A persistent index is more efficient, but the
program must ensure that the stored index corresponds to the current
documents, chunking configuration, and embedding model.

### Step 7 --- Accept a user question

The question is the query. The application embeds it with the compatible
embedding model and searches the vector index.

The question does not need to use the same words as the source document.
Semantic retrieval can find relevant text with different wording,
although it is not guaranteed to understand every query correctly.

### Step 8 --- Retrieve the top-k chunks

The retriever ranks the indexed chunks by similarity and returns the
best matches.

Inspect retrieval results before calling the LLM. If the correct passage
is missing, the final answer will usually struggle even if the LLM
itself is powerful.

Common retrieval problems include: - Chunks that are too broad or too
short. - Poor text extraction. - A mismatch between the query and the
document language. - An unsuitable embedding model. - A top-k value that
is too small or too large. - Questions requiring exact keyword matching,
filters, or structured data rather than semantic search alone.

For more advanced systems, consider hybrid retrieval (keyword plus
vector search), metadata filters, reranking, and query rewriting.

### Step 9 --- Build the prompt

The application combines: - Instructions for the LLM. - The retrieved
text. - The user's question.

A grounded prompt should tell the model to use the provided context,
avoid inventing facts, and explicitly say when the context is
insufficient.

For example:

``` text
Answer the question using only the context below.
If the context does not contain the answer, say that you cannot
find enough information in the provided documents.

Context:
[Retrieved chunks go here]

Question:
[User question goes here]
```

This is a simplified prompt. In a real application, retrieved text must
be treated as untrusted data: documents can contain malicious or
irrelevant instructions. The system should tell the model not to follow
instructions found inside retrieved documents.

### Step 10 --- Generate the answer

The prompt is sent to the selected LLM. The model generates a response
based on its learned patterns and the supplied context.

RAG does not retrain or fine-tune the LLM. It provides relevant external
information at query time.

If using a hosted model, an API key and network connection may be
required. If using a local model, the model must be installed and the
machine must have sufficient resources.

### Step 11 --- Show the answer and sources

Display the final answer along with useful source metadata, such as
document filename and page number. If the implementation does not retain
metadata or attach sources to retrieved chunks, it cannot reliably
provide page-level citations.

A useful debugging output shows both: 1. The final answer. 2. The exact
chunks retrieved to support it.

This helps distinguish a retrieval failure from a generation failure.

------------------------------------------------------------------------

## 5. How to run the project

The exact commands depend on the implementation in
`rag_from_scratch.py`. A typical workflow is:

1.  Install the Python version supported by the chosen libraries.
2.  Create and activate a virtual environment.
3.  Install the dependencies listed by the project.
4.  Put the source PDF or text files in the configured data folder.
5.  Configure any required API keys as environment variables.
6.  Run the script from the project directory.

Example virtual environment commands:

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install the packages specified by the script or its
`requirements.txt`, and run:

``` bash
python rag_from_scratch.py
```

The package list and exact run behavior must match the actual Python
file. Do not install every RAG library at once; install only the
dependencies the implementation uses.

------------------------------------------------------------------------

## 6. How to test whether RAG is working

Try several types of questions:

1.  **Direct lookup:** Ask for a fact stated clearly in the document.
2.  **Paraphrase:** Ask for the same fact using different wording.
3.  **Multi-chunk question:** Ask something that may require information
    from more than one passage.
4.  **Out-of-scope question:** Ask about something not present in the
    documents.
5.  **Ambiguous question:** Ask a question that could refer to multiple
    passages.

For each test, inspect the retrieved chunks before judging the generated
answer.

### Quick troubleshooting guide

  -----------------------------------------------------------------------
  Symptom                 Possible cause          What to inspect
  ----------------------- ----------------------- -----------------------
  Answer is unrelated     Wrong chunks retrieved  Print the top-k chunks
                                                  and similarity scores

  Correct text is never   Extraction, chunking,   Inspect extracted text,
  retrieved               or embedding issue      chunk boundaries, and
                                                  query wording

  Answer invents details  Weak grounding or       Improve instructions
                          insufficient evidence   and evaluate the
                                                  context

  Answers miss important  Chunks too small or     Test larger chunks or a
  details                 top-k too low           different top-k

  Results contain too     Chunks too large or     Reduce chunk size or
  much irrelevant text    top-k too high          top-k and compare

  Startup is slow         Embeddings/index are    Persist and reuse the
                          rebuilt every run       index safely

  PDF text is empty or    Scanned pages or        Inspect extraction and
  scrambled               complex layout          consider OCR

  Results change after a  Embedding/index         Rebuild embeddings and
  model update            incompatibility         index consistently
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Common misunderstandings

-   **RAG is not fine-tuning.** Fine-tuning changes model parameters;
    RAG retrieves external information at query time.
-   **Embeddings are not the LLM's answer.** They are numerical vectors
    used for retrieval.
-   **A vector database does not generate answers by itself.** It
    retrieves relevant stored information.
-   **The highest similarity score does not prove the passage is
    correct.** It only ranks candidates according to the chosen
    similarity method.
-   **RAG does not eliminate hallucinations.** Retrieval quality, prompt
    design, model behavior, and evaluation all matter.
-   **More retrieved chunks are not always better.** Extra context can
    distract the model.
-   **A successful run does not prove quality.** Test retrieval and
    answer correctness on representative questions.

------------------------------------------------------------------------

## 8. Production considerations

A beginner script demonstrates the core idea, but a production RAG
application should also consider:

-   Document updates and index versioning.
-   Access control so users retrieve only documents they are allowed to
    see.
-   Secret management and safe logging.
-   Prompt injection and untrusted retrieved content.
-   PII handling and data retention.
-   Source citations and traceability.
-   Evaluation datasets for retrieval and answer quality.
-   Latency, token usage, API costs, and failure handling.
-   Monitoring and a clear fallback when evidence is insufficient.

Never treat retrieved document text as trusted system instructions.
Apply permissions before retrieval, not only after an answer has been
generated.

------------------------------------------------------------------------

## 9. Suggested learning order

If you are learning RAG from zero, study the script in this order:

1.  Python file handling and extracting text.
2.  Strings, lists, loops, and functions.
3.  Chunking and metadata.
4.  Embeddings and vector similarity.
5.  Vector search and top-k retrieval.
6.  Prompt construction.
7.  Calling an LLM.
8.  Showing sources and evaluating answers.
9.  Persisting the index and handling document updates.

The key learning habit is to print and inspect the output at every
stage. Do not treat RAG as a black box: check the extracted text,
chunks, embeddings' shape, retrieved passages, prompt, and final answer
separately.

------------------------------------------------------------------------

## 10. Important note

This README explains the standard RAG concepts and the responsibilities
of a teaching implementation. Since the actual `.py` file was not
supplied with this request, the function names, dependencies, file
paths, embedding provider, vector store, and LLM provider may differ
from your implementation. Update the commands and code references to
match the real script before publishing this README.
