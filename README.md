# BASH-AGENT-LANGGRAPH
![Untitled Diagram drawio](https://github.com/user-attachments/assets/89bf8f36-121e-4428-bdbb-f817149f5a7f)


A modular Bash scripting agent made using LangChain, Groq’s LLM, The workflow:

1. **Categorizes** user tasks.
2. **Generates** clarifying questions using RAG (Chroma + HuggingFace embeddings).
3. **Drafts** Bash scripts or further clarification.
4. **Analyzes** drafts for quality.
5. **Rewrites** scripts based on feedback.
6. **Outputs** a final robust Bash script or a request for clarification.

---

## Sample Output:
![image](https://github.com/user-attachments/assets/70de61a6-6213-4095-ac35-91722bdae15c)


## Installation

```bash
# Clone the repository
git clone https://github.com/suvraadeep/BASH-AGENT-LANGGRAPH.git
cd BASH-AGENT-LANGGRAPH

# Create and activate conda environment
conda create -n bash-agent-langgraph python=3.11 -y
conda activate bash-agent-langgraph

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

1. Create a `.env` file in the project root:
    ```dotenv
    GROQ_API_KEY=your_groq_api_key_here
    ```
---

## Usage
Run the main script:

```bash
python main.py
```

You will be prompted to enter a Bash task description. The agent will then step through the workflow and print the final Bash script (or a clarification question if details are missing).



## File Structure

```
.
├── data
│   └── bash_commands_full.json
├── linux_cmd_db
│   └── chroma             # Chroma vector store files
├── notebook
│   └── langraph_bashenv.ipynb
├── outputs
│   ├── Query 1
│   │   ├── Query.md
│   │   ├── Create Query.md
│   │   ├── draft_code.md
│   │   ├── final_code.md
│   │   ├── prompt_category.md
│   │   ├── rag_questions.md
│   │   └── research_info.md
│   ├── Query 2
│   │   ├── Query.md
│   │   ├── Create Query.md
│   │   ├── draft_code.md
│   │   ├── final_code.md
│   │   ├── prompt_category.md
│   │   ├── rag_questions.md
│   │   └── research_info.md
│   ├── Query 3
│   │   ├── Query.md
│   │   ├── Create Query.md
│   │   ├── draft_code.md
│   │   ├── final_code.md
│   │   ├── prompt_category.md
│   │   ├── rag_questions.md
│   │   └── research_info.md
│   └── Query 4
│       ├── Query.md
│       ├── Create Query.md
│       ├── draft_code.md
│       ├── final_code.md
│       ├── prompt_category.md
│       ├── rag_questions.md
│       └── research_info.md
├── .gitignore
├── LICENSE
├── README.md
├── main.py                
└── requirements.txt

```






 ## Agent (chain or callable node) workflow, its role, inputs, and outputs:

### 1. **`research_router`**  
- **What it does:** Decides whether to generate a script immediately or ask for clarification.  
- **Inputs:**  
  - `initial_prompt` (the user’s task description)  
  - `prompt_category` (the broad category of the task)  
- **Process:** Feeds a routing prompt into the LLM and parses its JSON response.  
- **Output:**  
  - `{ "router_decision": "generate_code" }` _or_  
  - `{ "router_decision": "request_clarification" }`

### 2. **`question_rag_chain`**  
- **What it does:** Crafts up to three targeted research questions needed to fill in any missing details for the task.  
- **Inputs:**  
  - `initial_prompt`  
  - `prompt_category`  
- **Process:** LLM generates concise clarifying questions.  
- **Output:**  
  - `{ "questions": ["What shell are you using?", "Do you need error handling?", …] }`


### 3. **`draft_writer_chain`**  
- **What it does:** Produces either:  
  1. A full Bash script (if the task category is “command_enquiry”), or  
  2. A single clarification question (otherwise).  
- **Inputs:**  
  - `initial_prompt`  
  - `prompt_category`  
  - `research_info` (text retrieved or synthesized via RAG)  
- **Process:** LLM uses a prompt template to decide which of those two outputs to return.  
- **Output:** Strict JSON with either:  
  - `"script_md": "<```bash …```>"`  
  - or `"clarification": "…question string…"`

### 4. **`rewrite_router`**  
- **What it does:** Compares the draft output to the original task and decides whether a rewrite is needed.  
- **Inputs:**  
  - `initial_prompt`  
  - `prompt_category`  
  - `draft_code` (the initial script or question)  
- **Process:** LLM judges alignment with user intent.  
- **Output:**  
  - `{ "router_decision": "no_rewrite" }` _or_  
  - `{ "router_decision": "rewrite" }`


### 5. **`draft_analysis_chain`**  
- **What it does:** Acts as a Quality-Control agent, analyzing the draft script or question against the task and RAG info.  
- **Inputs:**  
  - `initial_prompt`  
  - `prompt_category`  
  - `research_info`  
  - `draft_code`  
- **Process:** LLM provides feedback on completeness, correctness, style, etc.  
- **Output:**  
  - `{ "draft_analysis": "…detailed feedback…" }`


### 6. **`rewrite_chain`**  
- **What it does:** Takes the QC feedback and produces a polished final script or clarification.  
- **Inputs:**  
  - `initial_prompt`  
  - `prompt_category`  
  - `research_info`  
  - `draft_code`  
  - `code_analysis`  
- **Process:** LLM rewrites the draft to fully satisfy the task requirements.  
- **Output:**  
  - `{ "final_output": "<script or question>" }`


### 7. **`bash_code_generator`**  
- **What it does:** A straightforward “first draft” Bash-script generator for fully specified tasks.  
- **Inputs:**  
  - `initial_prompt`  
- **Process:** LLM outputs a robust, ready-to-run Bash script (no commentary).  
- **Output:**  
  - A plain string of Bash code (via `StrOutputParser`).


### 8. **`rag_chain`**  
- **What it does:** A Retrieval-Augmented-Generation loop that fetches context from your Chroma vector store and then answers a sub-question.  
- **Inputs:**  
  - `question` (from `question_rag_chain`)  
  - `context` (documents returned by the Chroma retriever)  
- **Process:** Inserts the retrieved snippets into a prompt and calls the LLM.  
- **Output:**  
  - A text snippet (used as part of `research_info`).


### 9. **StateGraph Nodes**  
These wrap the above chains into named steps and wire them together:  
- **`categorize_prompt`** → classifies the task category  
- **`research_info_search`** → runs the RAG questions and `rag_chain`  
- **`draft_email_writer`** → invokes `draft_writer_chain`  
- **Conditional routing**: uses `rewrite_router` to choose  
  - **`no_rewrite`** → skip rewriting  
  - **`analyze_draft_email`** → runs `draft_analysis_chain` and then `rewrite_chain`  

---

##  Summary of Agents
Summary of each component in the pipeline:

| Agent Name             | Role                                                                 |
|------------------------|----------------------------------------------------------------------|
| **categorize_prompt**  | Classifies the user’s Bash task into a category.                    |
| **research_info_search** | Generates targeted questions and retrieves context via RAG.        |
| **draft_email_writer** | Drafts either a full Bash script or a clarification question.       |
| **rewrite_router**     | Decides if the draft needs rewriting based on alignment.            |
| **analyze_draft_email** | Provides QC feedback on the draft script/question.                 |
| **rewrite_email**      | Rewrites the draft into a polished final output.                    |
| **no_rewrite**         | Skips rewriting and uses the draft as final output.                 |
| **bash_code_generator**| Simple one-shot Bash script generator for fully specified tasks.     |
| **rag_chain**          | Retrieval-augmented sub-question answerer using Chroma & LLM.        |

---

