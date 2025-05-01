import os
import json
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

print(f"GROQ_API_KEY loaded successfully: {groq_api_key[:4]}...")


# Initialize the LLM
GROQ_LLM = ChatGroq(model="llama3-70b-8192")

# Prepare embeddings and build a RAG database of Bash command descriptions
embeddings = HuggingFaceEmbeddings(model_name="microsoft/graphcodebert-base")
json_filepath = 'bash_commands_full.json'
with open(json_filepath, 'r', encoding='utf-8') as f:
    command_data = json.load(f)

docs = []
for item in command_data:
    docs.append(Document(
        page_content=item.get('description', ''),
        metadata={'command': item.get('command', '')}
    ))
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = splitter.split_documents(docs)

chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    collection_name='rag_linux_commands',
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="linux_cmd_db"
)
retriever = chroma_db.as_retriever(search_kwargs={"k": 5})

# Define prompt templates and chains
research_router_prompt = PromptTemplate(
    template="""\
You are an assistant routing a Bash task. Decide if you should generate code or request more info.
Return ONLY a JSON object with key "router_decision" (value "generate_code" or "request_clarification") in valid JSON format (use double quotes). No extra text.

TASK DESCRIPTION:
{initial_prompt}

TASK CATEGORY:
{prompt_category}
""",
    input_variables=["initial_prompt", "prompt_category"],
)
research_router = research_router_prompt | GROQ_LLM | JsonOutputParser()

search_rag_prompt = PromptTemplate(
    template="""\
You are an expert at formulating internal questions for Bash tasks.
Given the TASK DESCRIPTION and TASK CATEGORY, produce up to three concise questions needed for a robust script.
Return ONLY a JSON object with key "questions" whose value is a list of strings (JSON array, no more than 3). No extra text.

TASK DESCRIPTION:
{initial_prompt}

TASK CATEGORY:
{prompt_category}
""",
    input_variables=["initial_prompt", "prompt_category"],
)
question_rag_chain = search_rag_prompt | GROQ_LLM | JsonOutputParser()

draft_writer_prompt = PromptTemplate(
    template="""\
You are a Bash scripting assistant. Based on the TASK DESCRIPTION, TASK CATEGORY, and RESEARCH_INFO:

- If TASK_CATEGORY is "command_enquiry", produce a ready-to-run Bash script.
- Otherwise, produce a single clarification question.

**Output format**: Return ONLY one JSON object. If returning a script, the JSON should have a key "script_md" whose value is the entire script (including any ```bash code fences) as a string. If asking for clarification, use key "clarification" with the question string as value. Use strict JSON (double quotes) and no extra text.

TASK_DESCRIPTION:
{initial_prompt}

TASK_CATEGORY:
{prompt_category}

RESEARCH_INFO:
{research_info}
""",
    input_variables=["initial_prompt", "prompt_category", "research_info"],
)
draft_writer_chain = draft_writer_prompt | GROQ_LLM | JsonOutputParser()

rewrite_router_prompt = PromptTemplate(
    template="""\
You are an expert evaluating Bash scripts or questions for a task.
Compare the TASK_DESCRIPTION and TASK_CATEGORY to the GENERATED_OUTPUT.
Return ONLY a JSON object with key "router_decision" whose value is either "no_rewrite" or "rewrite". Use valid JSON (double quotes) and no extra text.

TASK_DESCRIPTION:
{initial_prompt}

TASK_CATEGORY:
{prompt_category}

GENERATED_OUTPUT:
{draft_code}
""",
    input_variables=["initial_prompt", "prompt_category", "draft_code"],
)
rewrite_router = rewrite_router_prompt | GROQ_LLM | JsonOutputParser()

draft_analysis_prompt = PromptTemplate(
    template="""\
You are the Quality Control Agent for Bash scripting tasks.
Read TASK_DESCRIPTION, TASK_CATEGORY, and RESEARCH_INFO, and analyze the GENERATED_OUTPUT (script or question).
Return ONLY a JSON object with key "draft_analysis" and the feedback string as value. Use valid JSON (double quotes), no extra text.

TASK_DESCRIPTION:
{initial_prompt}

TASK_CATEGORY:
{prompt_category}

RESEARCH_INFO:
{research_info}

GENERATED_OUTPUT:
{draft_code}
""",
    input_variables=["initial_prompt", "prompt_category", "research_info", "draft_code"],
)
draft_analysis_chain = draft_analysis_prompt | GROQ_LLM | JsonOutputParser()

rewrite_script_prompt = PromptTemplate(
    template="""\
You are the Final Bash Script Agent. Using the QC feedback, rewrite the draft Bash script to fully meet the TASK_DESCRIPTION.
Return ONLY a JSON object with key "final_output" whose value is the improved script (or clarification question) as a string. Use valid JSON (double quotes) and no extra text.

TASK_DESCRIPTION:
{initial_prompt}

TASK_CATEGORY:
{prompt_category}

RESEARCH_INFO:
{research_info}

DRAFT_OUTPUT:
{draft_code}

QC_FEEDBACK:
{code_analysis}
""",
    input_variables=["initial_prompt", "prompt_category", "research_info", "draft_code", "code_analysis"],
)
rewrite_chain = rewrite_script_prompt | GROQ_LLM | JsonOutputParser()

# Utility to write content to markdown files (used by nodes)
def write_markdown_file(content, filename):
    if hasattr(content, "to_string"):
        content = content.to_string()
    elif not isinstance(content, str):
        if isinstance(content, dict):
            content = "\n".join(f"{k}: {v}" for k, v in content.items())
        elif isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        else:
            content = str(content)
    with open(f"{filename}.md", "w", encoding="utf-8") as f:
        f.write(content)

# Define the state schema for the graph
class GraphState(TypedDict):
    initial_prompt: str
    prompt_category: str
    draft_code: str
    final_code: str
    research_info: List[str]
    info_needed: bool
    num_steps: int
    draft_code_feedback: str
    rag_questions: List[str]

# Define node functions
def categorize_prompt(state: GraphState) -> dict:
    print("---CATEGORIZING INITIAL PROMPT---")
    initial_prompt = state["initial_prompt"]
    num_steps = int(state.get("num_steps", 0)) + 1
    prompt_category = bash_code_generator.invoke({"initial_prompt": initial_prompt})
    write_markdown_file(prompt_category, "prompt_category")
    return {"prompt_category": prompt_category, "num_steps": num_steps}

def research_info_search(state: GraphState) -> dict:
    print("---RESEARCH INFO RAG---")
    initial_prompt = state["initial_prompt"]
    prompt_category = state["prompt_category"]
    num_steps = int(state.get("num_steps", 0)) + 1

    questions_dict = question_rag_chain.invoke({
        "initial_prompt": initial_prompt, 
        "prompt_category": prompt_category
    })
    questions = questions_dict.get("questions", [])

    rag_results = []
    for question in questions:
        print(question)
        temp_docs = rag_chain.invoke(question)
        print(temp_docs)
        question_results = question + "\n\n" + temp_docs + "\n\n\n"
        rag_results.append(question_results)

    write_markdown_file(rag_results, "research_info")
    write_markdown_file(questions, "rag_questions")
    return {"research_info": rag_results, "rag_questions": questions, "num_steps": num_steps}

def draft_email_writer(state: GraphState) -> dict:
    print("---DRAFT CODE WRITER---")
    initial_prompt = state["initial_prompt"]
    prompt_category = state["prompt_category"]
    research_info = state["research_info"]
    num_steps = int(state.get("num_steps", 0)) + 1

    draft_code_dict = draft_writer_chain.invoke({
        "initial_prompt": initial_prompt,
        "prompt_category": prompt_category,
        "research_info": research_info
    })
    draft_code = draft_code_dict.get("script_md", draft_code_dict.get("clarification", ""))
    write_markdown_file(draft_code, "draft_code")
    return {"draft_code": draft_code, "num_steps": num_steps}

def analyze_draft_email(state: GraphState) -> dict:
    print("---DRAFT CODE ANALYZER---")
    initial_prompt = state["initial_prompt"]
    prompt_category = state["prompt_category"]
    draft_code = state["draft_code"]
    research_info = state["research_info"]
    num_steps = int(state.get("num_steps", 0)) + 1

    analysis_dict = draft_analysis_chain.invoke({
        "initial_prompt": initial_prompt,
        "prompt_category": prompt_category,
        "research_info": research_info,
        "draft_code": draft_code
    })
    feedback_str = analysis_dict.get("draft_analysis", "")
    write_markdown_file(feedback_str, "draft_code_feedback")
    return {"draft_code_feedback": feedback_str, "num_steps": num_steps}

def rewrite_email(state: GraphState) -> dict:
    print("---REWRITE CODE---")
    initial_prompt = state["initial_prompt"]
    prompt_category = state["prompt_category"]
    draft_code = state["draft_code"]
    research_info = state["research_info"]
    code_analysis = state["draft_code_feedback"]

    try:
        result = rewrite_chain.invoke({
            "initial_prompt": initial_prompt,
            "prompt_category": prompt_category,
            "research_info": research_info,
            "draft_code": draft_code,
            "code_analysis": code_analysis
        })
    except Exception:
        print("Final rewrite JSON parse failed, using raw fallback.")
        raw = (rewrite_script_prompt | GROQ_LLM | StrOutputParser()).invoke({
            "initial_prompt": initial_prompt,
            "prompt_category": prompt_category,
            "research_info": research_info,
            "draft_code": draft_code,
            "code_analysis": code_analysis
        })
        try:
            result = json.loads(raw)
        except:
            result = {"final_output": ""}
    final_output = result.get("final_output", "")
    write_markdown_file(final_output, "final_code")
    return {"final_code": final_output}

def no_rewrite(state: GraphState) -> dict:
    print("---NO REWRITE CODE---")
    draft_code = state["draft_code"]
    return {"final_code": draft_code}

def route_to_rewrite(state: GraphState) -> str:
    print("---ROUTE TO REWRITE---")
    initial_prompt = state["initial_prompt"]
    prompt_category = state["prompt_category"]
    draft_code = state["draft_code"]
    try:
        router = rewrite_router.invoke({
            "initial_prompt": initial_prompt,
            "prompt_category": prompt_category,
            "draft_code": draft_code
        })
    except Exception:
        print("Rewrite router JSON parse failed, using raw fallback.")
        raw = (rewrite_router_prompt | GROQ_LLM | StrOutputParser()).invoke({
            "initial_prompt": initial_prompt,
            "prompt_category": prompt_category,
            "draft_code": draft_code
        })
        try:
            router = json.loads(raw)
        except:
            router = {"router_decision": "no_rewrite"}
    decision = router.get("router_decision", "no_rewrite")
    print(decision)
    return decision

# Also define chains for initial code generation and RAG
bash_code_prompt = PromptTemplate(
    template="""\
You are a Bash scripting expert.
When given a description of a task, output only the Bash code (no commentary) that accomplishes it.
Make the script robust: include comments, error-checking where appropriate, and use best practices.

TASK DESCRIPTION:
{initial_prompt}

# YOUR BASH SCRIPT:
""",
    input_variables=["initial_prompt"],
)
bash_code_generator = bash_code_prompt | GROQ_LLM | StrOutputParser()

rag_prompt = PromptTemplate(
    template="""\
You are an assistant for answering questions about Bash commands.
Use the provided context from command descriptions to answer the question.
If you don’t know the answer, just say "I don’t know."
Return just your code with no extra words.

QUESTION: {question}

CONTEXT:
{context}

Answer:
""",
    input_variables=["question", "context"],
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | GROQ_LLM
    | StrOutputParser()
)

# Build the StateGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("categorize_prompt", categorize_prompt)
workflow.add_node("research_info_search", research_info_search)
workflow.add_node("draft_email_writer", draft_email_writer)
workflow.add_node("analyze_draft_email", analyze_draft_email)
workflow.add_node("rewrite_email", rewrite_email)
workflow.add_node("no_rewrite", no_rewrite)

workflow.set_entry_point("categorize_prompt")
workflow.add_edge("categorize_prompt", "research_info_search")
workflow.add_edge("research_info_search", "draft_email_writer")
workflow.add_conditional_edges(
    "draft_email_writer",
    route_to_rewrite,
    {"rewrite": "analyze_draft_email", "no_rewrite": "no_rewrite"},
)
workflow.add_edge("analyze_draft_email", "rewrite_email")
workflow.add_edge("rewrite_email", END)
workflow.add_edge("no_rewrite", END)

app = workflow.compile()

# Execute the workflow with user input
user_query = input("Enter the Bash task description: ")
inputs = {"initial_prompt": user_query, "num_steps": 0}
result_state = app.invoke(inputs)
final_code = result_state.get("final_code", "")
print("\nGenerated Script:\n")
print(final_code)
