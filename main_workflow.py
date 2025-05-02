import os
import json
from typing import List, Dict, Any
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

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

print(f"GROQ_API_KEY loaded successfully: {groq_api_key[:4]}...")

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

class LangGraphAgent:
    def __init__(self, model_name="llama3-70b-8192", embeddings_model="microsoft/graphcodebert-base", json_path='./data/bash_commands_full.json', db_persist_dir="linux_cmd_db", db_collection_name="rag_linux_commands"):
        """Initializes the LangGraphAgent."""
        print("Initializing LangGraphAgent...")

        # Initialize LLM
        self.GROQ_LLM = ChatGroq(model=model_name, groq_api_key=groq_api_key)
        print("LLM initialized.")

        # Prepare embeddings and build RAG database
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        print("Loading command data...")
        with open(json_path, 'r', encoding='utf-8') as f:
            command_data = json.load(f)

        docs = []
        for item in command_data:
            docs.append(Document(
                page_content=item.get('description', ''),
                metadata={'command': item.get('command', '')}
            ))
        print(f"Loaded {len(docs)} command descriptions.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = splitter.split_documents(docs)
        print(f"Split into {len(chunked_docs)} chunks.")

        print("Building Chroma DB...")
        self.chroma_db = Chroma.from_documents(
            documents=chunked_docs,
            collection_name=db_collection_name,
            embedding=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=db_persist_dir
        )
        self.retriever = self.chroma_db.as_retriever(search_kwargs={"k": 5})
        print("Chroma DB and retriever ready.")

        # Define prompt templates
        self._define_prompts()
        print("Prompts defined.")

        # Define chains
        self._define_chains()
        print("Chains defined.")

        # Build the graph
        self._build_graph()
        print("LangGraph workflow compiled.")

    def _define_prompts(self):
        """Defines all PromptTemplate objects."""
        self.research_router_prompt = PromptTemplate(
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

        self.search_rag_prompt = PromptTemplate(
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

        self.draft_writer_prompt = PromptTemplate(
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

        self.rewrite_router_prompt = PromptTemplate(
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

        self.draft_analysis_prompt = PromptTemplate(
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

        self.rewrite_script_prompt = PromptTemplate(
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

        self.bash_code_prompt = PromptTemplate(
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

        self.rag_prompt = PromptTemplate(
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

    def _define_chains(self):
        """Defines all LangChain Runnable chains."""
        self.research_router = self.research_router_prompt | self.GROQ_LLM | JsonOutputParser()
        self.question_rag_chain = self.search_rag_prompt | self.GROQ_LLM | JsonOutputParser()
        self.draft_writer_chain = self.draft_writer_prompt | self.GROQ_LLM | JsonOutputParser()
        self.rewrite_router = self.rewrite_router_prompt | self.GROQ_LLM | JsonOutputParser()
        self.draft_analysis_chain = self.draft_analysis_prompt | self.GROQ_LLM | JsonOutputParser()
        self.rewrite_chain = self.rewrite_script_prompt | self.GROQ_LLM | JsonOutputParser()
        self.bash_code_generator = self.bash_code_prompt | self.GROQ_LLM | StrOutputParser()
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.GROQ_LLM
            | StrOutputParser()
        )

    @staticmethod
    def write_markdown_file(content: Any, filename: str):
        """Utility to write content to markdown files."""
        output_dir = "output_markdown"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.md")

        if hasattr(content, "to_string"):
            content_str = content.to_string()
        elif not isinstance(content, str):
            if isinstance(content, dict):
                content_str = "\n".join(f"{k}: {v}" for k, v in content.items())
            elif isinstance(content, list):
                content_str = "\n".join(str(item) for item in content)
            else:
                content_str = str(content)
        else:
            content_str = content

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_str)
            # print(f"--- Content written to {filepath} ---")
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")


    # --- Node Functions ---

    def categorize_prompt(self, state: GraphState) -> dict:
        """Node: Categorizes the initial user prompt."""
        print("---CATEGORIZING INITIAL PROMPT---")
        initial_prompt = state["initial_prompt"]
        num_steps = int(state.get("num_steps", 0)) + 1

        # This uses the bash_code_generator chain to get an initial idea,
        # which might not be the *intended* use based on the name,
        # but it's what the original code did.
        # Let's assume it's a placeholder or a simplified categorization.
        # A more robust categorization might involve another LLM call or specific logic.
        prompt_category = self.bash_code_generator.invoke({"initial_prompt": initial_prompt})
        # For simplicity, let's just call it "command_enquiry" based on the draft_writer logic
        # This part might need refinement depending on the desired categorization logic.
        # Using a fixed category for now based on original script's draft_writer logic.
        determined_category = "command_enquiry" # Placeholder - adjust as needed
        print(f"Determined Category: {determined_category}")
        LangGraphAgent.write_markdown_file(determined_category, "prompt_category") # Write the determined category
        return {"prompt_category": determined_category, "num_steps": num_steps} # Return the determined category

    def research_info_search(self, state: GraphState) -> dict:
        """Node: Performs RAG search based on generated questions."""
        print("---RESEARCH INFO RAG---")
        initial_prompt = state["initial_prompt"]
        prompt_category = state["prompt_category"]
        num_steps = int(state.get("num_steps", 0)) + 1

        try:
            questions_dict = self.question_rag_chain.invoke({
                "initial_prompt": initial_prompt,
                "prompt_category": prompt_category
            })
            questions = questions_dict.get("questions", [])
        except Exception as e:
            print(f"Error invoking question_rag_chain: {e}. Falling back.")
            # Fallback: Maybe use the initial prompt directly as a question?
            questions = [initial_prompt] # Simple fallback

        if not questions:
             print("No questions generated by question_rag_chain. Using initial prompt.")
             questions = [initial_prompt]

        rag_results = []
        print("Generated RAG Questions:")
        for question in questions:
            print(f"- {question}")
            try:
                temp_docs = self.rag_chain.invoke(question)
                # print("Retrieved Docs:", temp_docs) # Optional: print retrieved docs
                question_results = f"Question: {question}\n\nAnswer:\n{temp_docs}\n\n---\n\n"
                rag_results.append(question_results)
            except Exception as e:
                print(f"Error during RAG retrieval for question '{question}': {e}")
                rag_results.append(f"Question: {question}\n\nError retrieving answer.\n\n---\n\n")


        LangGraphAgent.write_markdown_file("\n".join(rag_results), "research_info")
        LangGraphAgent.write_markdown_file(questions, "rag_questions") # Write the actual questions asked
        return {"research_info": rag_results, "rag_questions": questions, "num_steps": num_steps}

    def draft_code_writer(self, state: GraphState) -> dict:
        """Node: Writes the initial draft script or clarification question."""
        print("---DRAFT CODE WRITER---")
        initial_prompt = state["initial_prompt"]
        prompt_category = state["prompt_category"]
        research_info = state["research_info"]
        num_steps = int(state.get("num_steps", 0)) + 1
        research_info_str = "\n".join(research_info) if research_info else "No research info provided."


        try:
            draft_code_dict = self.draft_writer_chain.invoke({
                "initial_prompt": initial_prompt,
                "prompt_category": prompt_category,
                "research_info": research_info_str
            })
            # Handle potential variations in output key
            draft_code = draft_code_dict.get("script_md", draft_code_dict.get("clarification", ""))
            if not draft_code:
                 print("Warning: draft_writer_chain returned empty content.")
                 draft_code = "# Error: Failed to generate draft code/clarification."

        except Exception as e:
             print(f"Error invoking draft_writer_chain: {e}. Generating fallback.")
             draft_code = f"# Error: Could not generate draft due to: {e}"


        LangGraphAgent.write_markdown_file(draft_code, "draft_code")
        return {"draft_code": draft_code, "num_steps": num_steps}

    def analyze_draft_code(self, state: GraphState) -> dict:
        """Node: Analyzes the draft code and provides feedback."""
        print("---DRAFT CODE ANALYZER---")
        initial_prompt = state["initial_prompt"]
        prompt_category = state["prompt_category"]
        draft_code = state["draft_code"]
        research_info = state["research_info"]
        num_steps = int(state.get("num_steps", 0)) + 1
        research_info_str = "\n".join(research_info) if research_info else "No research info provided."


        try:
            analysis_dict = self.draft_analysis_chain.invoke({
                "initial_prompt": initial_prompt,
                "prompt_category": prompt_category,
                "research_info": research_info_str,
                "draft_code": draft_code
            })
            feedback_str = analysis_dict.get("draft_analysis", "No feedback provided.")
        except Exception as e:
            print(f"Error invoking draft_analysis_chain: {e}. Generating fallback feedback.")
            feedback_str = f"Error during analysis: {e}"


        LangGraphAgent.write_markdown_file(feedback_str, "draft_code_feedback")
        return {"draft_code_feedback": feedback_str, "num_steps": num_steps}

    def rewrite_code(self, state: GraphState) -> dict:
        """Node: Rewrites the code based on feedback."""
        print("---REWRITE CODE---")
        initial_prompt = state["initial_prompt"]
        prompt_category = state["prompt_category"]
        draft_code = state["draft_code"]
        research_info = state["research_info"]
        code_analysis = state["draft_code_feedback"]
        num_steps = int(state.get("num_steps", 0)) + 1 # Increment step count
        research_info_str = "\n".join(research_info) if research_info else "No research info provided."


        try:
            result = self.rewrite_chain.invoke({
                "initial_prompt": initial_prompt,
                "prompt_category": prompt_category,
                "research_info": research_info_str,
                "draft_code": draft_code,
                "code_analysis": code_analysis
            })
            final_output = result.get("final_output", "")
            if not final_output:
                print("Warning: rewrite_chain returned empty content. Using draft as fallback.")
                final_output = draft_code # Fallback to draft if rewrite is empty
        except Exception as e:
            print(f"Final rewrite JSON parse failed or chain error: {e}. Using draft code as fallback.")
            # Attempt raw call as per original logic (though less likely needed with robust error handling)
            # Keeping it simple here: just use the draft code as fallback.
            final_output = draft_code # Fallback to the original draft

        LangGraphAgent.write_markdown_file(final_output, "final_code")
        # Ensure num_steps is part of the returned state if needed downstream, though END is next
        return {"final_code": final_output, "num_steps": num_steps}


    def no_rewrite(self, state: GraphState) -> dict:
        """Node: Finalizes the code without rewriting."""
        print("---NO REWRITE CODE---")
        draft_code = state["draft_code"]
        num_steps = state.get("num_steps", 0) # Keep step count
        LangGraphAgent.write_markdown_file(draft_code, "final_code")
        return {"final_code": draft_code, "num_steps": num_steps}

    # --- Conditional Edge Function ---

    def route_to_rewrite(self, state: GraphState) -> str:
        """Conditional Edge: Decides whether to rewrite the draft."""
        print("---ROUTE TO REWRITE---")
        initial_prompt = state["initial_prompt"]
        prompt_category = state["prompt_category"]
        draft_code = state["draft_code"]

        try:
            router_result = self.rewrite_router.invoke({
                "initial_prompt": initial_prompt,
                "prompt_category": prompt_category,
                "draft_code": draft_code
            })
            decision = router_result.get("router_decision", "no_rewrite")
        except Exception as e:
            print(f"Rewrite router failed: {e}. Defaulting to 'no_rewrite'.")
            decision = "no_rewrite" # Default decision on error

        print(f"Routing Decision: {decision}")
        if decision not in ["rewrite", "no_rewrite"]:
            print(f"Warning: Invalid routing decision '{decision}'. Defaulting to 'no_rewrite'.")
            decision = "no_rewrite"

        return decision

    # --- Graph Building ---

    def _build_graph(self):
        """Builds and compiles the StateGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("categorize_prompt", self.categorize_prompt)
        workflow.add_node("research_info_search", self.research_info_search)
        workflow.add_node("draft_code_writer", self.draft_code_writer) # Corrected node name
        workflow.add_node("analyze_draft_code", self.analyze_draft_code) # Corrected node name
        workflow.add_node("rewrite_code", self.rewrite_code) # Corrected node name
        workflow.add_node("no_rewrite", self.no_rewrite)

        # Define edges
        workflow.set_entry_point("categorize_prompt")
        workflow.add_edge("categorize_prompt", "research_info_search")
        workflow.add_edge("research_info_search", "draft_code_writer") # Corrected node name

        # Conditional edge from draft writer
        workflow.add_conditional_edges(
            "draft_code_writer", # Corrected start node name
            self.route_to_rewrite,
            {
                "rewrite": "analyze_draft_code", # Corrected target node name
                "no_rewrite": "no_rewrite"
            },
        )

        workflow.add_edge("analyze_draft_code", "rewrite_code") # Corrected edge start node name
        workflow.add_edge("rewrite_code", END) # Corrected edge start node name
        workflow.add_edge("no_rewrite", END)

        # Compile the graph
        self.app = workflow.compile()

    # --- Execution Method ---

    def run(self, user_query: str):
        """Executes the workflow with the given user query."""
        print("\n--- Starting Bash Script Generation Workflow ---")
        inputs = {"initial_prompt": user_query, "num_steps": 0}
        try:
            # Stream events for more detailed logging (optional)
            # for event in self.app.stream(inputs):
            #     for key, value in event.items():
            #         print(f"Event: {key}")
            #         # print(f"Data: {value}") # Can be very verbose
            #     print("---")
            # final_state = event # The last event should contain the final state at the END node

            # Regular invoke call
            final_state = self.app.invoke(inputs)

            final_code = final_state.get("final_code", "# Error: Final code not found in state.")
            print("\n--- Workflow Complete ---")
            print("\nGenerated Script/Output:\n")
            print(final_code)

        except Exception as e:
            print(f"\n--- Workflow Failed ---")
            print(f"An error occurred during graph execution: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Ensure the data directory and file exist or handle appropriately
    if not os.path.exists('./data/bash_commands_full.json'):
       print("Error: './data/bash_commands_full.json' not found.")
       print("Please ensure the JSON data file is in the 'data' subdirectory.")
       # You might want to exit or provide instructions to download/create the file
       exit(1)
    # Create the DB directory if it doesn't exist
    os.makedirs("linux_cmd_db", exist_ok=True)
    # Create output markdown directory
    os.makedirs("output_markdown", exist_ok=True)


    agent = LangGraphAgent()
    user_query = input("Enter the Bash task description: ")
    if user_query:
        agent.run(user_query)
    else:
        print("No input provided. Exiting.")