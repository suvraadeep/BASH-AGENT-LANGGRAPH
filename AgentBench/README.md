# Environment Setup (AgentBench & Docker)  
1. **Clone AgentBench** and create a Python environment. For example, on the remote Linux server:  
   ```bash
   git clone https://github.com/THUDM/AgentBench.git
   cd AgentBench
   python3 -m venv agentbench-env
   source agentbench-env/bin/activate
   pip install -r requirements.txt  # Install dependencies ([GitHub - THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)](https://github.com/THUDM/AgentBench#:~:text=cd%20AgentBench%20conda%20create%20,r%20requirements.txt))
   ```  
2. **Install and build Docker images** for the OS domain. Ensure Docker is running (`docker ps` to check). Then pull and build the required images:  
   ```bash
   docker pull ubuntu
   docker pull mysql
   docker build -f data/os_interaction/res/dockerfiles/default \
       data/os_interaction/res/dockerfiles --tag local-os/default
   docker build -f data/os_interaction/res/dockerfiles/packages \
       data/os_interaction/res/dockerfiles --tag local-os/packages
   docker build -f data/os_interaction/res/dockerfiles/ubuntu \
       data/os_interaction/res/dockerfiles --tag local-os/ubuntu  # OS images ([GitHub - THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)](https://github.com/THUDM/AgentBench#:~:text=docker%20pull%20mysql%20docker%20pull,os%2Fubuntu))
   ```  
   These images simulate the Ubuntu environment for the OS tasks.

# Wrapping Your LangGraph Agent as an AgentBench Client  
AgentBench expects agents to implement a standard interface (a Python module plus parameters) ([AgentBench/docs/Config_en.md at main · THUDM/AgentBench · GitHub](https://github.com/THUDM/AgentBench/blob/main/docs/Config_en.md#:~:text=The%20,configuration%20requires%20the%20following%20fields)). You can **adapt your LangGraph Bash agent** by serving it over HTTP or by writing a custom client wrapper. One simple approach is:

- **Run your agent as an HTTP service.** For example, create a `agent_server.py` that loads your LangGraph agent (from `main.py`) and exposes a `/predict` endpoint. For instance, using FastAPI:  
  ```python
  # agent_server.py
  from fastapi import FastAPI
  from pydantic import BaseModel
  from main import LangGraphAgent  # import your agent code

  app = FastAPI()
  agent = LangGraphAgent()  # initialize your LangGraph agent here

  class Query(BaseModel):
      prompt: str

  @app.post("/predict")
  def predict(q: Query):
      bash_script = agent.run(q.prompt)  # run the agent on the prompt
      return {"bash": bash_script}

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
  ```  
  Then launch this server in one terminal: `python agent_server.py`. This makes your agent accessible via `http://localhost:8000/predict`.  

- **Create an AgentBench agent config.** In `AgentBench/configs/agents/`, add (or edit) a YAML file for your agent, e.g. `langgraph-bash.yaml`:
  ```yaml
  langgraph-bash:
    module: src.client.agents.HTTPAgent   # use built-in HTTP client agent
    parameters:
      url: http://127.0.0.1:8000/predict  # URL of your agent server
      headers:
        Content-Type: application/json
      prompter:
        name: prompt_string               # sends the prompt string as-is
      return_format: "{response[bash]}"   # extract the "bash" field from JSON
      body:
        # You can include fixed fields if needed
        model: none
        temperature: 0.0
  ```  
  Here we use `HTTPAgent` so AgentBench will POST to your server and parse the JSON response ([AgentBench/docs/Config_en.md at main · THUDM/AgentBench · GitHub](https://github.com/THUDM/AgentBench/blob/main/docs/Config_en.md#:~:text=The%20,configuration%20requires%20the%20following%20fields)). Adjust the `return_format` or body as needed.  

3. **Test the agent interface.** AgentBench provides a test script to verify your agent config. Run, for example:  
   ```bash
   python -m src.client.agent_test --config configs/agents/langgraph-bash.yaml \
       --agent langgraph-bash
   ```  
   This should print debug logs confirming your agent responds. (If you had an OpenAI agent, it uses `configs/agents/api_agents.yaml` by default, but here we use our new config.) A successful test indicates AgentBench can communicate with your agent ([GitHub - THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)](https://github.com/THUDM/AgentBench#:~:text=python%20,0613)).

# Running the OS Benchmark Tasks  
1. **Start the AgentBench server and tasks.** Open a new terminal (with the conda/venv activated) and start the task controller and workers. You can let the controller auto-start or run manually. For OS tasks specifically, launch workers with:  
   ```bash
   python -m src.start_task --start os-std 5
   ```  
   This starts the task controller and 5 workers for the `os-std` environment (Ubuntu shell tasks). (Using `-a` would start default tasks for all domains ([GitHub - THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)](https://github.com/THUDM/AgentBench#:~:text=python%20)), but here we restrict to OS.) Wait until the logs show “200 OK” for all workers.  

2. **Launch the assigner.** In another terminal, run the assigner to actually dispatch tasks to your agent:  
   ```bash
   python -m src.assigner
   ```  
   By default, the assigner reads `configs/assignments/default.yaml` and runs all tasks (including OS) with your agent. If you only want OS tasks or dev/test splits, you can edit the assignment config to include `os-std-dev` or `os-std-test` only. The assigner will connect your agent to each OS task and save results in real time ([github.com](https://github.com/THUDM/AgentBench/raw/refs/heads/main/docs/Entrance_en.md#:~:text=then%20their%20ports%20will%20range,field%20already%20exists%2C%20the)).

# Interpreting Results  
The assigner writes output files in the configured output directory (default or as specified in `configs/assignments`). Each task has a JSON entry like:  
```json
"output": {
  "result": true,
  "error": null,
  "file": ".../os_interaction/data/dev.json",
  "index_in_file": 0
}
```  
where `"result": true` means your agent’s Bash output correctly solved that task (evaluated against the “match” criteria) ([Errors in dev data of OS-Interaction · Issue #34 · THUDM/AgentBench · GitHub](https://github.com/THUDM/AgentBench/issues/34#:~:text=,0)). Examine these JSON logs or summary files in the output directory to see pass/fail for each task. Typically, you can compute the overall success rate or accuracy as (# of `result: true`)/(# of tasks). If an `error` appears, it indicates the agent’s script failed or the task crashed. 