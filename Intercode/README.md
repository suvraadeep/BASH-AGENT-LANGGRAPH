# Using InterCode to Evaluate the LangGraph Bash Agent

The InterCode benchmark provides an *interactive coding* environment where an agent is given a natural-language prompt and can issue code commands step by step (e.g. in a shell) to solve a task ([InterCode](https://intercode-benchmark.github.io/#:~:text=InterCode%20is%20a%20benchmark%20for,code%20to%20resolve%20the%20issue)).  To evaluate your Bash-oriented LangGraph agent (`main.py`), you will integrate it with InterCode’s **Bash environment**. The high-level workflow is: set up InterCode, start a BashEnv instance, loop through each test prompt, feed it to your agent to generate a Bash script, execute that script stepwise via `env.step()`, and record the final score. 

Below are detailed steps covering environment setup, adapting the evaluation pipeline, handling prompt/output formats, and computing results.

## 1. Install and Set Up InterCode

1. **Install prerequisites.** InterCode requires **Python ≥3.8** and Docker (running locally) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=,)). Ensure Docker is installed and the daemon is running.  
2. **Get InterCode.** You can install via PyPI or from source. For example, using pip:
   ```bash
   pip install intercode-bench
   ```
   (This provides the same code as the repository) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=1)).  
   Alternatively, clone the source for full access:
   ```bash
   git clone https://github.com/princeton-nlp/intercode.git
   cd intercode
   ```
3. **Set up a Python environment.** (If you cloned the repo) Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate intercode
   ```
   This installs all InterCode dependencies ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=git%20clone%20https%3A%2F%2Fgithub.com%2Fprinceton,yml%20conda%20activate%20intercode)).
4. **Build the Docker images.**  InterCode comes with `setup.sh` to build the required environments (Bash, SQL, etc). Run:
   ```bash
   ./setup.sh
   ```
   This creates the Docker image for the Bash environment (as well as others) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=2.%20Run%20,CTF%2C%20Python%2C%20and%20SQL%20environments)). If you installed via pip, you can also use the Python API (see next section) to build only the Bash image.
5. **Verify installation.**  Optionally, you can run the demo Bash environment to check everything is working:
   ```bash
   python run_demo.py bash
   ```
   You should see a CLI prompt with “bash>” where you can manually type shell commands. (Press `Ctrl+C` to exit.) 

   > **Tip:** The quick-start example in the InterCode README shows how to launch BashEnv in Python ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=from%20intercode,envs%20import%20BashEnv)). We’ll follow a similar pattern below, but call your agent automatically instead of manual input.

## 2. Initialize the Bash Environment

Next, import and initialize the Bash environment in a Python script. You can adapt the example code from the InterCode docs ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=from%20intercode,envs%20import%20BashEnv)). For instance:

```python
from intercode.assets import bash_build_docker, bash_image_name, bash_test_data
from intercode.envs import BashEnv

# Build the Bash Docker image (if not already built)
bash_build_docker()

# Create the BashEnv, pointing to the test dataset
env = BashEnv(image_name=bash_image_name,
              data_path=bash_test_data,
              traj_dir="bash_logs/", verbose=False)
```

- `bash_build_docker()` ensures the Bash Docker image is built ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=from%20intercode,envs%20import%20BashEnv)). (If you already ran `setup.sh`, the image is built.)  
- `bash_image_name` is the default image name.  
- `bash_test_data` points to the default Bash test set (24 tasks) provided by InterCode ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=from%20intercode,envs%20import%20BashEnv)). You can supply your own data path if desired.  
- `traj_dir="bash_logs/"` tells InterCode to save logs of each trajectory (optional).  

After this, `env.reset(idx)` will prepare the *idx*-th task.  You can loop over tasks by index (0 to `len(env.data_loader)-1`). Each `env.reset(idx)` makes `env.query` contain the NL prompt, and `env.observation` is the initial observation (usually the same NL query) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=action%20%3D%20input%28%27,total%20number%20of%20scores)).

## 3. Wrap Your Agent for InterCode

You need to adapt the InterCode evaluation to call your Bash agent (`main.py`). The idea is: for each query, pass the prompt to `main.py`, get back a script (string of shell commands), and feed those commands one by one to `env.step`. Here’s a step-by-step outline:

- **Create a Python script** (e.g. `evaluate_agent.py`) that imports your agent or calls it via subprocess. For example, if `main.py` accepts a prompt as a command-line argument and prints the script, you might use `subprocess.run()` to call it.
- **Loop over the test prompts.** For each `idx`:
  1. Call `env.reset(idx)`. Store the prompt:  
     ```python
     prompt = env.query
     ```
  2. **Invoke your agent.** Send `prompt` to your agent (e.g. via `python main.py` or an imported function). Capture its output, which should be a Bash script text. For example:
     ```python
     import subprocess
     result = subprocess.run(
         ["python", "BASH-AGENT-LANGGRAPH/main.py", prompt],
         capture_output=True, text=True
     )
     bash_script = result.stdout
     ```
     (Adjust this based on how your agent is invoked.)  
  3. **Split into commands.** Separate `bash_script` into lines (one per command). For example:  
     ```python
     commands = [line.strip() for line in bash_script.splitlines() if line.strip()]
     ```
  4. **Execute commands stepwise.** For each `cmd` in `commands`, call:
     ```python
     obs, reward, done, info = env.step(cmd)
     ```
     This runs `cmd` in the Docker container. You can optionally check `done` or intermediate `reward` (usually 0 until submission).  
  5. **Submit and score.** After all commands, send the special action `"submit"` to end the task:
     ```python
     obs, reward, done, info = env.step("submit")
     ```
     After `"submit"`, `reward` will be the task score (1.0 if successful, 0 otherwise) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=action%20%3D%20input%28%27,total%20number%20of%20scores)).
  6. **Record results.** Save the score (and any logs) for analysis. You may also log the actions and `obs` for debugging.

pseudocode snippet summarizing the loop (for illustration):

```python
success_count = 0
total = len(env.data_loader)
for idx in range(total):
    env.reset(idx)
    prompt = env.query
    # Call your LangGraph Bash agent with the prompt
    bash_script = call_langgraph_agent(prompt)  
    # Split the script into individual commands
    commands = bash_script.strip().splitlines()
    # Execute each command
    for cmd in commands:
        obs, _, done, info = env.step(cmd)
    # Submit the solution
    obs, reward, done, info = env.step("submit")
    if reward == 1.0:
        success_count += 1
```

This custom loop bypasses the default LLM calls in InterCode and directly uses your agent’s output. 

## 4. Ensure Correct Prompt/Output Formatting

- **Prompt format.** InterCode’s BashEnv provides the task description as a plain string (`env.query`). Pass this string exactly as-is to your agent. The agent should interpret it as a single instruction (e.g. “Write a bash script to …”).  
- **Agent output.** Your agent’s output should be valid Bash commands. Do *not* include any special markers or extra text. It should be runnable shell commands (one per line). For example, `ls -l /path` or `grep "foo" file.txt`. Do *not* include prompts like `bash>` or comments—only executable commands.
- **Submission.** The environment expects you to issue a final `"submit"` action to signal completion ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=action%20%3D%20input%28%27,total%20number%20of%20scores)). This is not an actual shell command, but a keyword that tells InterCode to check the task. Make sure to call `env.step("submit")` after your commands.  

If your agent produces a multi-line script, simply splitting by lines and executing each line (as above) will work. If your agent only returns a one-line solution, that’s fine too; just treat that one line as the full script.

## 5. Run the Benchmark and Evaluate

Once your script is ready, run it from the command line. For example:

```bash
python evaluate_agent.py
```

This will iterate over all tasks in the Bash test set. The script should output or save the number of successes. For example, you might print:

```
Completed 24 tasks: 18 successes, success rate = 75.0%
```

Remember that by definition **success** means the final score is 1.0 (the environment gave full reward) ([GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&B] Code repository for InterCode benchmark https://arxiv.org/abs/2306.14898](https://github.com/princeton-nlp/intercode#:~:text=action%20%3D%20input%28%27,total%20number%20of%20scores)). You can compute success rate as `(success_count / total) * 100%`. 

You can also inspect the trajectories saved in `traj_dir` (here `"bash_logs/"`) for a detailed log of each task (states, actions, rewards). These logs can help debug failures.



