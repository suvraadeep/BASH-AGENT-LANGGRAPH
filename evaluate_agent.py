import subprocess
from intercode.assets import bash_build_docker, bash_image_name, bash_test_data
from intercode.envs import BashEnv
import os


def call_langgraph_agent(prompt: str) -> list[str]:
    """
    Invoke your LangGraph Bash agent (main.py) on the given prompt.
    Clears SSL_CERT_FILE to avoid SSL context errors when agent initializes API clients.
    Assumes evaluate_agent.py and main.py are in the same directory.
    Returns a list of shell commands (one per line).
    If the agent fails, logs the error and returns an empty list to allow continued evaluation.
    """
    # locate the agent script
    agent_script = os.path.join(os.path.dirname(__file__), "main.py")
    # prepare environment without SSL_CERT_FILE
    agent_env = os.environ.copy()
    agent_env.pop("SSL_CERT_FILE", None)
    try:
        result = subprocess.run(
            ["python", agent_script, prompt],
            capture_output=True,
            text=True,
            check=True,
            env=agent_env,
        )
        commands = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.CalledProcessError as e:
        print(f"[Agent Error] Exit code {e.returncode} for prompt: {prompt}")
        if e.stderr:
            print(f"stderr: {e.stderr.strip()}")
        commands = []
    return commands


def evaluate_all_tasks():
    # 1. Ensure the Docker image is built
    bash_build_docker()
    
    # 2. Create the BashEnv
    env = BashEnv(
        image_name=bash_image_name,
        data_path=bash_test_data,
        traj_dir="bash_logs/",
        verbose=False,
    )
    
    total = len(env.data_loader)
    success_count = 0
    
    for idx in range(total):
        # 3a. Reset and fetch prompt
        env.reset(idx)
        prompt = env.query
        
        # 3b. Get the agent's Bash script as a list of commands
        commands = call_langgraph_agent(prompt)
        if not commands:
            print(f"Skipping Task {idx+1}: no commands generated.\n")
            continue
        
        # 3c. Execute each command in the BashEnv
        for cmd in commands:
            obs, _, done, info = env.step(cmd)
            if done:
                break
        
        # 3d. Submit to score
        obs, reward, done, info = env.step("submit")
        
        # 3e. Record success
        if reward == 1.0:
            success_count += 1
        
        print(f"Task {idx+1}/{total} → reward={reward}\n")
    
    # 4. Final report
    success_rate = 100.0 * success_count / total
    print(f"Completed {total} tasks: {success_count} successes → {success_rate:.1f}% success rate")

if __name__ == "__main__":
    evaluate_all_tasks()
