from intercode.assets import bash_build_docker, bash_image_name, bash_test_data
from intercode.envs import BashEnv

# Build the Bash Docker image (if not already built)
bash_build_docker()

# Create the BashEnv, pointing to the test dataset
env = BashEnv(image_name=bash_image_name,
              data_path=bash_test_data,
              traj_dir="bash_logs/", verbose=False)
