import torch
from env import WebotsDroneCoverageEnv
from env import BenchMarlWebotsWrapper # Your custom wrapper class

def run_inference(checkpoint_path):
    # 1. Initialize the exact same environment setup
    raw_env = WebotsDroneCoverageEnv(num_drones=3)
    env = BenchMarlWebotsWrapper(raw_env, device="cpu")
    
    # 2. Re-create experiment configuration skeleton
    from benchmarl.algorithms import MappoConfig
    from benchmarl.models import MlpConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from task import WebotsDroneCoverageTask

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.loggers = ["csv"]

    experiment = Experiment(
        task=WebotsDroneCoverageTask(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        config=experiment_config,
        seed=42
    )
    
    # 3. Pull the policy module out and load trained weights
    experiment.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy = experiment.policy
    policy.eval()

    # 4. Run the inference execution loop
    print("Running trained swarm policy... Press Ctrl+C to exit.")
    with torch.no_grad(): # Disable gradient tracking
        tensordict = env.reset()
        
        while True:
            # Pass current observations through the policy network
            tensordict = policy(tensordict)
            
            # Step the environment forward using the policy's chosen actions
            tensordict = env.step(tensordict)
            
            # If all drones crash or complete the task, reset the arena
            if tensordict["done"].all():
                tensordict = env.reset()

if __name__ == "__main__":
    # Point this to the path of the saved .pt checkpoint file
    CHECKPOINT = "training_info\checkpoints\checkpoint_1000000.pt"
    run_inference(CHECKPOINT)