from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from task import WebotsDroneCoverageTask

# 2. Instantiate custom task object
task = WebotsDroneCoverageTask()

# 3. Set up configurations using the .get_from_yaml() defaults
algorithm_config = MappoConfig.get_from_yaml()
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.loggers = ["csv"]
experiment_config.clip_grad_norm = True
experiment_config.on_policy_collected_frames_per_batch = 2000
experiment_config.on_policy_minibatch_size = 400
experiment_config.max_n_frames = 1000000
experiment_config.checkpoint_at_end = True
experiment_config.checkpoint_interval = experiment_config.on_policy_collected_frames_per_batch
#experiment_config.device = "cuda" # Uncomment this line fo GPU training. Will likely be no faster as CPU is still necessary for simulation

# 4. Initialize the Experiment
experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    config=experiment_config,
    seed=42
)

# 5. Run training loop
experiment.run()