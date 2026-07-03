from typing import Callable, Dict, List, Optional
from torchrl.data import Composite
from benchmarl.environments.common import TaskClass
from torchrl.envs import EnvBase
from torchrl.data import Composite
from benchmarl.environments.common import TaskClass
from env import BenchMarlWebotsWrapper

class WebotsDroneCoverageTask(TaskClass):
    def __init__(self, name: str = "wildfire_drone_coverage", config: dict = None):
        if config is None:
            config = {"num_drones": 3, "map_size": 50, "spacing": 5}
        super().__init__(name=name, config=config)

    @staticmethod
    def env_name() -> str:
        return "webots_drone_env"

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: Optional[int], device: str
    ) -> Callable[[], EnvBase]:
        from env import WebotsDroneCoverageEnv
        
        def make_env():
            # Instantiate raw environment
            raw_env = WebotsDroneCoverageEnv(
                num_drones=self.config.get("num_drones", 3),
                map_size=self.config.get("map_size", 50),
                spacing=self.config.get("spacing", 5),
            )
            # Use custom wrapper
            return BenchMarlWebotsWrapper(raw_env, device=device)
            
        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return 1000

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [f"drone_{i}" for i in range(self.config.get("num_drones", 3))]}

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None