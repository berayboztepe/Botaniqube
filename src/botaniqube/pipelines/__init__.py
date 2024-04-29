import os 
from kedro.framework.context import KedroContext
from kedro.config import OmegaConfigLoader 

class ProjectContext(KedroContext): 
    '''Loads project configuration and registers pipelines''' 
    def get_config_loader(self): 
        """Loads configuration files from the 'conf' directory. """ 
        conf_paths = ["conf/base", "conf/local"] 
        # Load 'base' and 'local' configuration 
        conf_loader = OmegaConfigLoader(conf_paths) 
        return conf_loader 
    
    def _get_pipelines(self): 
        """Registers project pipelines. """ 
        from .dataset_loading_pipeline.pipeline import create_pipeline as dataset_pipeline 
        from .training_pipeline.pipeline import create_pipeline as training_pipeline 
        from .testing_pipeline.pipeline import create_pipeline as testing_pipeline 
        return { 
            "dataset_loading": dataset_pipeline(), 
            "training": training_pipeline(), 
            "testing": testing_pipeline(), 
            "default_": dataset_pipeline() + training_pipeline() + testing_pipeline() 
        }