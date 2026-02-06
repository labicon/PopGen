import logging
import hydra
import gc
import torch
from omegaconf import DictConfig

from .analyze.trajectory_diversity import TrajectoryDiversityExperiment
from .analyze.video_similarity import VideoSimilarityExperiment
from .analyze.trajectory_similarity import TrajectorySimilarityExperiment

logger = logging.getLogger(__name__)

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup Loggers
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Determine analysis type from config (or command line override)
    # Default to None if not present
    analysis_type = cfg.get("analysis_type", None)

    if not analysis_type:
        logger.error("Please provide 'analysis_type' argument.")
        logger.error("Usage: python run_analysis.py analysis_type=diversity|similarity|video|all")
        return

    logger.info(f"Selected analysis mode: {analysis_type}")

    experiment_classes = []

    if analysis_type == "diversity":
        experiment_classes.append(TrajectoryDiversityExperiment)
    elif analysis_type == "similarity":
        experiment_classes.append(TrajectorySimilarityExperiment)
    elif analysis_type == "video":
        experiment_classes.append(VideoSimilarityExperiment)
    elif analysis_type == "all":
        experiment_classes.append(TrajectoryDiversityExperiment)
        experiment_classes.append(TrajectorySimilarityExperiment)
        experiment_classes.append(VideoSimilarityExperiment)
    else:
        logger.error(f"Unknown analysis_type: {analysis_type}. Supported: 'diversity', 'similarity', 'video', 'all'.")
        return

    for ExpClass in experiment_classes:
        exp = ExpClass(cfg)
        name = exp.experiment_name
        logger.info(f"Starting {name}...")
        try:
            exp.run()
        except Exception as e:
            logger.error(f"Experiment {name} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Force cleanup of the experiment instance and GPU memory
            del exp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
