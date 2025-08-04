import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
import wandb

from models.backbones import get_backbone
from models.heads import ClassificationHead, ArcFaceHead, CosFaceHead
from datasets import build_face_cl_scenario
from strategies import get_strategy
from utils.metrics import FaceRecognitionMetrics, ContinualLearningMetrics


class FaceRecognitionModel(nn.Module):
    """Combined model with backbone and head."""
    
    def __init__(self, backbone, head, head_type='softmax'):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_type = head_type
    
    def forward(self, x, labels=None):
        # Get embeddings from backbone
        backbone_output = self.backbone(x)
        embeddings = backbone_output['embeddings']
        
        # Get logits from head
        if self.head_type in ['arcface', 'cosface'] and labels is not None:
            logits = self.head(embeddings, labels)
        else:
            logits = self.head(embeddings)
        
        return logits, embeddings


def create_model(config: DictConfig):
    """Create face recognition model."""
    # Create backbone
    backbone = get_backbone(config.backbone)
    
    # Create head
    head_type = config.get('head_type', 'softmax')
    num_classes = config.model.num_classes
    embedding_dim = config.model.embedding_dim
    
    if head_type == 'softmax':
        head = ClassificationHead(embedding_dim, num_classes)
    elif head_type == 'arcface':
        head = ArcFaceHead(embedding_dim, num_classes)
    elif head_type == 'cosface':
        head = CosFaceHead(embedding_dim, num_classes)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
    
    # Combine into single model
    model = FaceRecognitionModel(backbone, head, head_type)
    
    return model


def create_optimizer(model: nn.Module, config: DictConfig):
    """Create optimizer and scheduler."""
    opt_config = config.optimizer
    opt_type = opt_config.type
    
    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas,
            eps=opt_config.eps
        )
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config.lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    return optimizer


def create_evaluation_plugin(config: DictConfig, loggers):
    """Create evaluation plugin with metrics."""
    metrics = [
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    ]
    
    if config.metrics.track_forgetting:
        metrics.append(forgetting_metrics(experience=True, stream=True))
    
    eval_plugin = EvaluationPlugin(
        *metrics,
        loggers=loggers
    )
    
    return eval_plugin


@hydra.main(version_base=None, config_path="experiments/configs", config_name="base_config")
def main(config: DictConfig):
    # Set device
    device = torch.device(config.experiment.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config.experiment.seed)
    
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Create scenario
    print("\nCreating continual learning scenario...")
    scenario, val_dataset = build_face_cl_scenario(config.dataset)
    
    # Update number of classes based on dataset
    config.model.num_classes = scenario.n_classes
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create loggers
    loggers = [InteractiveLogger()]
    
    if config.logging.use_tensorboard:
        tb_logger = TensorboardLogger(config.logging.log_dir)
        loggers.append(tb_logger)
    
    if config.logging.use_wandb:
        wandb.init(
            project="face-recognition-cl",
            name=config.experiment.name,
            config=OmegaConf.to_dict(config)
        )
        wandb_logger = WandBLogger()
        loggers.append(wandb_logger)
    
    # Create evaluation plugin
    eval_plugin = create_evaluation_plugin(config, loggers)
    
    # Create strategy
    print(f"\nCreating strategy: {config.strategy.name}")
    strategy = get_strategy(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        strategy_config=config.strategy,
        eval_plugin=eval_plugin,
        device=device
    )
    
    # Initialize metrics
    face_metrics = FaceRecognitionMetrics()
    cl_metrics = ContinualLearningMetrics()
    
    # Training loop
    print("\nStarting training...")
    results = []
    
    for experience in scenario.train_stream:
        print(f"\n--- Experience {experience.current_experience} ---")
        print(f"Classes in experience: {experience.classes_in_this_experience}")
        
        # Train on experience
        strategy.train(experience)
        
        # Evaluate on all test experiences
        print("\nEvaluating...")
        exp_results = {}
        
        for test_exp_id, test_experience in enumerate(scenario.test_stream):
            # Run evaluation
            metrics_dict = strategy.eval(test_experience)
            
            # Extract accuracy
            acc_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{test_exp_id:03d}"
            if acc_key in metrics_dict:
                accuracy = metrics_dict[acc_key]
                cl_metrics.update_accuracy_matrix(
                    train_exp=experience.current_experience,
                    test_exp=test_exp_id,
                    accuracy=accuracy
                )
                exp_results[f"test_exp_{test_exp_id}"] = accuracy
        
        # Compute CL metrics
        cl_metrics_dict = cl_metrics.compute_all_metrics()
        exp_results.update(cl_metrics_dict)
        
        results.append({
            'experience': experience.current_experience,
            'metrics': exp_results
        })
        
        # Print results
        print(f"\nResults after experience {experience.current_experience}:")
        print(f"Average accuracy: {cl_metrics_dict['average_accuracy']:.4f}")
        print(f"Forgetting: {cl_metrics_dict['forgetting']:.4f}")
        print(f"Forward transfer: {cl_metrics_dict['forward_transfer']:.4f}")
        print(f"Backward transfer: {cl_metrics_dict['backward_transfer']:.4f}")
        
        # Save checkpoint
        if config.experiment.save_checkpoint:
            checkpoint_path = os.path.join(
                config.experiment.checkpoint_dir,
                f"exp_{experience.current_experience}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'experience': experience.current_experience,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cl_metrics': cl_metrics_dict,
            }, checkpoint_path)
    
    # Final evaluation
    if config.evaluation.final_eval_on_test:
        print("\n=== Final Evaluation ===")
        final_results = {}
        
        for test_exp_id, test_experience in enumerate(scenario.test_stream):
            metrics_dict = strategy.eval(test_experience)
            acc_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{test_exp_id:03d}"
            if acc_key in metrics_dict:
                final_results[f"test_exp_{test_exp_id}"] = metrics_dict[acc_key]
        
        print("\nFinal accuracies per experience:")
        for exp_id, acc in final_results.items():
            print(f"  {exp_id}: {acc:.4f}")
    
    # Save final results
    results_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "results.yaml")
    OmegaConf.save(results, results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Close loggers
    if config.logging.use_wandb:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()