try:
    import torchaudio
except ImportError:
    raise ModuleNotFoundError(
        "TorchAudio package is required to load its dataset. "
        "You can install it as extra dependency with "
        "`pip install avalanche-lib[extra]`"
    )
import torch
import avalanche as avl
from avalanche.benchmarks.datasets.torchaudio_wrapper import SpeechCommands
from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised.strategy_wrappers import Naive


def main():
    n_exp = 7  # 7 experiences -> 5 classes per experience
    hidden_rnn_size = 32
    lr = 1e-3
    # WARNING: Enabling MFCC greatly slows down the runtime execution
    mfcc = False

    if mfcc:
        mfcc_preprocess = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 50, "hop_length": 10}
        )
    else:
        mfcc_preprocess = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SpeechCommands(subset="training", mfcc_preprocessing=mfcc_preprocess)
    test_ds = SpeechCommands(
        subset="testing",  # you may also use "validation"
        mfcc_preprocessing=mfcc_preprocess,
    )

    benchmark = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=test_ds,
        shuffle=True,
        train_transform=None,
        eval_transform=None,
        n_experiences=n_exp,
        task_labels=False,
    )

    classes_in_experience = [
        benchmark.classes_in_experience["train"][i]
        for i in range(benchmark.n_experiences)
    ]
    print(f"Number of training experiences: {len(benchmark.train_stream)}")
    print(f"Number of test experiences: {len(benchmark.test_stream)}")
    print(f"Number of classes: {benchmark.n_classes}")
    print(f"Classes per experience: " f"{classes_in_experience}")

    input_size = 1 if mfcc_preprocess is None else mfcc_preprocess.n_mfcc
    model = avl.models.SimpleSequenceClassifier(
        input_size=input_size,
        hidden_size=hidden_rnn_size,
        n_classes=benchmark.n_classes,
    )
    
    # FIX 1: Move model to GPU
    model = model.to(device)
    print(f"Model moved to: {next(model.parameters()).device}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    # FIX 2: Increase batch sizes for better GPU utilization
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=256,  # Increased from 64
        train_epochs=1,
        eval_mb_size=512,   # Increased from 256
        device=device,
        evaluator=eval_plugin,
        # FIX 3: Use more data loader workers for faster CPU preprocessing
        num_workers=4,      # Add workers for parallel data loading
        persistent_workers=True,  # Keep workers alive between epochs
    )

    # Optional: Enable mixed precision for faster training
    # from torch.cuda.amp import autocast, GradScaler
    # scaler = GradScaler()

    # Store results for summary
    all_results = []
    
    for exp in benchmark.train_stream:
        print(f"\nTraining on experience {exp.current_experience}")
        strategy.train(exp)
        eval_results = strategy.eval(benchmark.test_stream)
        all_results.append(eval_results)
    
    # Print accuracy summary
    print("\n" + "="*60)
    print("FINAL ACCURACY SUMMARY")
    print("="*60)
    
    # Get final accuracies for each experience
    final_accs = []
    for i in range(benchmark.n_experiences):
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in eval_results:
            acc = eval_results[key]
            final_accs.append(acc)
            print(f"Experience {i}: {acc:.4f}")
    
    if final_accs:
        avg_acc = sum(final_accs) / len(final_accs)
        print(f"\nAverage Accuracy: {avg_acc:.4f}")
        print(f"Forgetting: {final_accs[0] - final_accs[0]:.4f} (Exp 0 final - initial)")
        
        # Show accuracy degradation
        print("\nAccuracy degradation from first exposure:")
        for i in range(1, len(final_accs)):
            if i < len(all_results):
                # Get exp 0 accuracy after training on exp i
                key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'
                if key in all_results[i]:
                    current_exp0_acc = all_results[i][key]
                    initial_exp0_acc = all_results[0][key] if key in all_results[0] else 0
                    degradation = initial_exp0_acc - current_exp0_acc
                    print(f"  After exp {i}: {degradation:.4f} degradation")
    
    print("="*60)


if __name__ == "__main__":
    # Set torch to use TF32 for faster training on 3090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()