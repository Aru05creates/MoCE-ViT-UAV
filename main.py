from train import run_training
from evaluate import evaluate_external

if __name__ == "__main__":
    print("MoCoViT Training started")
    run_training()

    print("Evaluating on External Datasets")
    evaluate_external()

    print(" All stages completed.")

