import os
import argparse
from src.data_generation import generate_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_evaluation import evaluate_model

# Import different model training scripts
from src.model_training_random_forest import train_model as train_random_forest
from src.model_training_decision_tree import train_decision_tree as train_decision_tree
from src.model_training_knn import train_knn_model as train_knn
from src.model_training_svm import train_svm_model as train_svm
from src.model_training_logistic_regression import train_logistic_regression as train_logistic_regression
from src.model_training_naive_bayes import train_naive_bayes as train_naive_bayes
from src.model_training_linear_regression import train_linear_model as train_linear_regression

def run_pipeline(model_name):
    """Runs the complete AI pipeline from data generation to evaluation using the chosen model."""
    generate_data()
    preprocess_data()
    feature_engineering()
    
    # Select the correct training model
    if model_name == "random_forest":
        train_random_forest()
    elif model_name == "decision_tree":
        train_decision_tree()
    elif model_name == "knn":
        train_knn()
    elif model_name == "svm":
        train_svm()
    elif model_name == "logistic_regression":
        train_logistic_regression()
    elif model_name == "naive_bayes":
        train_naive_bayes()
    elif model_name == "linear_regression":
        train_linear_regression()
    else:
        raise ValueError("❌ Invalid model name provided.")
    
    #evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI pipeline.")
    parser.add_argument(
        "--step", type=str,
        choices=["generate", "preprocess", "feature", "train", "evaluate", "all"],
        default="all"
    )
    parser.add_argument("--model", type=str, choices=[
        "random_forest", "decision_tree", "knn", "svm", 
        "logistic_regression", "naive_bayes", "linear_regression"
    ], default="random_forest", help="Choose the ML model for training.")

    args = parser.parse_args()

    if args.step == "generate":
        generate_data()
    elif args.step == "preprocess":
        preprocess_data()
    elif args.step == "feature":
        feature_engineering()
    elif args.step == "train":
        if args.model:
            run_pipeline(args.model)
        else:
            print("⚠️ Please specify a model using --model.")
    elif args.step == "evaluate":
        evaluate_model()
    elif args.step == "all":
        run_pipeline(args.model)
    else:
        print("❌ Invalid step provided.")
