# =============================================================================
# This Script is adapted from the following source:
# https://github.com/jafarbakhshaliyev/Wave-Augs


import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import DLinear, iTransformer, SCINet
from train_eval import train, test
from dataloader import TimeSeriesDataset
from dataset_parameter import dataset_configs
import gc

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()


# Main experiment
def main(epochs, learning_rate, patience, num_iterations, label_len):
    print("Starting main experiment...")
    models = ["SCINet"]

    # Create directories
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Run experiments for each dataset
    for dataset_name, config in dataset_configs.items():
        # Initialize per-iteration results CSV file
        iteration_csv_path = f"/home/abid/WaveFreqAug/baseline_papers/results/iteration_results_{dataset_name}.csv"
        with open(iteration_csv_path, "w") as f_iter:
            f_iter.write(
                "dataset,model,pred_len,aug_type,iteration,val_loss,mae,mse,rse\n"
            )

        # Initialize average results CSV file
        average_csv_path = f"/home/abid/WaveFreqAug/baseline_papers/results/average_results_{dataset_name}.csv"
        with open(average_csv_path, "w") as f_avg:
            f_avg.write(
                "dataset,model,pred_len,aug_type,val_loss,mae,mse,rse,mae_std,mse_std,rse_std\n"
            )

        for model_name in models:
            print(
                f"\n=== Processing dataset: {dataset_name} with model: {model_name} ==="
            )
            # Load datasets
            for pred_len in config["pred_lens"]:
                print(f"\nPrediction length: {pred_len}")
                train_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="train",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                    n_imf=max(
                        [
                            config["aug_params"][pred_len][aug_type]["n_imf"]
                            for aug_type in config["aug_types"]
                        ]
                    ),
                )
                val_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="val",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                )
                test_dataset = TimeSeriesDataset(
                    data_path=config["data_path"],
                    data_name=config["data_name"],
                    flag="test",
                    seq_len=config["seq_len"],
                    label_len=label_len,
                    pred_len=pred_len,
                    enc_in=config["enc_in"],
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    drop_last=True,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    drop_last=True,
                )

                # Run experiments for each augmentation type
                for aug_type in config["aug_types"]:
                    params = config["aug_params"][pred_len][aug_type]
                    mse_list, mae_list, rse_list, val_loss_list = [], [], [], []
                    print(
                        f"\nRunning experiment with {aug_type} augmentation for {dataset_name}, pred_len={pred_len}..."
                    )
                    for itr in range(num_iterations):
                        print(f"Iteration {itr+1}/{num_iterations}")
                        if model_name == "DLinear":
                            model = DLinear(
                                config["seq_len"],
                                pred_len,
                                enc_in=config["enc_in"],
                                individual=False,
                            ).to(device)
                        elif model_name == "iTransformer":
                            model = iTransformer(
                                seq_len=config["seq_len"],
                                pred_len=pred_len,
                                enc_in=config["enc_in"],
                                d_model=512,
                                n_heads=8,
                                e_layers=4,
                                d_ff=2048,
                                dropout=0.1,
                            ).to(device)
                        elif model_name == "SCINet":
                            model = SCINet(
                                input_len=config["seq_len"],
                                output_len=pred_len,
                                input_dim=config["enc_in"],
                                hid_size=1,
                                num_stacks=1,
                                num_levels=3,
                                kernel_size=5,
                                dropout=0.2,
                            ).to(device)
                        val_loss = train(
                            model,
                            train_loader,
                            val_loader,
                            device,
                            aug_type,
                            config["seq_len"],
                            label_len,
                            pred_len,
                            aug_rate=params["aug_rate"],
                            rates=params["rates"],
                            wavelet=params["wavelet"],
                            level=params["level"],
                            sampling_rate=params["sampling_rate"],
                            n_imf=params["n_imf"],
                            epochs=epochs,
                            lr=learning_rate,
                            patience=patience,
                        )
                        model.load_state_dict(
                            torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
                        )
                        mae, mse, rse = test(
                            model, test_loader, device, train_dataset.scaler, pred_len
                        )
                        mse_list.append(mse)
                        mae_list.append(mae)
                        rse_list.append(rse)
                        val_loss_list.append(val_loss)
                        print(
                            f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                        )

                        # Save iteration metrics to iteration_results CSV
                        with open(iteration_csv_path, "a") as f_iter:
                            f_iter.write(
                                f"{dataset_name},{model_name},{pred_len},{aug_type},{itr+1},"
                                f"{val_loss:.6f},{mae:.6f},{mse:.6f},{rse:.6f}\n"
                            )

                    # Save average and standard deviation metrics to average_results CSV
                    with open(average_csv_path, "a") as f_avg:
                        f_avg.write(
                            f"{dataset_name},{model_name},{pred_len},{aug_type},"
                            f"{np.mean(val_loss_list):.6f},{np.mean(mae_list):.6f},{np.mean(mse_list):.6f},"
                            f"{np.mean(rse_list):.6f},{np.std(mae_list):.6f},{np.std(mse_list):.6f},{np.std(rse_list):.6f}\n"
                        )

                    print(
                        f"{aug_type} - Avg Val Loss: {np.mean(val_loss_list):.6f}, Avg MAE: {np.mean(mae_list):.6f}, "
                        f"Avg MSE: {np.mean(mse_list):.6f}, Avg RSE: {np.mean(rse_list):.6f}, "
                        f"MAE Std: {np.std(mae_list):.6f}, MSE Std: {np.std(mse_list):.6f}, RSE Std: {np.std(rse_list):.6f}"
                    )

                    # Plot predictions
                    model.load_state_dict(
                        torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
                    )
                    model.eval()
                    with torch.no_grad():
                        batch_x, batch_y, _ = next(iter(test_loader))
                        batch_x = batch_x.float().to(device)
                        outputs = model(batch_x)
                        outputs = outputs[:, -pred_len:, :].cpu().numpy()
                        batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
                        outputs = train_dataset.scaler.inverse_transform(outputs[0])
                        batch_y = train_dataset.scaler.inverse_transform(batch_y[0])
                        plt.figure()
                        plt.plot(batch_y[:, 0], label="Ground Truth")
                        plt.plot(outputs[:, 0], label=f"Prediction ({aug_type})")
                        plt.legend()
                        plt.savefig(
                            f"./plots/prediction_{dataset_name}_{aug_type}_{pred_len}.png"
                        )
                        plt.close()
                    # Clean up memory after aug_type
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                # Clean up datasets after pred_len
                del (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    train_loader,
                    val_loader,
                    test_loader,
                )
                torch.cuda.empty_cache()
                gc.collect()

    print("All experiments completed.")


if __name__ == "__main__":
    main(epochs=20, learning_rate=0.01, patience=10, num_iterations=3, label_len=0)
    print("Main experiment finished.")
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleanup completed.")
    print("You can now check the results and plots in the respective directories.")
