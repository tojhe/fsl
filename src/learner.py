#!/usr/bin/python

import torch as th
from torch import nn
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# Config
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

class Learner:

    def train_single_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.modules.loss
    ):  

        all_losses = []
        all_accuracies = []
        model.train()
        with tqdm(data_loader, total=len(data_loader)) as train_episodes:
            for data in train_episodes:
                
                support_images = data[0]
                support_labels = data[1]
                query_images = data[2]
                query_labels = data[3]

                optimizer.zero_grad()

                prediction_scores = model.forward(
                    support_images.to(DEVICE), 
                    support_labels.to(DEVICE), 
                    query_images.to(DEVICE)
                )

                if isinstance(loss_function, MSELoss):
                    ohe_query_labels = nn.functional.one_hot(query_labels.to(DEVICE), num_classes=5).float()
                    ohe_query_labels = ohe_query_labels.detach()
                    loss = loss_function(prediction_scores, ohe_query_labels)
                else:
                    loss = loss_function(prediction_scores, query_labels.to(DEVICE))

                loss.backward()
                loss_detach = loss.detach()
                optimizer.step()
                
                if model.output_softmax_score:
                    prediction_labels = th.argmax(prediction_scores, -1)
                else:
                    prediction_labels = th.argmax(nn.Softmax(dim=-1)(prediction_scores), -1)
            
                accuracy = accuracy_score(query_labels, prediction_labels.cpu())
                
                all_losses.append(loss_detach.cpu())
                all_accuracies.append(accuracy)

                train_episodes.set_postfix(
                    episode_loss = loss_detach.item(),
                    episode_accuracy = accuracy,
                    epoch_loss = np.mean(all_losses),
                    epoch_accuracy = np.mean(all_accuracies)
                )
                # train_episodes.set_postfix(episode_accuracy = accuracy)
                # train_episodes.set_postfix(epoch_loss = np.mean(all_losses))
                # train_episodes.set_postfix(epoch_accuracy = np.mean(all_accuracies))

        mean_epoch_loss = np.mean(all_losses)
        mean_epoch_accuracy = np.mean(all_accuracies)

        return  mean_epoch_loss, mean_epoch_accuracy
    
    @staticmethod
    def fit(
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.modules.loss,
        epochs: int = 10,
        model_path: str = None,
        tensorboard_log_path: str = './tensorboard/'
    ):  

        best_state = model.state_dict()
        best_val_accuracy = 0

        if tensorboard_log_path is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_log_path)
        
        for e in range(epochs):
            print (f"Training Epoch {e}")

            # episodic training for each epoch
            train_loss, train_accuracy = Learner.train_single_epoch(
                data_loader=train_data_loader,
                model=model, 
                optimizer=optimizer,
                loss_function=loss_function
            )
        
            # Run validation for epoch
            print (f"Validating Epoch {e}")
            model.eval()
            with th.no_grad():
                all_correct_predictions_counts = 0
                all_predictions_counts = 0
                with tqdm(val_data_loader, total=len(val_data_loader)) as val_episodes:

                    # episode is synonymous with task
                    for data in val_episodes:
                        
                        support_images = data[0]
                        support_labels = data[1]
                        query_images = data[2]
                        query_labels = data[3]

                        optimizer.zero_grad()

                        prediction_scores = model.forward(
                            support_images.to(DEVICE), 
                            support_labels.to(DEVICE), 
                            query_images.to(DEVICE)
                        )
                                                
                        if model.output_softmax_score:
                            prediction_labels = th.argmax(prediction_scores, -1)
                        else:
                            prediction_labels = th.argmax(th.nn.Softmax(dim=-1)(prediction_scores), -1)

                        episode_correct_predictions = (prediction_labels.cpu() == query_labels).sum().item()
                        episode_accuracy = accuracy_score(query_labels, prediction_labels.cpu())

                        all_correct_predictions_counts += episode_correct_predictions
                        all_predictions_counts += len(prediction_labels)

                        val_episodes.set_postfix(
                            episode_accuracy = episode_accuracy,
                            overall_accuracy = all_correct_predictions_counts/all_predictions_counts
                            )

                val_accuracy = all_correct_predictions_counts/all_predictions_counts

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_state = model.state_dict()
                
                if tensorboard_log_path is not None:
                    tb_writer.add_scalar("train/loss", train_loss, e)
                    tb_writer.add_scalar("train/accuracy", train_accuracy, e)
                    tb_writer.add_scalar("validation/accuracy", val_accuracy, e)

        if model_path:
            th.save(best_state, model_path)
        
        return best_val_accuracy
        
    @staticmethod
    def predict_single_image(
        support_images, 
        support_labels, 
        single_query_image,
        model
    ):  
        model.eval()
        single_query_image_in = single_query_image.unsqueeze(0)
        prediction_score = model.forward(
                    support_images.to(DEVICE), 
                    support_labels.to(DEVICE), 
                    single_query_image_in.to(DEVICE)
                )
        
        if model.output_softmax_score:
            prediction_label = th.argmax(prediction_score, -1)
        else:
            prediction_label = th.argmax(nn.Softmax(dim=-1)(prediction_score), -1)
        
        return prediction_label
    
    @staticmethod
    def predict_images(
        support_images, 
        support_labels, 
        query_images,
        model
    ):  
        model.eval()
        prediction_score = model.forward(
                    support_images.to(DEVICE), 
                    support_labels.to(DEVICE), 
                    query_images.to(DEVICE)
                )
        
        if model.output_softmax_score:
            prediction_labels = th.argmax(prediction_score, -1)
        else:
            prediction_labels = th.argmax(nn.Softmax(dim=-1)(prediction_score), -1)
        
        return prediction_labels.cpu()