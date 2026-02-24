# mlp_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import random
from linear_models import plot_only_predictions, plot_ic_distribution, plot_rolling_ic

class MLPConfig:
    """Configuration of the parameters for the models"""
    hidden_layers = [264, 128, 64]
    dropout_rate = 0.55
    
    num_epochs = 500
    batch_size = 128
    learning_rate = 0.001
    
    mse_weight = 0.8
    sharpe_weight = 0.3

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate=0.25):
        super(SimpleMLP, self).__init__()
        
        layers = []
        sizes = [input_size] + hidden_layers + [1]
        
        # Building the different layers of the MLP
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.normal_(layer.bias, mean=0.0, std=0.01)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class MLPModel:
    def __init__(self, config=None):
        self.config = config or MLPConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}

        self.dates = {}
        self.y_data = {}
        
    def set_seed(self, seed_value=42):
        """Fix seed for reproductibility"""
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.backends.cudnn.deterministic = True

    def improved_loss(self, predictions, targets):
        """Custom loss function combining MSE and Sharpe ratio"""
        
        # 1. MSE Loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 2. Sharpe-based loss
        portfolio_returns = predictions * targets
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8
        sharpe_loss = -mean_return / std_return
        
        # 3. Total Loss
        total_loss = self.config.mse_weight * mse_loss + self.config.sharpe_weight * sharpe_loss
        
        return total_loss, mse_loss, sharpe_loss

    def train_model(self, train_loader, val_loader):
        """Training the model with early stopping based on Sharpe ratio"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        history = {'train_loss': [], 'val_sharpe': []}
        best_val_sharpe = -float('inf')
        patience = 50
        patience_counter = 0
        
        print(f"Training the model for {self.config.num_epochs} epochs with patience of {patience}...")
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                predictions = self.model(inputs)
                loss, mse_loss, sharpe_loss = self.improved_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            self.model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    predictions = self.model(inputs)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Sharpe ratio validation
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            portfolio_returns = val_predictions * val_targets
            val_sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
            
            history['train_loss'].append(avg_train_loss)
            history['val_sharpe'].append(val_sharpe)
            
            # Early stopping
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 100 == 0:
                print(f'    Epoch {epoch+1}/{self.config.num_epochs} | Loss: {avg_train_loss:.4f} | Val Sharpe: {val_sharpe:.4f}')
            
            if patience_counter >= patience:
                print(f"    Stopped the training at epoch : {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        print(f"    Best validation sharpe ratio: {best_val_sharpe:.4f}")
        return history

    def fit_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train the MLP model with the provided datasets"""
        
        self.dates = {
            'train': X_train.index.unique(),
            'val': X_val.index.unique(), 
            'test': X_test.index.unique()
        }
        self.y_data = {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
        
        self.set_seed(42)
        
        # Standardisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train.values.ravel(), dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val.values.ravel(), dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test.values.ravel(), dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model = SimpleMLP(
            input_size=X_train_scaled.shape[1],
            hidden_layers=self.config.hidden_layers,
            dropout_rate=self.config.dropout_rate
        )
        
        # Training
        history = self.train_model(train_loader, val_loader)
        
        # Evaluation on test set
        predictions_test = self.evaluate_model(test_loader)
        
        # Prediction on train and validation sets
        self.model.eval()
        with torch.no_grad():
            train_val_dataset = TensorDataset(
                torch.tensor(np.vstack([X_train_scaled, X_val_scaled]), dtype=torch.float32),
                torch.tensor(np.concatenate([y_train.values.ravel(), y_val.values.ravel()]), dtype=torch.float32)
            )
            train_val_loader = DataLoader(train_val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            predictions_train = []
            for inputs, _ in train_val_loader:
                preds = self.model(inputs)
                predictions_train.extend(preds.cpu().numpy())
        
        self.results = {
            'predictions': predictions_test,
            'predictions_train': np.array(predictions_train),
            'mse': mean_squared_error(y_test.values.ravel(), predictions_test),
            'history': history
        }
        y_test_flat = y_test.values.flatten()
        df_pred = pd.DataFrame(
            {'predicted': predictions_test,
            'actual_return': y_test_flat
            }, index = y_test.index)
        pred_by_month = df_pred.groupby('date')

        ic_series = pred_by_month.apply(
            lambda x: pd.Series({
                'ic': spearmanr(x['predicted'], x['actual_return'])[0],
                'rmse': np.sqrt(mean_squared_error(x['predicted'], x['actual_return']))
            })
        )
        plot_only_predictions(self.dates['test'], predictions_test, y_test, title='MLP Predictions', filename='mlp_predictions.png')

        plot_ic_distribution(ic_series, filename='mlp_ic_distribution.png')
        plot_rolling_ic(ic_series,filename='mlp_rolling_ic.png')
        
        return self.results

    def evaluate_model(self, test_loader):
        """Evaluate the model on the test set and plot the results."""
        
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        timed_returns = targets.reshape(-1, 1) * predictions.reshape(-1, 1)
        together = pd.DataFrame(np.concatenate([targets.reshape(-1, 1), timed_returns], axis=1), 
                               columns=['Market', 'Portfolio'])
        (together / together.std()).cumsum().plot()
        plt.title(f'MLP - Sharpe Market: {targets.mean()/(targets.std()+1e-8):.3f} | Portfolio: {timed_returns.mean()/(timed_returns.std()+1e-8):.3f}')
        plt.tight_layout()
        plt.savefig('plots/mlp_strategy_performance.png')
        plt.close()
        
        return predictions
    
    def print_results(self):
        """Printing the results of the MLP model."""
        if not self.results:
            return
            
        print(f"MLP Neural Network:")
        print(f"   MSE: {self.results['mse']:.6f}")
        
        # Financial metrics
        predictions = self.results['predictions']
        targets = self.y_data['test'].values.ravel()
        portfolio_returns = predictions * targets
        sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
        total_return = portfolio_returns.sum()
        volatility = portfolio_returns.std()
        hit_rate = (portfolio_returns > 0).mean()
        
        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"   Total return: {total_return:.4f}")
        print(f"   Volatility: {volatility:.4f}")
        print(f"   Hit Rate: {hit_rate:.2%}")
    



