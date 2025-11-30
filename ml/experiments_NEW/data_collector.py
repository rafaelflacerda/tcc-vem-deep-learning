import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime

class DataCollector:
    """
    Coleta dados de treinamento sem interferir no processo.
    Adaptado especificamente para sobolev_loss.
    """
    def __init__(self, model, save_path="training_diagnostics.json"):
        self.model = model
        self.save_path = save_path
        self.data = self._initialize_data()
        self.prev_weights = None
        
    def _initialize_data(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'epoch': [],
            
            # Loss e componentes
            'train_loss_total': [],
            'val_loss_total': [],
            'train_mse': [],
            'train_sobolev': [],
            'val_mse': [],
            'val_sobolev': [],
            'mse_sobolev_ratio_train': [],
            'mse_sobolev_ratio_val': [],
            
            # Gradientes
            'grad_norm_total': [],
            'grad_norm_mean': [],
            'grad_norm_std': [],
            'grad_norm_max': [],
            'grad_norm_min': [],
            'grad_norm_per_layer': [],
            
            # Pesos
            'weight_norm_total': [],
            'weight_change_total': [],
            'weight_change_rate': [],
            'weight_norm_mean': [],
            'weight_norm_std': [],
            'weight_norm_per_layer': [],
            'weight_change_per_layer': [],
            
            # Predições e erros
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_max_error': [],
            'val_max_error': [],
            'train_error_std': [],
            'val_error_std': [],
            'train_pred_mean': [],
            'train_pred_std': [],
            'val_pred_mean': [],
            'val_pred_std': [],
            
            # Overfitting indicators
            'train_val_gap': [],
            'train_val_ratio': [],
            'mse_train_val_gap': [],
            'mse_train_val_ratio': [],
            
            # Otimização
            'learning_rate': [],
            
            # Variabilidade entre batches
            'train_loss_batch_mean': [],
            'train_loss_batch_std': [],
            'train_loss_batch_min': [],
            'train_loss_batch_max': [],
            'grad_norm_batch_std': [],
            
            # Sobolev específico
            'sobolev_lambda': [],
            'sobolev_contribution': [],
            'mse_contribution': []
        }
    
    def collect_gradients(self):
        """Coleta estatísticas dos gradientes"""
        grad_norms = []
        grad_norms_per_layer = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms.append(norm)
                grad_norms_per_layer[name] = norm
        
        if len(grad_norms) == 0:
            return None
        
        total_norm = np.sqrt(sum([n**2 for n in grad_norms]))
        
        return {
            'total': total_norm,
            'mean': np.mean(grad_norms),
            'std': np.std(grad_norms),
            'max': np.max(grad_norms),
            'min': np.min(grad_norms),
            'per_layer': grad_norms_per_layer
        }
    
    def collect_weights(self):
        """Coleta estatísticas dos pesos"""
        weight_norms = []
        weight_norms_per_layer = {}
        current_weights = []
        weight_changes_per_layer = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                norm = param.data.norm(2).item()
                weight_norms.append(norm)
                weight_norms_per_layer[name] = norm
                current_weights.append(param.data.clone().flatten())
        
        total_norm = np.sqrt(sum([n**2 for n in weight_norms]))
        
        # Calcular mudança desde última época
        total_change = 0.0
        if self.prev_weights is not None:
            changes = []
            idx = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    change = (param.data.flatten() - self.prev_weights[idx]).norm(2).item()
                    changes.append(change)
                    weight_changes_per_layer[name] = change
                    idx += 1
            total_change = np.sqrt(sum([c**2 for c in changes]))
        
        self.prev_weights = current_weights
        
        # Taxa de mudança normalizada
        change_rate = total_change / (total_norm + 1e-10)
        
        return {
            'total': total_norm,
            'change_total': total_change,
            'change_rate': change_rate,
            'mean': np.mean(weight_norms),
            'std': np.std(weight_norms),
            'per_layer': weight_norms_per_layer,
            'change_per_layer': weight_changes_per_layer
        }
    
    def collect_predictions(self, outputs, targets):
        """Coleta estatísticas das predições"""
        with torch.no_grad():
            errors = (outputs - targets).cpu().numpy()
            predictions = outputs.cpu().numpy()
            
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            max_error = np.max(np.abs(errors))
            error_std = np.std(errors)
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            return {
                'mae': mae,
                'rmse': rmse,
                'max_error': max_error,
                'error_std': error_std,
                'pred_mean': pred_mean,
                'pred_std': pred_std
            }
    
    def decompose_sobolev_loss(self, y_pred, y_true, lambda_theta):
        """
        Decompõe a sobolev_loss em componentes MSE e Sobolev
        Baseado na implementação original do código
        """
        with torch.no_grad():
            # Separar colunas (igual ao sobolev_loss original)
            w_pred, theta_pred = y_pred[:, 0], y_pred[:, 1]
            w_true, theta_true = y_true[:, 0], y_true[:, 1]
            
            # MSE de cada componente
            mse_w = torch.mean((w_pred - w_true) ** 2).item()
            mse_theta = torch.mean((theta_pred - theta_true) ** 2).item()
            
            # Componentes da loss
            mse_component = mse_w  # Apenas deslocamento
            sobolev_component = lambda_theta * mse_theta  # Rotação ponderada
            total_loss = mse_component + sobolev_component
            
            ratio = mse_component / (sobolev_component + 1e-10)
            sobolev_contrib = sobolev_component / (total_loss + 1e-10)
            mse_contrib = mse_component / (total_loss + 1e-10)
            
            return {
                'mse': mse_component,
                'sobolev': sobolev_component,
                'ratio': ratio,
                'sobolev_contribution': sobolev_contrib,
                'mse_contribution': mse_contrib
            }
    
    def log_epoch(self, epoch, train_loss, val_loss, 
                  train_outputs, train_targets,
                  val_outputs, val_targets,
                  batch_losses, batch_grad_norms,
                  lambda_theta, learning_rate):
        """
        Registra todos os dados de uma época
        """
        self.data['epoch'].append(epoch)
        
        # Loss total
        self.data['train_loss_total'].append(train_loss)
        self.data['val_loss_total'].append(val_loss)
        
        # Decompor loss (treino)
        train_decomp = self.decompose_sobolev_loss(train_outputs, train_targets, lambda_theta)
        self.data['train_mse'].append(train_decomp['mse'])
        self.data['train_sobolev'].append(train_decomp['sobolev'])
        self.data['mse_sobolev_ratio_train'].append(train_decomp['ratio'])
        self.data['sobolev_contribution'].append(train_decomp['sobolev_contribution'])
        self.data['mse_contribution'].append(train_decomp['mse_contribution'])
        
        # Decompor loss (validação)
        val_decomp = self.decompose_sobolev_loss(val_outputs, val_targets, lambda_theta)
        self.data['val_mse'].append(val_decomp['mse'])
        self.data['val_sobolev'].append(val_decomp['sobolev'])
        self.data['mse_sobolev_ratio_val'].append(val_decomp['ratio'])
        
        # Gradientes (último batch coletado)
        if len(batch_grad_norms) > 0:
            grad_info = batch_grad_norms[-1]  # Usar último batch
            if grad_info is not None:
                self.data['grad_norm_total'].append(grad_info['total'])
                self.data['grad_norm_mean'].append(grad_info['mean'])
                self.data['grad_norm_std'].append(grad_info['std'])
                self.data['grad_norm_max'].append(grad_info['max'])
                self.data['grad_norm_min'].append(grad_info['min'])
                self.data['grad_norm_per_layer'].append(grad_info['per_layer'])
        
        # Pesos
        weight_info = self.collect_weights()
        self.data['weight_norm_total'].append(weight_info['total'])
        self.data['weight_change_total'].append(weight_info['change_total'])
        self.data['weight_change_rate'].append(weight_info['change_rate'])
        self.data['weight_norm_mean'].append(weight_info['mean'])
        self.data['weight_norm_std'].append(weight_info['std'])
        self.data['weight_norm_per_layer'].append(weight_info['per_layer'])
        self.data['weight_change_per_layer'].append(weight_info['change_per_layer'])
        
        # Predições (treino)
        train_pred = self.collect_predictions(train_outputs, train_targets)
        self.data['train_mae'].append(train_pred['mae'])
        self.data['train_rmse'].append(train_pred['rmse'])
        self.data['train_max_error'].append(train_pred['max_error'])
        self.data['train_error_std'].append(train_pred['error_std'])
        self.data['train_pred_mean'].append(train_pred['pred_mean'])
        self.data['train_pred_std'].append(train_pred['pred_std'])
        
        # Predições (validação)
        val_pred = self.collect_predictions(val_outputs, val_targets)
        self.data['val_mae'].append(val_pred['mae'])
        self.data['val_rmse'].append(val_pred['rmse'])
        self.data['val_max_error'].append(val_pred['max_error'])
        self.data['val_error_std'].append(val_pred['error_std'])
        self.data['val_pred_mean'].append(val_pred['pred_mean'])
        self.data['val_pred_std'].append(val_pred['pred_std'])
        
        # Overfitting indicators
        train_val_gap = abs(train_loss - val_loss)
        train_val_ratio = val_loss / (train_loss + 1e-10)
        mse_gap = abs(train_decomp['mse'] - val_decomp['mse'])
        mse_ratio = val_decomp['mse'] / (train_decomp['mse'] + 1e-10)
        
        self.data['train_val_gap'].append(train_val_gap)
        self.data['train_val_ratio'].append(train_val_ratio)
        self.data['mse_train_val_gap'].append(mse_gap)
        self.data['mse_train_val_ratio'].append(mse_ratio)
        
        # Variabilidade entre batches
        self.data['train_loss_batch_mean'].append(np.mean(batch_losses))
        self.data['train_loss_batch_std'].append(np.std(batch_losses))
        self.data['train_loss_batch_min'].append(np.min(batch_losses))
        self.data['train_loss_batch_max'].append(np.max(batch_losses))
        
        grad_norms_only = [g['total'] for g in batch_grad_norms if g is not None]
        if len(grad_norms_only) > 0:
            self.data['grad_norm_batch_std'].append(np.std(grad_norms_only))
        else:
            self.data['grad_norm_batch_std'].append(0.0)
        
        # Otimização
        self.data['learning_rate'].append(learning_rate)
        
        # Sobolev lambda
        self.data['sobolev_lambda'].append(lambda_theta)
    
    def save(self):
        """Salva todos os dados coletados em JSON (com conversão para tipos nativos Python)"""
        import numpy as np
        
        # Converter todos os valores numpy para tipos nativos do Python
        def convert_to_native(obj):
            """Converte tipos numpy para tipos nativos do Python"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
            
        # Converter todos os dados
        data_to_save = convert_to_native(self.data)
        
        with open(self.save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"✅ Dados de diagnóstico salvos em: {self.save_path}")
    
    def save_csv(self):
        """Salva dados em formato CSV"""
        try:
            import pandas as pd
            csv_path = self.save_path.replace('.json', '.csv')
            
            # Criar DataFrame excluindo dados nested (per_layer)
            scalar_data = {}
            for key, values in self.data.items():
                if key not in ['timestamp', 'grad_norm_per_layer', 
                              'weight_norm_per_layer', 'weight_change_per_layer']:
                    scalar_data[key] = values
            
            df = pd.DataFrame(scalar_data)
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV salvo em: {csv_path}")
        except ImportError:
            print("⚠️  pandas não disponível, CSV não foi salvo")
        except Exception as e:
            print(f"⚠️  Erro ao salvar CSV: {e}")