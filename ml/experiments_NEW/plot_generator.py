import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# CONFIGURAÇÃO DE CAMINHOS (igual ao train.py)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

DATASET_NAME = "dataset_viga1D_EngastadaLivre_1Meter_44Elements_100kSamples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "tcc-vem-deep-learning", "build", "output", "ML_datasets", DATASET_NAME)
CSV_PATH = os.path.join(DATASET_PATH, "training_diagnostics.csv")

# ============================================================
# CARREGAR DADOS
# ============================================================
print(f"📂 Carregando dados de: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print(f"❌ Arquivo não encontrado: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
epochs = df['epoch']

print(f"✅ Dados carregados: {len(df)} épocas")
print(f"📊 Colunas disponíveis: {list(df.columns)[:10]}...")  # Mostrar primeiras 10 colunas

# ============================================================
# CONFIGURAR FIGURA COMPLETA
# ============================================================
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Análise Completa do Treinamento', fontsize=16, fontweight='bold')

# ============================================================
# 1. LOSS TOTAL (Train vs Val) - O MAIS IMPORTANTE
# ============================================================
ax1 = plt.subplot(4, 3, 1)
ax1.semilogy(epochs, df['train_loss_total'], label='Train', alpha=0.8, linewidth=2)
ax1.semilogy(epochs, df['val_loss_total'], label='Val', alpha=0.8, linewidth=2)
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('1. Loss Total - Convergência Geral')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============================================================
# 2. COMPONENTES MSE (Train vs Val)
# ============================================================
ax2 = plt.subplot(4, 3, 2)
ax2.semilogy(epochs, df['train_mse'], label='Train MSE', alpha=0.8)
ax2.semilogy(epochs, df['val_mse'], label='Val MSE', alpha=0.8)
ax2.set_xlabel('Época')
ax2.set_ylabel('MSE (log scale)')
ax2.set_title('2. Componente MSE (Deslocamento)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================
# 3. COMPONENTES SOBOLEV (Train vs Val)
# ============================================================
ax3 = plt.subplot(4, 3, 3)
ax3.semilogy(epochs, df['train_sobolev'], label='Train Sobolev', alpha=0.8)
ax3.semilogy(epochs, df['val_sobolev'], label='Val Sobolev', alpha=0.8)
ax3.set_xlabel('Época')
ax3.set_ylabel('Sobolev Loss (log scale)')
ax3.set_title('3. Componente Sobolev (Rotação)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================
# 4. RATIO MSE/SOBOLEV - Qual componente domina?
# ============================================================
ax4 = plt.subplot(4, 3, 4)
ax4.semilogy(epochs, df['mse_sobolev_ratio_train'], label='Train', alpha=0.8)
ax4.semilogy(epochs, df['mse_sobolev_ratio_val'], label='Val', alpha=0.8)
ax4.axhline(y=1.0, color='red', linestyle='--', label='Equilíbrio (1:1)', alpha=0.5)
ax4.set_xlabel('Época')
ax4.set_ylabel('Ratio MSE/Sobolev (log scale)')
ax4.set_title('4. Balanceamento MSE vs Sobolev')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================================
# 5. CONTRIBUIÇÃO % DE CADA COMPONENTE
# ============================================================
ax5 = plt.subplot(4, 3, 5)
ax5.plot(epochs, df['mse_contribution'] * 100, label='MSE %', alpha=0.8, linewidth=2)
ax5.plot(epochs, df['sobolev_contribution'] * 100, label='Sobolev %', alpha=0.8, linewidth=2)
ax5.axhline(y=50, color='red', linestyle='--', alpha=0.3)
ax5.set_xlabel('Época')
ax5.set_ylabel('Contribuição (%)')
ax5.set_title('5. Contribuição % no Loss Total')
ax5.set_ylim([0, 100])
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================
# 6. GRADIENTES - Detectar explosão/vanishing
# ============================================================
ax6 = plt.subplot(4, 3, 6)
ax6.semilogy(epochs, df['grad_norm_total'], alpha=0.8, linewidth=1.5)
ax6.axhline(y=10.0, color='red', linestyle='--', label='Alto (>10)', alpha=0.5)
ax6.axhline(y=0.01, color='orange', linestyle='--', label='Baixo (<0.01)', alpha=0.5)
ax6.set_xlabel('Época')
ax6.set_ylabel('Gradient Norm (log scale)')
ax6.set_title('6. Norma dos Gradientes')
ax6.legend()
ax6.grid(True, alpha=0.3)

# ============================================================
# 7. LEARNING RATE - Ver se scheduler está funcionando
# ============================================================
ax7 = plt.subplot(4, 3, 7)
ax7.semilogy(epochs, df['learning_rate'], alpha=0.8, linewidth=2, color='green')
ax7.set_xlabel('Época')
ax7.set_ylabel('Learning Rate (log scale)')
ax7.set_title('7. Learning Rate Schedule')
ax7.grid(True, alpha=0.3)

# ============================================================
# 8. OVERFITTING INDICATOR - Train-Val Gap
# ============================================================
ax8 = plt.subplot(4, 3, 8)
ax8.semilogy(epochs, df['train_val_gap'], alpha=0.8, linewidth=1.5, color='purple')
ax8.set_xlabel('Época')
ax8.set_ylabel('|Train - Val| (log scale)')
ax8.set_title('8. Train-Val Gap (Overfitting?)')
ax8.grid(True, alpha=0.3)

# ============================================================
# 9. OVERFITTING RATIO - Val/Train
# ============================================================
ax9 = plt.subplot(4, 3, 9)
ax9.plot(epochs, df['train_val_ratio'], alpha=0.8, linewidth=1.5, color='purple')
ax9.axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0)', alpha=0.5)
ax9.axhline(y=1.5, color='orange', linestyle='--', label='Moderado (1.5)', alpha=0.5)
ax9.axhline(y=2.0, color='red', linestyle='--', label='Severo (2.0)', alpha=0.5)
ax9.set_xlabel('Época')
ax9.set_ylabel('Val/Train Ratio')
ax9.set_title('9. Overfitting Ratio')
ax9.set_ylim([0, max(3, df['train_val_ratio'].max() * 1.1)])
ax9.legend()
ax9.grid(True, alpha=0.3)

# ============================================================
# 10. WEIGHT CHANGE RATE - Modelo está aprendendo?
# ============================================================
ax10 = plt.subplot(4, 3, 10)
ax10.semilogy(epochs, df['weight_change_rate'], alpha=0.8, linewidth=1.5, color='brown')
ax10.set_xlabel('Época')
ax10.set_ylabel('Weight Change Rate (log scale)')
ax10.set_title('10. Taxa de Mudança dos Pesos')
ax10.grid(True, alpha=0.3)

# ============================================================
# 11. ERROS (MAE) - Train vs Val
# ============================================================
ax11 = plt.subplot(4, 3, 11)
ax11.semilogy(epochs, df['train_mae'], label='Train MAE', alpha=0.8)
ax11.semilogy(epochs, df['val_mae'], label='Val MAE', alpha=0.8)
ax11.set_xlabel('Época')
ax11.set_ylabel('MAE (log scale)')
ax11.set_title('11. Mean Absolute Error')
ax11.legend()
ax11.grid(True, alpha=0.3)

# ============================================================
# 12. VARIABILIDADE ENTRE BATCHES
# ============================================================
ax12 = plt.subplot(4, 3, 12)
ax12.semilogy(epochs, df['train_loss_batch_std'], label='Std Loss', alpha=0.8, color='teal')
ax12.semilogy(epochs, df['grad_norm_batch_std'], label='Std Grad', alpha=0.8, color='coral')
ax12.set_xlabel('Época')
ax12.set_ylabel('Desvio Padrão (log scale)')
ax12.set_title('12. Variabilidade Entre Batches')
ax12.legend()
ax12.grid(True, alpha=0.3)

# ============================================================
# SALVAR E MOSTRAR
# ============================================================
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

output_path = os.path.join(DATASET_PATH, 'analise_completa.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico completo salvo em: {output_path}")

plt.show()