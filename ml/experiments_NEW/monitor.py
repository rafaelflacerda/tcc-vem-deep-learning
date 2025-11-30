import psutil
import torch
import time
import json
import pandas as pd
from collections import deque

class ResourceMonitor:
    def __init__(self, log_file="training_stats.json"):
        self.log_file = log_file
        self.logs = []
        
        # Deques para médias móveis
        self.data_times = deque(maxlen=100)
        self.compute_times = deque(maxlen=100)
        self.start_time = time.time()
        
    def log_batch_detailed(self, data_load_time, compute_time):
        """Registra tempos separados de I/O e GPU"""
        self.data_times.append(data_load_time)
        self.compute_times.append(compute_time)
        
    def get_gpu_stats(self):
        """Retorna estatísticas da GPU"""
        if not torch.cuda.is_available():
            return None
        
        stats = {
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }
        
        # Utilização da GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats['gpu_util_percent'] = util.gpu
            pynvml.nvmlShutdown()
        except:
            stats['gpu_util_percent'] = None
            
        return stats
    
    def get_cpu_ram_stats(self):
        """Retorna estatísticas de CPU e RAM"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_used_gb': psutil.virtual_memory().used / 1e9,
            'ram_total_gb': psutil.virtual_memory().total / 1e9,
            'ram_percent': psutil.virtual_memory().percent,
        }
    
    def print_compact(self, epoch, batch_idx, total_batches, batch_size):
        """Imprime apenas informações essenciais (COMPACTO)"""
        gpu_stats = self.get_gpu_stats()
        
        # Médias
        avg_data_time = sum(self.data_times) / len(self.data_times) if self.data_times else 0
        avg_compute_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
        total_batch_time = avg_data_time + avg_compute_time
        
        # Throughput
        throughput = batch_size / total_batch_time if total_batch_time > 0 else 0
        
        # GPU info
        gpu_util = gpu_stats.get('gpu_util_percent', 0) if gpu_stats else 0
        vram_pct = (gpu_stats['memory_allocated_gb']/gpu_stats['memory_total_gb']*100) if gpu_stats else 0
        
        # Log compacto em uma linha
        print(f"[E{epoch:3d}|B{batch_idx:3d}/{total_batches}] "
              f"⚡{throughput:>6.0f} samp/s | "
              f"🎮GPU {gpu_util:>2.0f}% | "
              f"💾VRAM {vram_pct:>4.1f}% | "
              f"I/O {avg_data_time*1000:>4.0f}ms | "
              f"GPU {avg_compute_time*1000:>4.0f}ms")
        
        # Salvar dados completos em arquivo (silenciosamente)
        cpu_ram = self.get_cpu_ram_stats()
        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'throughput_samples_s': throughput,
            'data_load_ms': avg_data_time * 1000,
            'compute_ms': avg_compute_time * 1000,
            'gpu_util': gpu_util,
            'vram_gb': gpu_stats['memory_allocated_gb'] if gpu_stats else None,
            'vram_percent': vram_pct,
            'cpu_percent': cpu_ram['cpu_percent'],
            'ram_percent': cpu_ram['ram_percent'],
        }
        self.logs.append(log_entry)
        
        # Salvar JSON a cada 50 logs
        if len(self.logs) % 50 == 0:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
    
    def diagnose_bottleneck(self):
        """Diagnostica onde está o gargalo"""
        if not self.data_times or not self.compute_times:
            print("⚠️  Dados insuficientes para diagnóstico")
            return
        
        avg_data = sum(self.data_times) / len(self.data_times)
        avg_compute = sum(self.compute_times) / len(self.compute_times)
        
        gpu_stats = self.get_gpu_stats()
        gpu_util = gpu_stats.get('gpu_util_percent', 0) if gpu_stats else 0
        
        print(f"\n{'='*70}")
        print(f"🔍 DIAGNÓSTICO DE GARGALO")
        print(f"{'='*70}")
        print(f"I/O médio: {avg_data*1000:.1f}ms | GPU médio: {avg_compute*1000:.1f}ms | GPU Util: {gpu_util}%")
        print(f"{'='*70}")
        
        if avg_data > avg_compute * 2:
            print("❌ GARGALO: I/O de Dados")
            print("   → Aumentar num_workers ou usar SSD")
        elif gpu_util and gpu_util < 50:
            print("❌ GARGALO: GPU Subutilizada")
            print("   → AUMENTAR BATCH_SIZE (teste 32k, 64k)")
        elif gpu_util and gpu_util > 90:
            print("✅ GPU otimizada!")
        else:
            print("⚖️  Sistema balanceado")
            print("   → Considere aumentar BATCH_SIZE para mais performance")
        
        print(f"{'='*70}\n")
    
    def save_excel_report(self, output_path="training_report.csv"):
        """Salva relatório completo em CSV/Excel"""
        if not self.logs:
            print("⚠️  Nenhum log para salvar")
            return
        
        df = pd.DataFrame(self.logs)
        df.to_csv(output_path, index=False)
        print(f"📊 Relatório CSV salvo em: {output_path}")
        
        # Estatísticas resumidas
        print(f"\n📈 RESUMO DO TREINAMENTO:")
        print(f"   Throughput médio: {df['throughput_samples_s'].mean():.0f} samples/s")
        print(f"   GPU Util média: {df['gpu_util'].mean():.1f}%")
        print(f"   VRAM média: {df['vram_percent'].mean():.1f}%")
        print(f"   Tempo I/O médio: {df['data_load_ms'].mean():.1f}ms")
        print(f"   Tempo GPU médio: {df['compute_ms'].mean():.1f}ms\n")