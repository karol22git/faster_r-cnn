import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Dodaj ścieżkę do katalogu z plikiem
sys.path.append('../source/')
from BasicBlock import BasicBlock
def test_basic_block():
    # Test 1: Ten sam rozmiar (stride=1, te same kanały)
    print("Test 1: stride=1, in_channels=out_channels")
    block = BasicBlock(64, 64, stride=1)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  shortcut jest Identity? {isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) == 0}")
    
    # Test 2: Różne kanały (wymaga projekcji)
    print("\nTest 2: stride=1, in_channels != out_channels")
    block = BasicBlock(64, 128, stride=1)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  shortcut ma {len(block.shortcut)} warstw")
    
    # Test 3: Downsampling (stride=2)
    print("\nTest 3: stride=2, downsampling")
    block = BasicBlock(64, 128, stride=2)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Przestrzenny downsampling: 32x32 → {y.shape[2]}x{y.shape[3]}")
    
    # Test 4: Sprawdzenie gradientów
    print("\nTest 4: Gradient flow")
    loss = y.sum()
    loss.backward()
    
    # Sprawdź czy gradienty są obliczone
    has_gradients = any(p.grad is not None and p.grad.sum().item() != 0 
                       for p in block.parameters())
    print(f"  Gradienty płyną? {has_gradients}")

# Uruchom test
test_basic_block()
#print("hello")