import random
import hashlib
import time
from copy import deepcopy
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = DEVICE.type

POP_SIZE = 20
N_GENERATIONS = 12
TOURNAMENT_SIZE = 3
ELITISM = 2
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.3

EVAL_EPOCHS = 3               # epochs por evaluación (pocas para surrogate fitness)
EARLY_STOPPING_PATIENCE = 2   # early stopping interno a la evaluación
TRAIN_SUBSET_FRAC = 0.25      # usa solo este % del train para acelerar
BATCH_SIZE = 128
NUM_WORKERS = 4               # dataloader workers

FILTER_CHOICES = [16, 32, 48, 64, 96, 128]
KERNEL_CHOICES = [3, 5]
POOL_CHOICES = [0, 1]  # 0=no pool, 1=MaxPool
FC_UNITS_CHOICES = [64, 128, 256, 384, 512]
ACT_CHOICES = ["relu", "leaky"]
DROPOUT_CHOICES = [0.0, 0.25, 0.5]

# Dataset: CIFAR-10 loaders

def get_data_loaders(subset_frac=TRAIN_SUBSET_FRAC, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # split train -> train_small + val
    n_train = len(trainset)
    indices = list(range(n_train))
    random.shuffle(indices)
    # usa subset_frac para entrenamiento (surrogate) y 10% para validación
    n_train_small = max(1000, int(n_train * subset_frac))
    n_val = int(n_train * 0.1)
    train_idx = indices[:n_train_small]
    val_idx = indices[n_train_small:n_train_small + n_val]

    train_sub = Subset(trainset, train_idx)
    val_sub = Subset(trainset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# Genotipo -> Modelo (constructor)

class EvolvedCNN(nn.Module):
    def __init__(self, genotype: List[int], num_classes=10):
        super().__init__()
        # genotype:
        # [n_conv, f1, k1, p1, f2, k2, p2, f3, k3, p3, fc_units, act_code, dropout_code]
        self.genotype = genotype
        n_conv = genotype[0]
        act_code = genotype[-2]
        drop_code = genotype[-1]

        activation = ACT_CHOICES[act_code]
        dropout = DROPOUT_CHOICES[drop_code]

        conv_layers = []
        in_ch = 3
        for i in range(n_conv):
            f = genotype[1 + i*3 + 0]
            k = genotype[1 + i*3 + 1]
            p_flag = genotype[1 + i*3 + 2]
            padding = k // 2
            conv_layers.append(nn.Conv2d(in_ch, f, kernel_size=k, padding=padding))
            conv_layers.append(nn.BatchNorm2d(f))
            if activation == "relu":
                conv_layers.append(nn.ReLU(inplace=True))
            else:
                conv_layers.append(nn.LeakyReLU(0.1, inplace=True))
            if p_flag == 1:
                conv_layers.append(nn.MaxPool2d(2))
            in_ch = f

        self.conv = nn.Sequential(*conv_layers)

        # calcular tamaño de salida después de convs
        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            out = self.conv(dummy)
        flat = int(np.prod(out.shape[1:]))

        fc_units = genotype[-3]
        fc_layers = []
        fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Linear(flat, fc_units))
        if activation == "relu":
            fc_layers.append(nn.ReLU(inplace=True))
        else:
            fc_layers.append(nn.LeakyReLU(0.1, inplace=True))
        if dropout > 0.0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(fc_units, num_classes))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Fitness cache helpers
fitness_cache: Dict[str, float] = {}

def genotype_to_hash(genotype: List[int]) -> str:
    s = ",".join(map(str, genotype))
    return hashlib.md5(s.encode()).hexdigest()

# Training / Evaluation helpers

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        total += xb.size(0)
        correct += (preds == yb).sum().item()
    return running_loss / total, correct / total

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            total += xb.size(0)
            correct += (preds == yb).sum().item()
    return running_loss / total, correct / total

def fitness_from_genotype(genotype: List[int], train_loader, val_loader, device,
                          epochs=EVAL_EPOCHS, patience=EARLY_STOPPING_PATIENCE, verbose=False) -> float:
    key = genotype_to_hash(genotype)
    if key in fitness_cache:
        if verbose:
            print("Cache hit for genotype", key)
        return fitness_cache[key]

    model = EvolvedCNN(genotype).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        if verbose:
            print(f"  [eval] ep={epoch+1}/{epochs} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        # early stopping
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print("  Early stopping triggered")
                break

    fitness_cache[key] = best_val_acc
    return best_val_acc

# GA operators

def random_genotype() -> List[int]:
    n_conv = random.randint(1, 3)
    genotype = [n_conv]
    for i in range(3):  # reservar 3 slots (cada uno: filters,kernel,pool) aunque algunas capas no se usen
        f = random.choice(FILTER_CHOICES)
        k = random.choice(KERNEL_CHOICES)
        p = random.choice(POOL_CHOICES)
        genotype += [f, k, p]
    fc = random.choice(FC_UNITS_CHOICES)
    act = random.randint(0, len(ACT_CHOICES)-1)
    drop = random.randint(0, len(DROPOUT_CHOICES)-1)
    genotype += [fc, act, drop]
    return genotype

def tournament_selection(pop: List[List[int]], fitnesses: List[float], k=TOURNAMENT_SIZE) -> List[int]:
    chosen = random.sample(range(len(pop)), k)
    best = max(chosen, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])

def uniform_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    if random.random() > CROSSOVER_PROB:
        return deepcopy(p1), deepcopy(p2)
    c1, c2 = deepcopy(p1), deepcopy(p2)
    for i in range(len(c1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    c1[0] = int(np.clip(c1[0], 1, 3))
    c2[0] = int(np.clip(c2[0], 1, 3))
    return c1, c2

def mutate(gen: List[int], prob=MUTATION_PROB) -> None:
    # mutate n_conv
    if random.random() < prob:
        gen[0] = random.randint(1, 3)
    # mutate conv slots
    for i in range(3):
        if random.random() < prob:
            gen[1 + i*3 + 0] = random.choice(FILTER_CHOICES)  # filters
        if random.random() < prob:
            gen[1 + i*3 + 1] = random.choice(KERNEL_CHOICES)   # kernel
        if random.random() < prob:
            gen[1 + i*3 + 2] = random.choice(POOL_CHOICES)     # pool
    if random.random() < prob:
        gen[-3] = random.choice(FC_UNITS_CHOICES)
    if random.random() < prob:
        gen[-2] = random.randint(0, len(ACT_CHOICES)-1)
    if random.random() < prob:
        gen[-1] = random.randint(0, len(DROPOUT_CHOICES)-1)

# Main GA loop

def run_evolution():
    print(f"Device: {PRINT_DEVICE} (torch.cuda.is_available={torch.cuda.is_available()})")
    train_loader, val_loader, test_loader = get_data_loaders()

    # inicializa población
    population = [random_genotype() for _ in range(POP_SIZE)]

    # eval inicial
    fitnesses = []
    print("Evaluando población inicial...")
    for i,ind in enumerate(population):
        f = fitness_from_genotype(ind, train_loader, val_loader, DEVICE, verbose=False)
        fitnesses.append(f)
        print(f"  [{i+1}/{POP_SIZE}] fit={f:.4f} gen={ind}")

    for gen in range(N_GENERATIONS):
        print(f"\n=== GENERACIÓN {gen+1}/{N_GENERATIONS} ===")
        # ordenar
        order = np.argsort(fitnesses)[::-1]
        pop_sorted = [population[i] for i in order]
        fit_sorted = [fitnesses[i] for i in order]

        best = pop_sorted[0]; best_fit = fit_sorted[0]
        avg_fit = float(np.mean(fit_sorted))
        print(f"Mejor fitness: {best_fit:.4f} | Promedio: {avg_fit:.4f} | Mejor gen: {best}")

        # elitismo
        new_pop = [deepcopy(pop_sorted[i]) for i in range(ELITISM)]

        # generar hijos hasta llenar
        pbar = trange((POP_SIZE - ELITISM + 1)//2, desc="Cruzando mutando", leave=False)
        for _ in pbar:
            parent1 = tournament_selection(pop_sorted, fit_sorted)
            parent2 = tournament_selection(pop_sorted, fit_sorted)
            child1, child2 = uniform_crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_pop.append(child1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(child2)

        # si se generaron de más, cortar
        new_pop = new_pop[:POP_SIZE]
        population = new_pop

        # evaluar nueva población (puedes paralelizar si quieres, cuidado con GPU)
        fitnesses = []
        for i,ind in enumerate(population):
            f = fitness_from_genotype(ind, train_loader, val_loader, DEVICE, verbose=False)
            fitnesses.append(f)
            print(f"  [{i+1}/{POP_SIZE}] fit={f:.4f} gen={ind}")

    # resultado final
    order = np.argsort(fitnesses)[::-1]
    best = population[order[0]]
    best_fit = fitnesses[order[0]]
    print("\n=== EVOLUCIÓN COMPLETADA ===")
    print("Mejor genotipo:", best)
    print(f"Mejor fitness (val acc surrogate): {best_fit:.4f}")

    # Entrenar mejor arquitectura en todo el train (sin subset) por más epochs y evaluar en test
    print("\nEntrenando la mejor arquitectura final sobre todo el train (más epochs) y evaluando en test...")
    # Cargar loaders completos (sin subset y con un val split pequeño)
    train_full_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
    train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_full_transform)
    test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    # dividir train_full en train_final y val_final
    n = len(train_full)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val_final = int(0.05 * n)
    train_idx = idxs[n_val_final:]
    val_idx = idxs[:n_val_final]
    train_final = Subset(train_full, train_idx)
    val_final = Subset(train_full, val_idx)
    train_loader_final = DataLoader(train_final, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_final = DataLoader(val_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_final = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # entrenar
    model = EvolvedCNN(best).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    E_FINAL = 10
    best_val = 0.0
    best_state = None
    for ep in range(E_FINAL):
        t_loss, t_acc = train_one_epoch(model, train_loader_final, optimizer, criterion, DEVICE)
        v_loss, v_acc = evaluate_model(model, val_loader_final, criterion, DEVICE)
        print(f"[final train] ep {ep+1}/{E_FINAL} train_acc={t_acc:.3f} val_acc={v_acc:.3f}")
        if v_acc > best_val:
            best_val = v_acc
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate_model(model, test_loader_final, criterion, DEVICE)
    print(f"\nResultado final en test: test_acc={test_acc:.4f}")

if __name__ == "__main__":
    start = time.time()
    run_evolution()
    print("Tiempo total (s):", int(time.time() - start))
