import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.align import build_dataset

if __name__ == "__main__":
    ind, aux = build_dataset()
    print(ind.shape, ind.index.min().date(), ind.index.max().date())
    print(aux.describe().T)
