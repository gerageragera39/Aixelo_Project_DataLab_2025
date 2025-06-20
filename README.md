# Aixelo_Project_DataLab_2025

## 🚀 Getting Started

### 1. Download the Data

Download the required dataset from the following link:

👉 [Download `AIxelo_data.zip`](https://drive.google.com/file/d/1RYSgDHryxP-1T7bevY000ewgbMvBy_S8/view?usp=sharing)

### 2. Extract the Archive

After downloading `AIxelo_data.zip`, **make sure to extract** folder 'data' **into the root directory** of the repository:


> ⚠️ **Important:** The fingerprint generation scripts depend on the `data/` directory being in the correct place. Do **not** skip this step.

---

## ⚙️ Fingerprint Generation

To generate fingerprints, follow these steps:

### Step 1: Match the data
```bash

python src/fingerprints/MatchData.py

```
### Step 2: Generate fingerprints

```bash
python src/fingerprints/generators/45FP/45FingerprintsGenerator.py

python src/fingerprints/generators/120FP/120FingerprintsGenerator.py

```
