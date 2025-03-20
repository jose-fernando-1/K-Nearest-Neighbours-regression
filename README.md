# 🧱 k-NN Regression — Previsão da Resistência à Compressão do Concreto

Este projeto implementa o algoritmo **k-Nearest Neighbors (k-NN)** para regressão, utilizando apenas as bibliotecas **pandas** e **matplotlib**, aplicado ao dataset **Concrete Compressive Strength** do UCI Machine Learning Repository.

---

## 📊 Sobre o Dataset
- **Fonte:** [Concrete Compressive Strength Dataset — UCI](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)  
- **Atributos de entrada (features):**
  - Cement (kg/m³)
  - Blast Furnace Slag (kg/m³)
  - Fly Ash (kg/m³)
  - Water (kg/m³)
  - Superplasticizer (kg/m³)
  - Coarse Aggregate (kg/m³)
  - Fine Aggregate (kg/m³)
  - Age (dias)  
- **Target:**  
  - Concrete compressive strength (MPa)

---

## 🚀 Como executar o projeto localmente

### 1. Clone o repositório:
```bash
git clone https://github.com/jose-fernando-1/K-Nearest-Neighbours-regression.git
cd K-Nearest-Neighbours-regression
```
### 2. Crie um ambiente virtual:
```bash
python -m venv venv
.\venv\Scripts\activate
```
### 3. Instale as dependências:
```bash
pip install -r requirements.txt
```
### 4. Execute o script:
```bash
python KNN_Regression.py
```