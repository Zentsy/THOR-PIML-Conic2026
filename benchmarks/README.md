# BENCHMARKS OFICIAIS — MODELOS DE DEEP LEARNING & DOWNSCALING CLIMÁTICO
> **Diretório de Baselines Canônicos e Modelos de Referência Internacional**  
> Todos os repositórios abaixo foram selecionados e clonados diretamente de suas publicações oficiais para comparação rigorosa de arquiteturas neurais de downscaling e séries temporais.

---

## 1. Mapeamento dos Repositórios Oficiais

### 1.1. DeepDownscaling / CNN-ESD — Santander Meteorology Group
* **Repositório**: `benchmarks/DeepDownscaling/`
* **URL Oficial**: [https://github.com/SantanderMetGroup/DeepDownscaling](https://github.com/SantanderMetGroup/DeepDownscaling)
* **Autores Principais**: Jorge Baño-Medina, Rodrigo Manzanas, José Manuel Gutiérrez (CSIC / Univ. Cantabria, Espanha).
* **Artigo Canônico**:
  * Baño-Medina, J., Manzanas, R., & Gutiérrez, J. M. (2020). *"Configuration of convolutional neural networks for statistical downscaling of precipitation"*. *Geoscientific Model Development (GMD)*, 13(4), 2109–2124. DOI: [10.5194/gmd-13-2109-2020](https://doi.org/10.5194/gmd-13-2109-2020).
* **Arquitetura**: CNN profunda 2D com convoluções espaciais sobre campos sinóticos da reanálise.

---

### 1.2. NeuralHydrology (ResLSTM — Padrão Ouro em Hidrologia Neural)
* **Repositório**: `benchmarks/neuralhydrology/`
* **URL Oficial**: [https://github.com/neuralhydrology/neuralhydrology](https://github.com/neuralhydrology/neuralhydrology)
* **Autores Principais**: Frederik Kratzert, Daniel Klotz, Grey Nearing, Sepp Hochreiter (JKU Linz / Google Research).
* **Artigos Canônicos**:
  * Kratzert, F., et al. (2018). *"Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks"*. *Hydrology and Earth System Sciences (HESS)*, 22(11), 6005–6022. DOI: [10.5194/hess-22-6005-2018](https://doi.org/10.5194/hess-22-6005-2018).
  * Kratzert, F., et al. (2022). *"NeuralHydrology — A Python library for Deep Learning research in hydrology"*. *Journal of Open Source Software (JOSS)*, 7(71), 4050.
* **Arquitetura**: Long Short-Term Memory (LSTM) com memória contínua de umidade e inércia do solo.

---

### 1.3. Temporal Convolutional Networks (TCN Pura — CMU LocusLab)
* **Repositório**: `benchmarks/TCN/`
* **URL Oficial**: [https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)
* **Autores Principais**: Shaojie Bai, J. Zico Kolter, Vladlen Koltun (Carnegie Mellon University - CMU).
* **Artigo Canônico**:
  * Bai, S., Kolter, J. Z., & Koltun, V. (2018). *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"*. *arXiv preprint arXiv:1803.01271*.
* **Arquitetura**: Convoluções causais dilatadas para captura de ondas atmosféricas de multi-escala (MJO e frentes frias).

---

### 1.4. Empirical Quantile Mapping (EQM — Padrão OMM / IPCC / CORDEX)
* **Repositório**: `benchmarks/scikit-downscale/`
* **URL Oficial**: [https://github.com/pangeo-data/scikit-downscale](https://github.com/pangeo-data/scikit-downscale)
* **Organização**: Pangeo Data / NCAR.
* **Artigo Fundamental**:
  * Gudmundsson, L., et al. (2012). *"Technical Note: Downscaling RCM precipitation to the station scale using statistical transformations – a comparison of methods"*. *Hydrology and Earth System Sciences (HESS)*, 16(9), 3383–3390. DOI: [10.5194/hess-16-3383-2012](https://doi.org/10.5194/hess-16-3383-2012).
* **Método**: Mapeamento não-paramétrico de quantis da CDF empírica.

---

## 2. Estrutura do Diretório

```
benchmarks/
├── README.md               # Documento formal com referências bibliográficas e DOIs
├── setup_benchmarks.sh     # Script automatizado para clone dos repositórios oficiais
├── run_benchmarks.py       # Pipeline de execução e avaliação dos modelos de Deep Learning
├── DeepDownscaling/        # Repositório Oficial do CSIC/Santander (CNN-ESD)
├── neuralhydrology/        # Repositório Oficial de Deep Learning Hidrológico (ResLSTM)
├── TCN/                    # Repositório Oficial de Redes Convolucionais Temporais (CMU)
└── scikit-downscale/       # Repositório Oficial do Pangeo-Data (EQM / CORDEX)
```
