# sim-planner-bridge


**Sim-Planner Bridge** is a lightweight orchestration framework for connecting  
🤖 **AI Planners** (e.g., GRID, LLM-based, symbolic) with  
🏠 **3D Simulators** (e.g., VirtualHome, Habitat, AI2-THOR, iGibson, etc.).

It provides a clean interface for running **observe → plan → act** loops across heterogeneous systems,  
handling communication, data translation, and experiment management in a modular and extensible way.

---

## 🚀 Overview

The bridge acts as a **middleware layer**:

- Loads and configures a *planner* (e.g. GRID or custom PyTorch model)  
- Interfaces with a *simulator* (e.g. VirtualHome, Habitat)  
- Translates between symbolic actions, visual observations, and environment feedback  
- Logs, visualizes, and manages experiment runs

---

## 🧩 Key Features

- 🧠 **Planner Abstraction** — Plug in any planning model (neural, symbolic, or rule-based).  
- 🎮 **Simulator Abstraction** — Interface uniformly with Unity-based or Python-based simulators.  
- 🔄 **Interchangeable Modules** — Mix-and-match planners and simulators via configuration.  
- 🧱 **Lightweight Adapters** — Each backend wrapped by a small adapter class.  
- ⚙️ **Configurable Runs** — Control goals, environment IDs, and settings via YAML or CLI.  
- 📊 **Experiment Logging** — Store plan, simulation, and performance data for analysis.  

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/<your-user>/sim-planner-bridge.git
cd sim-planner-bridge

git clone https://github.com/.../GRID external/GRID
git clone https://github.com/.../VirtualHome external/VirtualHome

conda create -n simbridge python=3.10 pytorch torchvision torchaudio
conda activate simbridge

pip install -e external/GRID
pip install -e external/VirtualHome
pip install -e .

# Repository Structure

sim-planner-bridge/
│
├── orchestrator/
│   ├── adapters/
│   │   ├── base.py           # Abstract interfaces for planners/simulators
│   │   ├── planner_grid.py   # GRID planner adapter
│   │   ├── planner_gpt.py    # (example) LLM-based planner
│   │   ├── sim_virtualhome.py # VirtualHome simulator adapter
│   │   └── sim_habitat.py    # (example) Habitat adapter
│   ├── core/
│   │   └── controller.py     # Main observe → plan → act loop
│   └── interfaces/
│       └── types.py          # Common dataclasses (actions, observations)
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── run_episode.py
│   └── eval_batch.py
│
├── tests/
│   ├── test_adapters.py
│   └── test_integration_smoke.py
│
├── pyproject.toml
└── README.md
