# Satellite Routing

本项目用于研究动态卫星网络中的路由 / 覆盖式遍历问题。

整体训练流程：

```text
生成专家数据
→ 监督学习训练 PolicyNet
→ 使用 SL checkpoint warm start PPO
→ 评估 PPO 模型
```

项目已经改成配置化架构，所有实验参数都通过 `configs/*.yaml` 控制，训练结果统一保存到 `outputs/`。

---

## 1. 项目结构

```text
Satellite_Routing/
  configs/
    expert/
      greedy_clean.yaml
      greedy_failure005.yaml

    sl/
      sl_greedy_clean.yaml
      sl_greedy_failure005.yaml

    ppo/
      ppo_cold_start.yaml
      ppo_warm_start.yaml
      ppo_repeat_penalty_020.yaml
      ppo_delay_scale_010.yaml

    eval/
      eval_best.yaml

  env/
    __init__.py
    orbit_dynamics.py
    topology.py
    sat_env.py
    rl_sat_env.py

  models/
    __init__.py
    gnn_encoder.py
    policy_net.py
    actor_critic_net.py

  data/
    __init__.py
    dataset_sl.py
    generate_expert.py

  trainers/
    __init__.py
    sl_trainer.py
    ppo_trainer.py

  scripts/
    __init__.py
    generate_expert.py
    train_sl.py
    train_ppo.py
    evaluate.py

  utils/
    __init__.py
    config.py
    paths.py
    checkpoint.py
    seed.py
    metrics.py
    logger.py

  tests/
    conftest.py
    test_env.py
    test_model_shape.py
    test_checkpoint.py

  outputs/
    datasets/
    runs/
    checkpoints/

  README.md
  requirements.txt
```

目录职责：

```text
configs/              所有实验配置
scripts/              命令行启动入口
trainers/             训练逻辑
models/               网络结构
env/                  卫星环境、拓扑、轨道动力学
data/                 专家数据生成与数据加载
utils/                配置、路径、checkpoint、日志、指标
tests/                单元测试
outputs/datasets/     专家数据
outputs/runs/         每次训练的完整输出
outputs/checkpoints/  固定 checkpoint 位置
```

注意：

```text
models/ 只放模型结构代码。
训练出来的模型不要放到 models/。
模型权重统一放到 outputs/runs/.../checkpoints/ 或 outputs/checkpoints/。
```

---

## 2. 环境准备

进入项目目录：

```bash
cd ~/Project/Satellite_Routing
```

激活环境：

```bash
conda activate route
```

安装依赖：

```bash
pip install torch numpy networkx pyyaml pytest
```

可以把依赖写入 `requirements.txt`：

```text
torch
numpy
networkx
pyyaml
pytest
```

---

## 3. 运行测试

每次大改后先跑测试：

```bash
pytest tests/ -q
```

如果通过，说明环境、模型、checkpoint、import 基本正常。

---

## 4. 配置文件

### 4.1 专家数据：failure = 0

文件：`configs/expert/greedy_clean.yaml`

```yaml
experiment:
  name: expert_greedy_clean
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.0
  max_link_distance: 10000000.0
  max_steps_factor: 3

expert:
  policy: greedy_shortest_unvisited
  num_episodes: 5000
  chunk_episode_size: 100
  save_dir: outputs/datasets/expert_greedy_clean
```

### 4.2 专家数据：failure = 0.05

文件：`configs/expert/greedy_failure005.yaml`

```yaml
experiment:
  name: expert_greedy_failure005
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.05
  max_link_distance: 10000000.0
  max_steps_factor: 3

expert:
  policy: greedy_shortest_unvisited
  num_episodes: 5000
  chunk_episode_size: 100
  save_dir: outputs/datasets/expert_greedy_failure005
```

### 4.3 监督学习：clean expert

文件：`configs/sl/sl_greedy_clean.yaml`

```yaml
experiment:
  name: sl_greedy_clean
  seed: 42

data:
  train_path: outputs/datasets/expert_greedy_clean

model:
  input_dim: 5
  hidden_dim: 64

sl:
  train_path: outputs/datasets/expert_greedy_clean
  epochs: 25
  batch_size: 512
  lr: 0.002
  num_workers: 0

logging:
  output_root: outputs/runs
  save_latest_every: 1
  save_plot_every: 10
```

### 4.4 监督学习：failure expert

文件：`configs/sl/sl_greedy_failure005.yaml`

```yaml
experiment:
  name: sl_greedy_failure005
  seed: 42

data:
  train_path: outputs/datasets/expert_greedy_failure005

model:
  input_dim: 5
  hidden_dim: 64

sl:
  train_path: outputs/datasets/expert_greedy_failure005
  epochs: 25
  batch_size: 512
  lr: 0.002
  num_workers: 0

logging:
  output_root: outputs/runs
  save_latest_every: 1
  save_plot_every: 10
```

### 4.5 PPO cold start

文件：`configs/ppo/ppo_cold_start.yaml`

```yaml
experiment:
  name: ppo_cold_start
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.05
  max_link_distance: 10000000.0
  max_steps_factor: 3

reward:
  delay_scale: 1.0
  new_node: 0.2
  repeat: -0.1
  fail: -5.0
  success: 10.0

model:
  input_dim: 5
  hidden_dim: 64

ppo:
  pretrained_checkpoint: null
  num_episodes: 4000
  eval_every: 100

  lr: 0.0001
  gamma: 0.99
  k_epochs: 10
  eps_clip: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 1.0

logging:
  output_root: outputs/runs
  save_latest_every: 100
  save_plot_every: 10
```

### 4.6 PPO warm start

文件：`configs/ppo/ppo_warm_start.yaml`

```yaml
experiment:
  name: ppo_warm_start
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.05
  max_link_distance: 10000000.0
  max_steps_factor: 3

reward:
  delay_scale: 1.0
  new_node: 0.2
  repeat: -0.1
  fail: -5.0
  success: 10.0

model:
  input_dim: 5
  hidden_dim: 64

ppo:
  pretrained_checkpoint: outputs/checkpoints/sl_greedy_clean_best.pt
  num_episodes: 4000
  eval_every: 100

  lr: 0.0001
  gamma: 0.99
  k_epochs: 10
  eps_clip: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 1.0

logging:
  output_root: outputs/runs
  save_latest_every: 100
  save_plot_every: 10
```

### 4.7 PPO repeat penalty ablation

文件：`configs/ppo/ppo_repeat_penalty_020.yaml`

```yaml
experiment:
  name: ppo_repeat_penalty_020
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.05
  max_link_distance: 10000000.0
  max_steps_factor: 3

reward:
  delay_scale: 1.0
  new_node: 0.2
  repeat: -0.2
  fail: -5.0
  success: 10.0

model:
  input_dim: 5
  hidden_dim: 64

ppo:
  pretrained_checkpoint: outputs/checkpoints/sl_greedy_clean_best.pt
  num_episodes: 4000
  eval_every: 100

  lr: 0.0001
  gamma: 0.99
  k_epochs: 10
  eps_clip: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 1.0

logging:
  output_root: outputs/runs
  save_latest_every: 100
  save_plot_every: 10
```

### 4.8 PPO delay scale ablation

文件：`configs/ppo/ppo_delay_scale_010.yaml`

```yaml
experiment:
  name: ppo_delay_scale_010
  seed: 42

env:
  num_planes: 6
  sats_per_plane: 10
  failure_prob: 0.05
  max_link_distance: 10000000.0
  max_steps_factor: 3

reward:
  delay_scale: 10.0
  new_node: 0.2
  repeat: -0.1
  fail: -5.0
  success: 10.0

model:
  input_dim: 5
  hidden_dim: 64

ppo:
  pretrained_checkpoint: outputs/checkpoints/sl_greedy_clean_best.pt
  num_episodes: 4000
  eval_every: 100

  lr: 0.0001
  gamma: 0.99
  k_epochs: 10
  eps_clip: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 1.0

logging:
  output_root: outputs/runs
  save_latest_every: 100
  save_plot_every: 10
```

### 4.9 Evaluation

文件：`configs/eval/eval_best.yaml`

```yaml
eval:
  checkpoint: outputs/checkpoints/ppo_best.pt
  num_episodes: 100
  greedy: true
  save_rollouts: false
```

---

## 5. 生成专家数据

### 5.1 生成 clean expert 数据

```bash
python -m scripts.generate_expert --config configs/expert/greedy_clean.yaml
```

输出位置：

```text
outputs/datasets/expert_greedy_clean/
  config.yaml
  metadata.json
  chunk_00000.pkl
  chunk_00001.pkl
  ...
```

检查 metadata：

```bash
cat outputs/datasets/expert_greedy_clean/metadata.json
```

### 5.2 可选：生成 failure expert 数据

```bash
python -m scripts.generate_expert --config configs/expert/greedy_failure005.yaml
```

输出位置：

```text
outputs/datasets/expert_greedy_failure005/
```

---

## 6. 训练监督学习模型

### 6.1 使用 clean expert 训练 SL

```bash
python -m scripts.train_sl --config configs/sl/sl_greedy_clean.yaml
```

输出位置：

```text
outputs/runs/<timestamp>_sl_greedy_clean/
  config.yaml
  checkpoints/
    best.pt
    latest.pt
  logs/
    metrics.csv
```

查看最新 SL run：

```bash
ls -td outputs/runs/*_sl_greedy_clean | head -1
```

固定保存 SL best checkpoint：

```bash
mkdir -p outputs/checkpoints

cp "$(ls -td outputs/runs/*_sl_greedy_clean | head -1)/checkpoints/best.pt" \
   outputs/checkpoints/sl_greedy_clean_best.pt
```

检查：

```bash
ls -lh outputs/checkpoints/sl_greedy_clean_best.pt
```

### 6.2 可选：使用 failure expert 训练 SL

```bash
python -m scripts.train_sl --config configs/sl/sl_greedy_failure005.yaml
```

---

## 7. 训练 PPO

### 7.1 PPO warm start

确保这个文件存在：

```text
outputs/checkpoints/sl_greedy_clean_best.pt
```

运行：

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_warm_start.yaml
```

输出位置：

```text
outputs/runs/<timestamp>_ppo_warm_start/
  config.yaml
  checkpoints/
    best.pt
    latest.pt
  logs/
    metrics.csv
  eval/
```

固定保存 PPO best checkpoint：

```bash
mkdir -p outputs/checkpoints

cp "$(ls -td outputs/runs/*_ppo_warm_start | head -1)/checkpoints/best.pt" \
   outputs/checkpoints/ppo_best.pt
```

检查：

```bash
ls -lh outputs/checkpoints/ppo_best.pt
```

### 7.2 PPO cold start

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_cold_start.yaml
```

### 7.3 PPO repeat penalty 实验

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_repeat_penalty_020.yaml
```

### 7.4 PPO delay scale 实验

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_delay_scale_010.yaml
```

---

## 8. 评估模型

确保这个文件存在：

```text
outputs/checkpoints/ppo_best.pt
```

运行：

```bash
python -m scripts.evaluate --config configs/eval/eval_best.yaml
```

输出：

```text
outputs/runs/<ppo_run_name>/eval/eval_result.json
```

评估指标：

```text
success_rate
mean_steps
mean_coverage_ratio
mean_total_delay
mean_episode_return
mean_repeat_count
mean_repeat_ratio
termination_reasons
```

---

## 9. 完整 warm start 流程

```bash
# 1. 测试
pytest tests/ -q

# 2. 生成 clean expert 数据
python -m scripts.generate_expert --config configs/expert/greedy_clean.yaml

# 3. 训练 SL
python -m scripts.train_sl --config configs/sl/sl_greedy_clean.yaml

# 4. 固定 SL checkpoint
mkdir -p outputs/checkpoints
cp "$(ls -td outputs/runs/*_sl_greedy_clean | head -1)/checkpoints/best.pt" \
   outputs/checkpoints/sl_greedy_clean_best.pt

# 5. PPO warm start
python -m scripts.train_ppo --config configs/ppo/ppo_warm_start.yaml

# 6. 固定 PPO checkpoint
cp "$(ls -td outputs/runs/*_ppo_warm_start | head -1)/checkpoints/best.pt" \
   outputs/checkpoints/ppo_best.pt

# 7. 评估
python -m scripts.evaluate --config configs/eval/eval_best.yaml
```

---

## 10. 修改实验参数

不要改 Python 文件，只改 YAML。

### 修改专家数据数量

```yaml
expert:
  num_episodes: 10000
```

### 修改 SL 训练轮数

```yaml
sl:
  epochs: 50
```

### 修改 PPO 训练轮数

```yaml
ppo:
  num_episodes: 8000
```

### 修改链路故障率

```yaml
env:
  failure_prob: 0.1
```

### 修改重复访问惩罚

```yaml
reward:
  repeat: -0.2
```

### 修改 delay 权重

```yaml
reward:
  delay_scale: 10.0
```

### 修改模型隐藏维度

```yaml
model:
  hidden_dim: 128
```

---

## 11. 运行方式注意事项

推荐使用：

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_warm_start.yaml
```

不要使用：

```bash
python scripts/train_ppo.py
```

原因是 `python -m` 会让 Python 正确识别项目根目录，避免：

```text
ModuleNotFoundError: No module named 'env'
```

---

## 12. 输出文件说明

### 专家数据

```text
outputs/datasets/expert_greedy_clean/
  config.yaml
  metadata.json
  chunk_00000.pkl
  ...
```

### 单次训练 run

```text
outputs/runs/<timestamp>_<experiment_name>/
  config.yaml
  checkpoints/
    best.pt
    latest.pt
  logs/
    metrics.csv
  plots/
  eval/
```

### 固定 checkpoint

```text
outputs/checkpoints/
  sl_greedy_clean_best.pt
  ppo_best.pt
```

---

## 13. 推荐实验顺序

主实验：

```bash
python -m scripts.generate_expert --config configs/expert/greedy_clean.yaml
python -m scripts.train_sl --config configs/sl/sl_greedy_clean.yaml
python -m scripts.train_ppo --config configs/ppo/ppo_warm_start.yaml
python -m scripts.evaluate --config configs/eval/eval_best.yaml
```

对照实验：

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_cold_start.yaml
```

消融实验：

```bash
python -m scripts.train_ppo --config configs/ppo/ppo_repeat_penalty_020.yaml
python -m scripts.train_ppo --config configs/ppo/ppo_delay_scale_010.yaml
```