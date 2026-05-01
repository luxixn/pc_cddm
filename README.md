# PC-CDDM: PSD-Conditioned Conditional Denoising Diffusion Model

复杂电磁环境下低信噪比微弱 IQ 信号去噪。条件扩散模型 + SNR-aware FiLM 注入 + 时间自适应 PSD 物理约束。

---

## 仓库结构

```
pc_cddm/                 ← git 仓库根
├── configs/             ← 所有 yaml 配置
├── pc_cddm/             ← Python 包源码
│   ├── data/            ← H5 dataset / dataloader
│   ├── models/          ← UNet / FiLM / 条件编码器
│   ├── diffusion/       ← β 调度 / 训练 loss / 反向采样
│   └── utils/           ← PSD / 指标 / 日志
├── notebooks/           ← Kaggle 训练 notebook
├── train.py             ← 训练入口
├── evaluate.py          ← 评估入口
└── requirements.txt
```

---

## 工作流

### 本地开发

**用途**：写代码、跑模块自检、CPU 上小规模 dry run、git 管理。

```bash
# 一次性环境搭建
git clone https://github.com/<你的用户名>/pc_cddm.git
cd pc_cddm
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 模块自检（每个模块开发完都跑一下）
python -m pc_cddm.utils.psd

# 提交代码
git add .
git commit -m "..."
git push origin main
```

**本地不做训练**——CPU 上跑不动，只用来确认代码无 bug。

### Kaggle 训练

**用途**：真实数据集上的训练与评估，必须 GPU。

**首次配置**：

1. 在 Kaggle 上创建 Dataset `pc-cddm-data`，上传 H5 数据集文件。
2. 创建新 Notebook，Settings 里：
   - Accelerator: GPU P100（或 T4 x2）
   - Internet: ON（需手机验证账号）
   - Add Data: 添加上一步的 `pc-cddm-data`

**Notebook 内容**（参考 `notebooks/kaggle_train.ipynb`）：

```python
# Cell 1: 拉取最新代码
!rm -rf /kaggle/working/code
!git clone --depth 1 https://github.com/<你的用户名>/pc_cddm.git /kaggle/working/code

import sys
sys.path.insert(0, '/kaggle/working/code')

# Cell 2: 加载并覆盖配置
import yaml
with open('/kaggle/working/code/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['paths']['data'] = '/kaggle/input/pc-cddm-data/dataset.h5'
cfg['paths']['output_root'] = '/kaggle/working/runs'
cfg['paths']['exp_name'] = 'kaggle_run_v1'

# 续训（首次跳过；从上一个 notebook 版本输出引用）
# cfg['paths']['resume_from'] = '/kaggle/input/<previous-notebook-output>/runs/kaggle_run_v1/ckpt_latest.pt'

# Cell 3: 启动训练
from pc_cddm.train import main
main(cfg)
```

**训练完成 / 会话即将超时**：
1. 训练循环到达 `max_wallclock_hours` 会自动保存 `ckpt_latest.pt` 并优雅退出。
2. 在 Kaggle Notebook 右上角点 **"Save Version"**（选 "Save & Run All" 或 "Quick Save"），输出 `/kaggle/working/` 持久化。
3. 下次新会话，把上一个 version 作为输入引用（Add Data → Notebook Output），通过 `resume_from` 路径续训。

### 评估

```python
from pc_cddm.evaluate import main as eval_main
cfg['paths']['resume_from'] = '/kaggle/working/runs/kaggle_run_v1/ckpt_best.pt'
cfg['eval']['use_ddim'] = False  # 完整 1000 步采样（论文最终结果）
eval_main(cfg)
```

---

## 配置说明

所有超参在 `configs/default.yaml`。**不在代码里硬编码**。

关键超参：
- `diffusion.num_timesteps`: T = 1000
- `diffusion.psd_loss_threshold_ratio`: 0.25 → T_threshold = 250
- `train.lambda_psd`: PSD 损失权重 = 0.1
- `eval.psd_refine_interval`: 推理时每 K=50 步重估 PSD
- `train.max_wallclock_hours`: 8.0 → Kaggle 9h 限制留 1h 余量

---

## 已知约束

- Kaggle 免费 GPU：每周 30 小时配额，单次会话 9 小时上限。
- P100 16GB 显存：batch=32 + AMP 可跑；显存吃紧调 `grad_accum_steps`。
- 完整反向采样（T=1000）评估慢：单样本 ~10s，6k 验证集需分块。

---

## 引用

待论文发表后补充。
