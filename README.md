下面是按你确认的两项核心改进 **DCF（Device-Conditioned Fusion）+ DCDIR（Device-Conditioned DIR Bank）** 重写后的 `README.md`（从 0 开始可落地、JSON manifest 加载样本、覆盖训练/损失/推理/消融）。你可以直接保存为仓库根目录 `README.md`。

---

# DAME: Device-Aware Multi-Expert with Device-Conditioned Fusion & DIR Bank (ASC)

本项目实现一个面向跨设备声学场景分类（ASC）的高泛化系统 **DAME**（冲顶版），核心贡献聚焦两点：

* **DCF（Device-Conditioned Fusion）**：利用 device 信息对多个 expert 输出进行**条件化门控融合**，并可选做**device-conditioned temperature calibration**。
* **DCDIR（Device-Conditioned DIR Bank）**：构建“设备风格原型库（DIR Bank）”，并通过 device embedding 条件化采样/混合，生成贴合设备链路差异的风格增强。

此外，系统支持 **Stage-1 General → Stage-2 Per-Device** 两阶段训练，推理时 **known device 走 device-specific，unknown 回退 general**。

> 说明：多 backbone、强增强、TTA/ensemble 属于“冲榜必要条件”；本项目把创新聚焦在 **DCF + DCDIR** 的结构化设计上，便于论文化与消融。

---

## 1. 系统概览（你最终要实现的东西）

### 1.1 数据流

1. JSON manifest 读入音频（或特征）+ scene label + device label
2. 特征：log-Mel（推荐用于 PaSST/AST/HTSAT），或 waveform（用于 BEATs）
3. 增强：

   * **DCDIR**：device-conditioned DIR Bank（mel-EQ / FIR / DRC）
   * Freq-MixStyle（可选）
   * SpecAug / time roll（可选）
4. 多 expert 前向：

   * Expert-A：BEATs（waveform） *可选*
   * Expert-B：PaSST（log-Mel） *推荐*
   * Expert-C：CNN（CP-ResNet/EfficientAT 风格）*可选*
5. **DCF** 融合：device-conditioned gating + temperature（可选）
6. 输出：scene logits → 计算损失 → 反向传播

### 1.2 两阶段训练

* **Stage-1（General）**：全设备混合训练，学到通用鲁棒表征与融合规律
* **Stage-2（Per-Device）**：按设备微调（expert adapters/head 或融合器），提升 known device 性能
* 推理：known device 走 stage-2 专用；unknown 回退 stage-1

---

## 2. 仓库结构（建议）

```
dame_asc/
  README.md
  requirements.txt
  configs/
    stage1_general.yaml
    stage2_device.yaml
    infer.yaml
    sweep.json
  data/
    manifests/
      train.json
      val.json
      test.json
      label_map.json
      device_map.json
  src/
    datasets/
      manifest.py
      audio_dataset.py
    features/
      logmel.py
    augment/
      specaug.py
      freq_mixstyle.py
      dcdir_bank.py
    models/
      experts/
        beats_expert.py
        passt_expert.py
        cnn_expert.py
      fusion/
        dcf.py
      dame.py
    losses/
      ce.py
      consistency.py
      reg.py
    train/
      train_stage1.py
      train_stage2.py
      evaluate.py
    infer/
      infer.py
      tta.py
    tools/
      build_device_table.py
      fit_dcf_only.py
      summarize_runs.py
```

---

## 3. 安装与环境

* Python >= 3.9
* PyTorch >= 2.0（推荐 2.1+）
* torchaudio / soundfile / librosa（任选其一读取音频）
* pyyaml, numpy, scipy, tqdm
* 记录：tensorboard 或 wandb（可选）

安装：

```bash
pip install -r requirements.txt
```

---

## 4. 数据：JSON manifest（必须）

### 4.1 推荐格式（dict list）

`data/manifests/train.json`：

```json
[
  {"path":"E:/data/a.wav","scene":3,"device":1,"city":0},
  {"path":"E:/data/b.wav","scene":7,"device":0,"city":2}
]
```

字段：

* `path`: wav/flac 或 npy
* `scene`: 0..C-1
* `device`: 0..D-1；unknown 用 `-1`
* `city`: 可选（仅用于分析）

映射文件：

* `label_map.json`: name→id
* `device_map.json`: name→id（必须包含 `unknown`）

### 4.2 兼容 list-of-lists（Option）

如果你已有：

```json
[["E:/feat/a.npy",0,3,1],["E:/feat/b.npy",2,7,0]]
```

配置 `dataset.manifest_format: "list4"`，含义 `[path, city, scene, device]`。

---

## 5. 输入与特征

### 5.1 log-Mel（推荐）

用于 PaSST/AST/CNN：

```yaml
input:
  type: logmel
  sample_rate: 32000
  n_fft: 1024
  hop_length: 320
  win_length: 1024
  n_mels: 128
  fmin: 20
  fmax: 16000
  to_db: true
```

### 5.2 waveform（Option）

用于 BEATs：

```yaml
input:
  type: waveform
  sample_rate: 16000
  clip_seconds: 10.0
  mono: true
```

---

## 6. DCDIR：Device-Conditioned DIR Bank（核心改进 1）

### 6.1 目标

把“随机 DIR 增强”升级为“**device-conditioned 风格库**”：

* 学一个风格原型集合 ({h_k}_{k=1}^K)
* device embedding (e_d) 生成混合权重 (w(d))
* 得到风格 (h(d)=\sum_k w_k(d)h_k)
* 将风格作用到输入（mel 或 waveform）

### 6.2 实现选项（从易到难）

#### Option A（推荐起步，稳定）：Mel-EQ Bank

* 原型：每个 (h_k) 是长度 `n_mels` 的增益曲线（dB）
* 平滑：卷积平滑核保证“设备频响”平滑
* 作用：`mel_aug = mel + clamp(h(d),[-max_db,+max_db])`

配置：

```yaml
augment:
  dcdir:
    enable: true
    mode: mel_eq_bank
    p: 0.7
    bank_size: 16
    max_db: 6.0
    smooth_kernel: 9
```

#### Option B：Wave-FIR Bank + DRC（更逼真）

* 原型：FIR filters（1D depthwise conv）
* 可选 DRC：轻量压缩器（参数也可条件化）

### 6.3 训练稳定性建议（很重要）

* 先用 Option A 跑通，再上 Option B
* `p` 从 0.3 → 0.7 做 curriculum
* `max_db` 不要超过 9dB（起步 6dB）

---

## 7. Experts：多 backbone（为 DCF 提供互补性）

你至少需要 **2 个差异化 expert** 才能体现 DCF 的价值。

推荐组合：

* Expert-1：PaSST（log-Mel）
* Expert-2：BEATs（waveform）或 CNN（log-Mel）

配置示例：

```yaml
model:
  experts:
    - name: passt
      pretrained: true
      ckpt_path: ""
    - name: cnn
      pretrained: false
```

---

## 8. DCF：Device-Conditioned Fusion（核心改进 2）

### 8.1 动机（落地判定）

如果不同 device 上 expert 强弱不同（排序发生交换），DCF 通常能超过平均 ensemble。

**你必须先输出一张 per-device 表验证**：
`tools/build_device_table.py` 读取各 expert 的预测，输出 `Acc(expert, device)`。

### 8.2 融合形式（推荐）

给定 K 个 expert logits (z_k(x))：

* gating 权重：(\pi(d)=\text{softmax}(g(e_d)))
* temperature（可选）：(T_k(d)=\text{softplus}(t_k(e_d)) + 1)

融合：
[
p(y|x,d)=\sum_{k=1}^K \pi_k(d),\text{softmax}\left(\frac{z_k(x)}{T_k(d)}\right)
]
输出 logits 可用 `log(p)` 或再投影一层（Option）。

配置：

```yaml
model:
  fusion:
    name: dcf
    embed_dim: 128
    hidden: 256
    use_temperature: true
```

### 8.3 DCF 的训练方式（两个选项）

* **Option A（推荐，稳）：端到端 joint training**
  DCF 与 experts 一起训练（或部分冻结）
* **Option B（快速验证）：冻结 experts，只训练 DCF**
  先跑出各 expert logits 存盘，再训练 DCF（几小时内能验证是否有效）

---

## 9. 损失函数（全部写清楚，可落地）

### 9.1 主损失：Scene CE

* 可选 label smoothing

```yaml
loss:
  ce:
    label_smoothing: 0.05
```

### 9.2 一致性损失（推荐，稳定提升）

同一样本 x 与增强版本 x’（尤其 DCDIR）预测一致：
[
L_{cons} = KL(p(y|x)\ |\ p(y|x'))
]
配置：

```yaml
loss:
  consistency:
    enable: true
    weight: 0.5
    type: kl
```

### 9.3 正则（可选）

* gating 熵正则（防止 DCF 过拟合到单一 expert）
* bank 平滑/范数正则（防止 DCDIR 生成不合理曲线）

```yaml
loss:
  reg:
    gate_entropy_weight: 1e-3
    dcdir_l2_weight: 1e-4
```

### 9.4 总损失

[
L = L_{CE} + \lambda_{cons}L_{cons} + \lambda_{reg}L_{reg}
]

> 注意：不再默认引入 CMD/MMD/弱 VAE。它们可以作为对比实验单独加入，但不应与 DCF+DCDIR 主线强耦合。

---

## 10. 训练流程（Stage-1 / Stage-2）

### 10.1 Stage-1：General（必须先跑）

`configs/stage1_general.yaml` 核心字段：

```yaml
train:
  stage: general
  epochs: 120
  batch_size: 64
  optimizer: adamw
  lr: 1e-4
  weight_decay: 1e-2
  warmup_epochs: 10
  ema: true
  ema_decay: 0.999
  freeze:
    experts: false
    fusion: false

augment:
  dcdir: {enable: true, mode: mel_eq_bank, p: 0.7, bank_size: 16, max_db: 6.0, smooth_kernel: 9}
  specaug: {enable: true, p: 0.5, freq_mask_ratio: 0.1, time_mask_ratio: 0.04}
  freq_mixstyle: {enable: true, p: 0.5, alpha: 0.6}

model:
  experts: [ ... ]
  fusion: {name: dcf, embed_dim: 128, hidden: 256, use_temperature: true}

loss:
  ce: {label_smoothing: 0.05}
  consistency: {enable: true, weight: 0.5, type: kl}
  reg: {gate_entropy_weight: 1e-3, dcdir_l2_weight: 1e-4}
```

运行：

```bash
python -m src.train.train_stage1 \
  --config configs/stage1_general.yaml \
  --train_manifest data/manifests/train.json \
  --val_manifest data/manifests/val.json \
  --workdir runs/stage1_general
```

### 10.2 Stage-2：Per-Device（known device 冲分）

目标：在不牺牲 unknown 的情况下提升已知设备。

实现选项：

* **Option A（推荐）**：冻结 experts 主干，仅微调 fusion（以及每 device 的温度/偏置）
* Option B：微调每个 expert 的 adapters/head（更强但风险更大）

配置要点：

```yaml
train:
  stage: device_specific
  epochs: 30
  lr: 3e-4
  freeze:
    experts: true
    fusion: false
  device_specific:
    mode: fusion_only
```

运行单设备：

```bash
python -m src.train.train_stage2 \
  --config configs/stage2_device.yaml \
  --train_manifest data/manifests/train.json \
  --val_manifest data/manifests/val.json \
  --init_ckpt runs/stage1_general/best.ckpt \
  --device_id 0 \
  --workdir runs/stage2_dev0
```

运行所有设备：

```bash
python -m src.train.train_stage2 \
  --config configs/stage2_device.yaml \
  --train_manifest data/manifests/train.json \
  --val_manifest data/manifests/val.json \
  --init_ckpt runs/stage1_general/best.ckpt \
  --all_devices \
  --workdir runs/stage2_all
```

---

## 11. 推理（known 用专用，unknown 回退 general）

`configs/infer.yaml`：

```yaml
infer:
  tta:
    enable: true
    num_crops: 5
    time_shift: true
```

运行：

```bash
python -m src.infer.infer \
  --config configs/infer.yaml \
  --test_manifest data/manifests/test.json \
  --general_ckpt runs/stage1_general/best.ckpt \
  --device_ckpt_dir runs/stage2_all \
  --out_csv runs/preds.csv
```

逻辑：

* 若样本 `device>=0` 且存在对应 stage-2 ckpt → 用 stage-2
* 否则 → 用 stage-1 general

---

## 12. 必做的“有效性自检”（避免再次做无效模块）

### 12.1 DCF 前提验证（必须）

先单独跑每个 expert，输出 per-device acc 表：

```bash
python -m src.tools.build_device_table \
  --val_manifest data/manifests/val.json \
  --expert_ckpts runs/experts_ckpts.json \
  --out_csv runs/per_device_table.csv
```

若不同 device 上 expert 排序交换明显 → DCF 值得做；否则 DCF 预计增益有限（可退化为静态权重或只做温度校准）。

### 12.2 DCDIR 前提验证（建议）

先用 **random mel-EQ**（非条件化）跑一版：

* 如果 random 已带来提升 → 条件化 bank 通常更稳更强
* 如果 random 完全无收益 → 优先检查增强强度/输入尺度/数据切分，而不是直接上 bank

---

## 13. 消融实验矩阵（论文写作友好）

建议最少跑以下对照：

**Fusion 消融：**

1. best single expert
2. avg ensemble
3. static learned weights（不输入 device）
4. **DCF（输入 device）**
5. **DCF + temperature（完整）**

**DIR 消融：**

1. no DIR
2. random DIR-style（随机 EQ）
3. DIR bank（无条件采样）
4. **DCDIR（device-conditioned bank）**

**组合：**

* (4) + (4) 即最终系统（DCF + DCDIR）

---

## 14. Sweep（可选）

`configs/sweep.json` 里可扫：

* `dcdir.bank_size ∈ {8,16,32}`
* `dcdir.max_db ∈ {4,6,8}`
* `fusion.use_temperature ∈ {false,true}`
* `consistency.weight ∈ {0.2,0.5,1.0}`

---

## 15. FAQ（给选项，不阻塞）

### Q: test 没有 device ID？

* Option A：manifest 里 `device=-1` 走 unknown embedding + general
* Option B：实现 device embedding inference（后续扩展，不影响当前 DCF+DCDIR 主线）

### Q: 只想先跑通最小版本？

MVP：2 experts（PaSST + CNN） + DCDIR(mel_eq_bank) + DCF（无 temperature） + CE + consistency。

---

如果你希望我继续把这一版 README 对应的**“详细流程图（包含损失与 Stage-1/Stage-2 训练步骤）”**也生成成最终版本，我可以按这份 README 的结构重新画一张流程图（把 DCF、DCDIR、consistency、entropy reg、stage2 fusion_only 全部标进去），并给你一份可直接放进论文/报告的图注。

---

# 给新手（傻瓜式）说明

下面用最简单的方式说清楚：每个主要文件是干什么的、需要什么输入、怎么跑。

## 1) 这个仓库主要文件做什么

**训练/推理入口（脚本）**

* `src/train/train_stage1.py`：第一阶段训练（通用模型）。
* `src/train/train_stage2.py`：第二阶段训练（按设备微调）。
* `src/infer/infer.py`：推理（有设备 ckpt 就用，没有就回退通用 ckpt）。
* `src/tools/build_device_table.py`：统计“每个设备上每个 expert 的准确率”表。

**核心模块（库代码）**

* `dame_asc/models/experts/`：各个 expert（这里是 numpy MLP 原型，不是 PyTorch）。
* `dame_asc/models/fusion/dcf.py`：DCF 融合（device-conditioned gating）。
* `dame_asc/augment/dcdir_bank.py`：DCDIR（device-conditioned mel EQ bank）。
* `dame_asc/features.py`：把输入样本转成特征（支持 `.npy` 或 sample 自带 features）。
* `dame_asc/data/loader.py`：读取 manifest.json 的样本列表。
* `dame_asc/config.py`：加载配置文件（yaml/json）。

---

## 2) 你需要准备什么输入

**最重要：准备 `.npy` 特征文件（推荐 log-mel）**

每个样本都需要一个 `.npy` 文件，形状建议是 `[T, F]`（时间 * 频率），例如 `[100, 128]`。

manifest 示例：

```json
[
  {"path": "data/features/a.npy", "scene": 3, "device": 1},
  {"path": "data/features/b.npy", "scene": 7, "device": 0}
]
```

字段说明：

* `path`: 指向 `.npy` 文件
* `scene`: 类别 id（0..C-1）
* `device`: 设备 id（0..D-1），unknown 用 `-1`

**注意**

* 如果你不给 `.npy`，系统会退回到“伪随机特征”，训练没有意义。
* 所以真正训练时，请确保 manifest 里每条都有 `.npy`。

---

## 3) 最简单的运行命令（可直接复制）

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 Stage-1 训练（通用）

```bash
python src/train/train_stage1.py \
  --config config.json \
  --train_manifest data/manifests/train.json \
  --workdir runs/stage1_general \
  --epochs 1
```

输出：

* `runs/stage1_general/best.ckpt`

### 3.3 Stage-2 训练（按设备微调）

```bash
python src/train/train_stage2.py \
  --config config.json \
  --train_manifest data/manifests/train.json \
  --init_ckpt runs/stage1_general/best.ckpt \
  --all_devices \
  --workdir runs/stage2_all
```

输出：

* `runs/stage2_all/device_*/best.ckpt`

### 3.4 推理（有 device ckpt 就用，没有就回退 general）

```bash
python src/infer/infer.py \
  --config config.json \
  --test_manifest data/manifests/test.json \
  --general_ckpt runs/stage1_general/best.ckpt \
  --device_ckpt_dir runs/stage2_all \
  --out_csv runs/preds.csv
```

输出：

* `runs/preds.csv`

### 3.5 统计每设备每 expert 表

```bash
python src/tools/build_device_table.py \
  --val_manifest data/manifests/val.json \
  --out_csv runs/per_device_table.csv
```

输出：

* `runs/per_device_table.csv`

---

## 4) 配置文件最小示例（config.json）

```json
{
  "input": {
    "n_mels": 128,
    "n_frames": 100
  },
  "model": {
    "num_classes": 10,
    "experts": [{"name": "passt"}, {"name": "cnn"}],
    "fusion": {"name": "dcf", "num_devices": 16, "embed_dim": 32, "hidden": 64}
  },
  "augment": {
    "dcdir": {
      "enable": true,
      "mode": "mel_eq_bank",
      "p": 0.7,
      "bank_size": 16,
      "max_db": 6.0,
      "smooth_kernel": 9,
      "num_devices": 16
    }
  },
  "loss": {
    "ce": {"label_smoothing": 0.05},
    "consistency": {"enable": true, "weight": 0.5, "type": "mse"}
  },
  "train": {
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-3,
    "freeze": {"experts": false, "fusion": false, "dcdir": false}
  },
  "infer": {
    "tta": {"enable": true, "num_crops": 5, "time_shift": true}
  }
}
```

---

## 5) 常见问题（傻瓜提示）

**Q: 为什么训练出来很差？**

A: 你没有给真实 `.npy` 特征（系统用了伪随机特征）。

**Q: `device=-1` 会怎样？**

A: 会走 unknown device 分支（不会折叠到真实设备）。

**Q: 这是 PyTorch 吗？**

A: 不是。当前是 numpy 手写反传原型，适合验证机制，不是冲榜版。
